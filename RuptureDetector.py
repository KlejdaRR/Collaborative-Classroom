import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HamiltonianRuptureDetector(nn.Module):
    """
    Hamiltonian rupture detector optimized for THU dataset.
    """

    def __init__(self, input_dim=2, hidden_dim=64, costate_decay=0.98):
        super(HamiltonianRuptureDetector, self).__init__()

        # Larger network for THU dataset complexity
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.input_dim = input_dim

        # Initialize with small random weights for better learning
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

        self.net.apply(init_weights)

        # State and costate
        self.register_buffer('state', torch.tensor(0.5))
        self.register_buffer('costate', torch.tensor(0.0))

        self.costate_decay = costate_decay

        # Tracking
        self.costate_history = []
        self.state_history = []
        self.loss_history = []
        self.dL_dv_history = []

    def forward(self, x):
        """Forward pass: compute rupture probability from input + current state"""
        # Ensure x is 2D: [batch_size, input_dim]
        if x.dim() == 0:
            x = x.unsqueeze(0)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.size(0)
        state_tensor = self.state.expand(batch_size, 1)
        combined = torch.cat([x, state_tensor], dim=1)

        return self.net(combined).squeeze()

    def compute_dL_dv(self, x, y_true):
        """
        Compute ∂L/∂v using automatic differentiation.
        Returns the sensitivity of loss to changes in state.
        """
        # Ensure x is 2D
        if x.dim() == 0:
            x = x.unsqueeze(0)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Create a differentiable version of state
        state_var = self.state.clone().detach().requires_grad_(True)

        # Forward pass with state_var
        batch_size = x.size(0)
        state_tensor = state_var.expand(batch_size, 1)
        combined = torch.cat([x, state_tensor], dim=1)
        pred = self.net(combined).squeeze()

        # Compute loss
        loss = F.binary_cross_entropy(pred, y_true.float())

        # Compute gradient with respect to state
        dL_dv = torch.autograd.grad(loss, state_var, create_graph=False)[0]

        return dL_dv.item()

    def hamiltonian_step(self, x, y_true, learning_rate=0.01, dt=0.01):
        """One Hamiltonian learning step."""

        # Ensure correct dimensions
        if x.dim() == 0:
            x = x.unsqueeze(0)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Ensure y_true is scalar
        if hasattr(y_true, 'dim'):
            y_true = y_true.squeeze()

        # Forward pass
        pred = self.forward(x)

        # Ensure pred has same shape as y_true
        if pred.dim() == 1 and pred.size(0) == 1:
            pred = pred.squeeze()

        loss = F.binary_cross_entropy(pred, y_true.float())

        # ===== 1. Update network parameters =====
        self.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in self.parameters():
                if param.grad is not None:
                    # Clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_([param], max_norm=1.0)
                    param.add_(-learning_rate * param.grad)

        # ===== 2. Compute ∂L/∂v more accurately =====
        dL_dv = self.compute_dL_dv(x, y_true)

        # Clip to reasonable range
        dL_dv = np.clip(dL_dv, -1.0, 1.0)

        # ===== 3. Update costate (RUPTURE SIGNAL!) =====
        with torch.no_grad():
            # Reduce decay when drift is detected (allows faster response)
            current_drift_signal = abs(dL_dv)
            adaptive_decay = self.costate_decay

            # If we detect a potential drift, decay less (preserve the signal)
            if current_drift_signal > 0.5:  # Threshold for drift detection
                adaptive_decay = min(0.99, self.costate_decay * 1.1)

            self.costate.data = (
                    self.costate.data * adaptive_decay
                    - dt * dL_dv * (1.0 + current_drift_signal)  # Amplify during drifts
            )

        # Add costate clipping to prevent numerical issues
        self.costate.data = torch.clamp(self.costate.data, -1.0, 1.0)

        # ===== 4. Update state =====
        with torch.no_grad():
            pred_mean = pred.mean().item()
            # Adaptive state update based on prediction error
            error = abs(pred_mean - y_true.float().mean().item())
            alpha = 0.1 + 0.5 * error  # Update faster when error is high
            self.state.data = (1 - alpha) * self.state.data + alpha * pred_mean
            # Keep state in [0,1]
            self.state.data = torch.clamp(self.state.data, 0.0, 1.0)

        # ===== 5. Record history =====
        self.costate_history.append(self.costate.item())
        self.state_history.append(self.state.item())
        self.loss_history.append(loss.item())
        self.dL_dv_history.append(dL_dv)

        return loss.item(), pred

    def detect_rupture(self, threshold=0.005):
        """Simpler rupture detection based on absolute costate value"""
        if len(self.costate_history) < 20:
            return False

        # Recent costate magnitude
        recent = np.mean(np.abs(self.costate_history[-10:]))

        # Return True if costate magnitude exceeds threshold
        return recent > threshold

    def get_metrics(self):
        return {
            'state': self.state.item(),
            'costate': self.costate.item(),
            'avg_costate': np.mean(np.abs(self.costate_history[-100:])) if self.costate_history else 0,
            'avg_dL_dv': np.mean(np.abs(self.dL_dv_history[-100:])) if self.dL_dv_history else 0
        }