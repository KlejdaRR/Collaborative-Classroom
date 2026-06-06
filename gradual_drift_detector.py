# gradual_drift_detector.py (FIXED VERSION)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque


class HamiltonianGradualDriftDetector(nn.Module):
    """
    Enhanced Hamiltonian detector for gradual concept drift.

    Gradual drifts require:
    1. Slower adaptation rate
    2. Multi-scale costate monitoring
    3. Persistent state tracking
    """

    def __init__(self, input_dim=2, hidden_dim=64, costate_decay=0.99):
        super(HamiltonianGradualDriftDetector, self).__init__()

        # Store input_dim
        self.input_dim = input_dim

        # Multi-scale network architecture - FIXED: input_dim + 2
        self.net = nn.Sequential(
            nn.Linear(input_dim + 2, hidden_dim),  # +2 for state + costate history
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.costate_decay = costate_decay

        # State and costate with history
        self.register_buffer('state', torch.tensor(0.5))
        self.register_buffer('costate', torch.tensor(0.0))

        # Multi-scale costate tracking for gradual drift
        self.costate_history = deque(maxlen=10000)
        self.costate_slow = 0.0  # Long-term average
        self.costate_fast = 0.0  # Short-term average

        # Drift detection thresholds
        self.gradual_threshold = 0.3  # For gradual drift
        self.sudden_threshold = 0.8  # For sudden drift

        # Tracking metrics
        self.drift_confidence = 0.0
        self.drift_type = None
        self.state_history = []
        self.loss_history = []
        self.dL_dv_history = []

    def forward(self, x):
        """Forward pass with costate history"""
        # Ensure x is 2D
        if x.dim() == 0:
            x = x.unsqueeze(0)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.size(0)

        # Include state and recent costate trend
        state_tensor = self.state.expand(batch_size, 1)
        costate_trend = torch.tensor(self.get_costate_trend()).float().expand(batch_size, 1)

        # Move to same device as x
        if x.device != state_tensor.device:
            state_tensor = state_tensor.to(x.device)
            costate_trend = costate_trend.to(x.device)

        combined = torch.cat([x, state_tensor, costate_trend], dim=1)

        # Ensure combined has correct dimension
        expected_dim = self.input_dim + 2
        if combined.shape[1] != expected_dim:
            # Pad or trim if needed
            if combined.shape[1] < expected_dim:
                padding = torch.zeros(batch_size, expected_dim - combined.shape[1]).to(x.device)
                combined = torch.cat([combined, padding], dim=1)
            else:
                combined = combined[:, :expected_dim]

        return self.net(combined).squeeze()

    def get_costate_trend(self):
        """Calculate trend in costate for drift characterization"""
        if len(self.costate_history) < 100:
            return 0.0

        recent = np.mean(list(self.costate_history)[-50:])
        older = np.mean(list(self.costate_history)[-200:-50])

        # Trend: positive = increasing drift, negative = stabilizing
        return (recent - older) / (abs(older) + 1e-8)

    def update_costate_stats(self):
        """Update multi-scale costate statistics"""
        recent_vals = list(self.costate_history)[-100:] if len(self.costate_history) >= 100 else list(
            self.costate_history)

        if recent_vals:
            self.costate_fast = 0.9 * self.costate_fast + 0.1 * np.mean(np.abs(recent_vals))
            self.costate_slow = 0.995 * self.costate_slow + 0.005 * np.mean(np.abs(recent_vals))

    def detect_drift_type(self):
        """Determine if drift is gradual or sudden based on costate dynamics"""
        if len(self.costate_history) < 200:
            return "none"  # Changed from "insufficient_data" to avoid errors

        # Calculate volatility
        recent = list(self.costate_history)[-100:]
        volatility = np.std(recent) if len(recent) > 0 else 0

        # Sudden drift: high peak, low volatility before/after
        max_peak = np.max(np.abs(recent)) if recent else 0

        if max_peak > self.sudden_threshold and volatility < 0.2:
            return "sudden"
        elif self.costate_fast > self.gradual_threshold and volatility > 0.1:
            return "gradual"
        else:
            return "none"

    def compute_dL_dv(self, x, y_true):
        """Compute sensitivity to state changes"""
        if x.dim() == 1:
            x = x.unsqueeze(0)

        state_var = self.state.clone().detach().requires_grad_(True)
        batch_size = x.size(0)
        state_tensor = state_var.expand(batch_size, 1)
        costate_trend = torch.tensor(self.get_costate_trend()).float().expand(batch_size, 1)

        # Move to same device
        if x.device != state_tensor.device:
            state_tensor = state_tensor.to(x.device)
            costate_trend = costate_trend.to(x.device)

        combined = torch.cat([x, state_tensor, costate_trend], dim=1)

        # Ensure correct dimension
        expected_dim = self.input_dim + 2
        if combined.shape[1] != expected_dim:
            if combined.shape[1] < expected_dim:
                padding = torch.zeros(batch_size, expected_dim - combined.shape[1]).to(x.device)
                combined = torch.cat([combined, padding], dim=1)
            else:
                combined = combined[:, :expected_dim]

        pred = self.net(combined).squeeze()
        loss = F.binary_cross_entropy(pred, y_true.float())

        return torch.autograd.grad(loss, state_var, create_graph=False)[0].item()

    def hamiltonian_step(self, x, y_true, learning_rate=0.005, dt=0.01):
        """Hamiltonian update with gradual drift adaptation"""

        # Ensure correct dimensions
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if hasattr(y_true, 'dim'):
            y_true = y_true.squeeze()

        # Forward pass
        pred = self.forward(x)
        if pred.dim() == 1 and pred.size(0) == 1:
            pred = pred.squeeze()

        loss = F.binary_cross_entropy(pred, y_true.float())

        # 1. Update network parameters (slower for gradual drift)
        self.zero_grad()
        loss.backward()

        with torch.no_grad():
            for param in self.parameters():
                if param.grad is not None:
                    # Adaptive learning rate based on drift type
                    drift_type = self.detect_drift_type()
                    if drift_type == "gradual":
                        lr = learning_rate * 0.5  # Slower adaptation
                    else:
                        lr = learning_rate

                    torch.nn.utils.clip_grad_norm_([param], max_norm=0.5)
                    param.add_(-lr * param.grad)

        # 2. Compute ∂L/∂v
        dL_dv = self.compute_dL_dv(x, y_true)
        dL_dv = np.clip(dL_dv, -0.5, 0.5)

        # 3. Update costate with adaptive decay
        with torch.no_grad():
            # Adaptive decay based on drift type
            drift_type = self.detect_drift_type()
            if drift_type == "gradual":
                decay = self.costate_decay * 0.995  # Slower decay for gradual drift
            elif drift_type == "sudden":
                decay = self.costate_decay * 0.95  # Faster reset after sudden drift
            else:
                decay = self.costate_decay

            # Gradual drift needs costate accumulation
            if drift_type == "gradual":
                self.costate.data = self.costate.data * decay - dt * dL_dv * 0.5
            else:
                self.costate.data = self.costate.data * decay - dt * dL_dv

        # 4. Update state with confidence weighting
        with torch.no_grad():
            pred_mean = pred.mean().item()
            y_mean = y_true.float().mean().item() if hasattr(y_true, 'mean') else y_true
            confidence = 1.0 - abs(pred_mean - y_mean)

            # Adaptive update rate
            if drift_type == "gradual":
                alpha = 0.05 + 0.2 * (1 - confidence)
            else:
                alpha = 0.1 + 0.5 * (1 - confidence)

            self.state.data = (1 - alpha) * self.state.data + alpha * pred_mean
            self.state.data = torch.clamp(self.state.data, 0.0, 1.0)

        # 5. Record and update statistics
        self.costate_history.append(self.costate.item())
        self.state_history.append(self.state.item())
        self.loss_history.append(loss.item())
        self.dL_dv_history.append(dL_dv)
        self.update_costate_stats()

        return loss.item(), pred