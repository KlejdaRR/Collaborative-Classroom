import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


class HamiltonianStudent(nn.Module):
    """
    Student with Hamiltonian dynamics - implements costate variables
    following Melacci et al. 2024: "A Unified Framework for Neural
    Computation and Learning Over Time" (Equations E⋆1-E⋆4)

    Key theoretical references:
    - Eq. 9: Hamiltonian H = L + z^T · h_dot
    - s = -1 for forward-in-time learning (Section 3)
    - Dissipation term -ηp ensures p(t) → 0
    """

    def __init__(self, num_classes=10, costate_decay=0.995, temporal_horizon=50):
        super(HamiltonianStudent, self).__init__()

        # CNN Architecture (same as SimpleStudent for fair comparison)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # ===== HAMILTONIAN COMPONENTS (Melacci et al. 2024, Eq. E⋆1-E⋆4) =====
        self.costate_decay = costate_decay
        self.temporal_horizon = temporal_horizon

        # Costates as nn.ParameterList for gradient tracking
        self.costates = nn.ParameterList([
            nn.Parameter(torch.randn_like(p) * 0.01)
            for p in self.parameters()
        ])

        # Tracking metrics
        self.last_lagrangian = None
        self.last_hamiltonian = None
        self.step_count = 0
        self.hamiltonian_weight = 0.001  # β in Eq. E⋆2

    def hamiltonian_step(self, x, target, learning_rate=0.001, dt=0.01):
        """
        Hamiltonian update step following Melacci et al. 2024.

        Implements:
        - E⋆1: State evolution (via forward pass)
        - E⋆2: Weight evolution: θ = -β ⊙ ω
        - E⋆3-E⋆4: Costate evolution with s = -1 (forward time)

        The dissipation term (-ηp) ensures costates decay to zero,
        satisfying the boundary condition p(T) → 0.
        """
        self.train()
        self.step_count += 1

        # ===== Forward pass =====
        output = self(x)
        loss = F.nll_loss(output, target)
        self.last_lagrangian = loss.item()

        # ===== Compute gradients of loss w.r.t parameters =====
        self.zero_grad()
        loss.backward(retain_graph=True)

        # Store gradients (cloned to avoid modification issues)
        param_grads = []
        for param in self.parameters():
            param_grads.append(param.grad.clone() if param.grad is not None else None)

        # ===== Compute Hamiltonian: H' = L + Σ(p_i * ∂L/∂w_i) =====
        # (Melacci et al. 2024, Eq. 9)
        grads_with_graph = torch.autograd.grad(
            loss, self.parameters(),
            create_graph=True, retain_graph=True, allow_unused=True
        )

        hamiltonian = loss.clone()
        for costate, grad in zip(self.costates, grads_with_graph):
            if grad is not None:
                hamiltonian = hamiltonian + self.hamiltonian_weight * torch.sum(costate * grad)

        self.last_hamiltonian = hamiltonian.item()

        # ===== Update parameters using stored gradients =====
        # (Weight evolution: θ = -β ⊙ ω)
        with torch.no_grad():
            for param, grad in zip(self.parameters(), param_grads):
                if grad is not None:
                    # Find matching costate (by shape)
                    for costate in self.costates:
                        if costate.shape == param.shape:
                            # β = learning_rate, ω = costate
                            param_update = -learning_rate * (grad + self.hamiltonian_weight * costate)
                            param.add_(param_update)
                            break

        # ===== Update costates using Hamiltonian gradients =====
        # (Costate evolution with s = -1 for forward time)
        hamiltonian_grads = torch.autograd.grad(
            hamiltonian, self.costates,
            retain_graph=False, allow_unused=True
        )

        with torch.no_grad():
            for costate, grad in zip(self.costates, hamiltonian_grads):
                if grad is not None:
                    # Costate update: p ← p - dt * ∂H/∂p - η·p·dt
                    # The dissipation term (-ηp) ensures p(t) → 0
                    costate_update = -dt * grad - (1 - self.costate_decay) * costate
                    costate.add_(costate_update)

        return output

    def forward(self, x):
        """Standard forward pass"""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    def get_metrics(self):
        """Return Hamiltonian metrics for monitoring"""
        avg_costate_norm = 0.0
        if len(self.costates) > 0:
            norms = [torch.norm(c).item() for c in self.costates]
            avg_costate_norm = sum(norms) / len(norms)

        return {
            'lagrangian': self.last_lagrangian if self.last_lagrangian is not None else 0.0,
            'hamiltonian': self.last_hamiltonian if self.last_hamiltonian is not None else 0.0,
            'avg_costate_norm': avg_costate_norm,
            'step': self.step_count
        }

    def to_device(self, device):
        """Move model and costates to device"""
        self.to(device)
        for i, costate in enumerate(self.costates):
            if costate.device != device:
                self.costates[i] = nn.Parameter(costate.to(device))
        return self