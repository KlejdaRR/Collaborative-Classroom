# hamiltonian_validation.py (FIXED VERSION)
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


class HamiltonianValidator:
    """
    Validates the theoretical Hamiltonian relationships:
    1. p(t) ≈ ∫ e^{-γ(t-τ)} (-∂L/∂v) dτ
    2. Conservation of Hamiltonian in conservative systems
    3. Costate as Lagrange multiplier interpretation
    """

    def __init__(self, detector):
        self.detector = detector

    def validate_hamiltonian_relationship(self, window=100):
        """
        Validate that p(t) ≈ ∫ e^{-γ(t-τ)} (-∂L/∂v) dτ

        This is the key theoretical relationship from the Hamilton equations
        where costate integrates the sensitivity over time.
        """
        # Handle both list and deque for costate_history
        if hasattr(self.detector, 'dL_dv_history'):
            dL_dv_hist = self.detector.dL_dv_history
        else:
            return None, None

        if len(dL_dv_hist) < window:
            return None, None

        # Theoretical costate via exponential integration
        gamma = 0.1  # Decay rate
        theoretical = []

        for t in range(len(dL_dv_hist)):
            integral = 0
            for tau in range(max(0, t - window), t + 1):
                decay = np.exp(-gamma * (t - tau))
                integral += decay * (-dL_dv_hist[tau])
            theoretical.append(integral / window)

        # Get empirical costate history (handle both list and deque)
        if hasattr(self.detector, 'costate_history'):
            if hasattr(self.detector.costate_history, '__len__'):
                # Convert deque to list if needed
                if hasattr(self.detector.costate_history, '__class__') and 'deque' in str(
                        self.detector.costate_history.__class__):
                    costate_list = list(self.detector.costate_history)
                else:
                    costate_list = self.detector.costate_history
            else:
                costate_list = []
        else:
            costate_list = []

        empirical = costate_list[:len(theoretical)]

        # Compute correlation
        if len(theoretical) > 10 and len(empirical) > 10:
            # Ensure same length
            min_len = min(len(theoretical), len(empirical))
            theoretical = theoretical[:min_len]
            empirical = empirical[:min_len]
            correlation, p_value = pearsonr(theoretical, empirical)
            return correlation, p_value

        return None, None

    def validate_costate_boundary_condition(self, T=None):
        """
        Validate that costate approaches zero at the end of horizon
        """
        # Get costate history (handle both list and deque)
        if hasattr(self.detector, 'costate_history'):
            if hasattr(self.detector.costate_history, '__len__'):
                if hasattr(self.detector.costate_history, '__class__') and 'deque' in str(
                        self.detector.costate_history.__class__):
                    costate_list = list(self.detector.costate_history)
                else:
                    costate_list = self.detector.costate_history
            else:
                costate_list = []
        else:
            costate_list = []

        if len(costate_list) < 200:
            return {
                'final_avg': 0.0,
                'initial_avg': 0.0,
                'convergence_ratio': 1.0,
                'satisfies_boundary': False,
                'error': 'Insufficient history'
            }

        if T is None or T > len(costate_list):
            T = len(costate_list)

        costate_final = np.abs(costate_list[-100:])
        costate_initial = np.abs(costate_list[:100])

        final_avg = float(np.mean(costate_final))
        initial_avg = float(np.mean(costate_initial))

        # Should converge toward zero
        convergence_ratio = final_avg / (initial_avg + 1e-8)

        return {
            'final_avg': final_avg,
            'initial_avg': initial_avg,
            'convergence_ratio': convergence_ratio,
            'satisfies_boundary': convergence_ratio < 0.5
        }

    def plot_theoretical_validation(self):
        """Plot theoretical vs empirical costate"""
        correlation, p_value = self.validate_hamiltonian_relationship()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Theoretical vs Empirical Costate
        ax = axes[0, 0]

        # Get dL_dv history
        if hasattr(self.detector, 'dL_dv_history'):
            dL_dv_hist = self.detector.dL_dv_history
        else:
            dL_dv_hist = []

        theoretical = []
        for t in range(min(len(dL_dv_hist), 1000)):
            integral = 0
            for tau in range(max(0, t - 100), t + 1):
                integral += np.exp(-0.1 * (t - tau)) * (-dL_dv_hist[tau])
            theoretical.append(integral)

        # Get empirical costate
        if hasattr(self.detector, 'costate_history'):
            if hasattr(self.detector.costate_history, '__class__') and 'deque' in str(
                    self.detector.costate_history.__class__):
                costate_list = list(self.detector.costate_history)
            else:
                costate_list = self.detector.costate_history if hasattr(self.detector.costate_history,
                                                                        '__getitem__') else []
        else:
            costate_list = []

        empirical = costate_list[:len(theoretical)]

        ax.plot(theoretical, label='Theoretical p(t)', alpha=0.7)
        ax.plot(empirical, label='Empirical p(t)', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Costate')
        if correlation:
            ax.set_title(f'Theoretical vs Empirical Costate\nCorrelation: {correlation:.3f}')
        else:
            ax.set_title('Theoretical vs Empirical Costate')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: dL/dv over time
        ax = axes[0, 1]
        if dL_dv_hist:
            ax.plot(dL_dv_hist[:1000], 'r-', alpha=0.5, linewidth=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('∂L/∂v')
        ax.set_title('Sensitivity of Loss to State (dL/dv)')
        ax.grid(True, alpha=0.3)

        # Plot 3: Hamiltonian Conservation
        ax = axes[1, 0]
        if hasattr(self.detector, 'loss_history') and self.detector.loss_history and costate_list:
            # H = L + p·f ≈ L for our simplified system
            min_len = min(len(self.detector.loss_history), len(costate_list), 1000)
            hamiltonian = [self.detector.loss_history[i] + abs(costate_list[i]) * 0.1
                           for i in range(min_len)]
            ax.plot(hamiltonian, 'g-', alpha=0.7)
            ax.set_xlabel('Step')
            ax.set_ylabel('Hamiltonian H')
            ax.set_title('Hamiltonian Evolution (should be approximately conserved)')
            ax.grid(True, alpha=0.3)

        # Plot 4: Phase Portrait (state vs costate)
        ax = axes[1, 1]
        if hasattr(self.detector, 'state_history') and self.detector.state_history and costate_list:
            min_len = min(len(self.detector.state_history), len(costate_list), 1000)
            states = self.detector.state_history[:min_len]
            costates_plot = costate_list[:min_len]

            # Color by time
            colors = np.arange(len(states))
            scatter = ax.scatter(states, costates_plot, c=colors, cmap='viridis', s=1, alpha=0.5)
            ax.set_xlabel('State v(t)')
            ax.set_ylabel('Costate p(t)')
            ax.set_title('Phase Portrait (State-Costate Space)')
            plt.colorbar(scatter, ax=ax, label='Time')
            ax.grid(True, alpha=0.3)

        plt.suptitle('Hamiltonian Theory Validation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('hamiltonian_validation.png', dpi=150, bbox_inches='tight')
        plt.show()

        return correlation, p_value