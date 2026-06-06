# hamiltonian_validation.py
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
        self.theoretical_costate = []
        self.empirical_costate = []

    def validate_hamiltonian_relationship(self, window=100):
        """
        Validate that p(t) ≈ ∫ e^{-γ(t-τ)} (-∂L/∂v) dτ

        This is the key theoretical relationship from the Hamilton equations
        where costate integrates the sensitivity over time.
        """
        if len(self.detector.dL_dv_history) < window:
            return None, None

        # Theoretical costate via exponential integration
        gamma = 0.1  # Decay rate
        theoretical = []

        for t in range(len(self.detector.dL_dv_history)):
            integral = 0
            for tau in range(max(0, t - window), t + 1):
                decay = np.exp(-gamma * (t - tau))
                integral += decay * (-self.detector.dL_dv_history[tau])
            theoretical.append(integral / window)

        empirical = self.detector.costate_history[:len(theoretical)]

        # Compute correlation
        if len(theoretical) > 10 and len(empirical) > 10:
            correlation, p_value = pearsonr(theoretical, empirical)
            return correlation, p_value

        return None, None

    def validate_costate_boundary_condition(self, T=None):
        """
        Validate that costate approaches zero at the end of horizon
        """
        if T is None or T > len(self.detector.costate_history):
            T = len(self.detector.costate_history)

        costate_final = np.abs(self.detector.costate_history[-100:])
        costate_initial = np.abs(self.detector.costate_history[:100])

        final_avg = np.mean(costate_final)
        initial_avg = np.mean(costate_initial)

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
        theoretical = []
        for t in range(len(self.detector.dL_dv_history)):
            integral = 0
            for tau in range(max(0, t - 100), t + 1):
                integral += np.exp(-0.1 * (t - tau)) * (-self.detector.dL_dv_history[tau])
            theoretical.append(integral)

        ax.plot(theoretical[:1000], label='Theoretical p(t)', alpha=0.7)
        ax.plot(self.detector.costate_history[:1000], label='Empirical p(t)', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Costate')
        ax.set_title(
            f'Theoretical vs Empirical Costate\nCorrelation: {correlation:.3f}' if correlation else 'Theoretical vs Empirical Costate')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: dL/dv over time
        ax = axes[0, 1]
        ax.plot(self.detector.dL_dv_history[:1000], 'r-', alpha=0.5, linewidth=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('∂L/∂v')
        ax.set_title('Sensitivity of Loss to State (dL/dv)')
        ax.grid(True, alpha=0.3)

        # Plot 3: Hamiltonian Conservation
        ax = axes[1, 0]
        if hasattr(self.detector, 'loss_history'):
            # H = L + p·f ≈ L for our simplified system
            hamiltonian = [self.detector.loss_history[i] +
                           abs(self.detector.costate_history[i]) * 0.1
                           for i in range(min(1000, len(self.detector.loss_history)))]
            ax.plot(hamiltonian, 'g-', alpha=0.7)
            ax.set_xlabel('Step')
            ax.set_ylabel('Hamiltonian H')
            ax.set_title('Hamiltonian Evolution (should be approximately conserved)')
            ax.grid(True, alpha=0.3)

        # Plot 4: Phase Portrait (state vs costate)
        ax = axes[1, 1]
        states = self.detector.state_history[:1000]
        costates = self.detector.costate_history[:1000]

        # Color by time
        colors = np.arange(len(states))
        scatter = ax.scatter(states, costates, c=colors, cmap='viridis', s=1, alpha=0.5)
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


# Enhanced experiment runner for gradual drift
def run_gradual_drift_experiment():
    """Test gradual drift detection"""
    from thu_drift_loader import create_sequential_stream
    from gradual_drift_detector import HamiltonianGradualDriftDetector

    # Load gradual drift dataset
    dataset_name = "CNNS_Nonlinear_Gradual_ChocolateRotation"

    try:
        loader, rupture_positions, drift_type = create_sequential_stream(
            dataset_name=dataset_name,
            batch_size=1,
            return_rupture_positions=True
        )
    except:
        print(f"Gradual dataset not found, using sudden with different settings")
        loader, rupture_positions, drift_type = create_sequential_stream(
            dataset_name="CNNS_Nonlinear_Sudden_ChocolateRotation",
            batch_size=1,
            return_rupture_positions=True
        )
        drift_type = "gradual_simulated"

    sample_X, sample_Y, _ = next(iter(loader))
    input_dim = sample_X.shape[1]

    # Use gradual drift detector
    detector = HamiltonianGradualDriftDetector(
        input_dim=input_dim,
        hidden_dim=64,
        costate_decay=0.99
    )

    # Process stream
    costates = []
    drift_types_detected = []

    for batch_idx, (X, Y, idx) in enumerate(loader):
        X = X.squeeze()
        Y = Y.squeeze()

        loss, pred = detector.hamiltonian_step(X, Y, learning_rate=0.005, dt=0.01)
        costates.append(detector.costate.item())

        # Detect drift type periodically
        if batch_idx % 1000 == 0:
            drift_type_detected = detector.detect_drift_type()
            drift_types_detected.append((batch_idx, drift_type_detected))

    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    ax = axes[0]
    ax.plot(costates, 'purple', linewidth=0.5)
    ax.set_ylabel('Costate p(t)')
    ax.set_title(f'Gradual Drift Detection - Type: {drift_type}')
    ax.grid(True, alpha=0.3)

    # Mark drift positions
    if rupture_positions:
        for pos in rupture_positions:
            ax.axvline(x=pos, color='red', linestyle='--', alpha=0.5)

    ax = axes[1]
    # Moving average to show trend
    window = 500
    ma = np.convolve(costates, np.ones(window) / window, mode='valid')
    ax.plot(ma, 'orange', linewidth=1)
    ax.set_ylabel('Costate (Moving Avg)')
    ax.set_title('Trend for Gradual Drift Detection')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if drift_types_detected:
        steps, types = zip(*drift_types_detected)
        type_codes = [0 if t == 'none' else (1 if t == 'gradual' else 2) for t in types]
        ax.scatter(steps, type_codes, c=type_codes, cmap='RdYlGn', s=20)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['None', 'Gradual', 'Sudden'])
        ax.set_ylabel('Detected Drift Type')
        ax.set_xlabel('Step')
        ax.set_title('Drift Type Classification Over Time')

    plt.tight_layout()
    plt.savefig('gradual_drift_detection.png', dpi=150, bbox_inches='tight')
    plt.show()

    return detector, costates


# Run multi-agent experiment
def run_multi_agent_experiment():
    """Test multi-agent collaborative drift detection"""
    from thu_drift_loader import create_sequential_stream
    from multi_agent_drift_detector import MultiAgentHamiltonianDetector

    loader, rupture_positions, drift_type = create_sequential_stream(
        dataset_name="CNNS_Nonlinear_Sudden_ChocolateRotation",
        batch_size=1,
        return_rupture_positions=True
    )

    sample_X, sample_Y, _ = next(iter(loader))
    input_dim = sample_X.shape[1]

    # Create multi-agent system
    mas = MultiAgentHamiltonianDetector(input_dim=input_dim, num_agents=7)

    # Track results
    drift_signals = []
    drift_detections = []
    agent_diversity = []

    for batch_idx, (X, Y, idx) in enumerate(loader):
        X = X.squeeze()
        Y = Y.squeeze()

        result = mas.detect_drift_ensemble(X, Y, learning_rate=0.005, dt=0.01)

        drift_signals.append(result['drift_signal'])
        drift_detections.append(1 if result['drift_detected'] else 0)

        if batch_idx % 500 == 0:
            diversity = mas.get_agent_diversity_metrics()
            agent_diversity.append((batch_idx, diversity['diversity']))

            if result['drift_detected']:
                print(
                    f"Step {batch_idx}: DRIFT DETECTED ({result['drift_type']}) - Signal: {result['drift_signal']:.3f}")

    # Plot multi-agent results
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    ax = axes[0]
    ax.plot(drift_signals, 'orange', linewidth=0.5)
    if rupture_positions:
        for pos in rupture_positions:
            ax.axvline(x=pos, color='red', linestyle='--', alpha=0.5, label='True Drift')
    ax.set_ylabel('Drift Signal')
    ax.set_title('Multi-Agent Ensemble Drift Signal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(drift_detections, 'g-', linewidth=0.5)
    ax.set_ylabel('Detection')
    ax.set_title('Binary Drift Detections')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if agent_diversity:
        steps, divs = zip(*agent_diversity)
        ax.plot(steps, divs, 'purple', marker='o', markersize=3)
    ax.set_ylabel('Agent Diversity')
    ax.set_title('Costate Diversity Across Agents (High = Drift)')
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    # Costate trajectories of first 3 agents
    for i in range(min(3, mas.num_agents)):
        costates = [h['agent_costates'][i] for h in mas.detection_history[-1000:]]
        ax.plot(costates, label=f'Agent {i}', alpha=0.7)
    ax.set_xlabel('Step (last 1000)')
    ax.set_ylabel('|Costate|')
    ax.set_title('Individual Agent Costate Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Multi-Agent Hamiltonian Drift Detection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('multi_agent_drift_detection.png', dpi=150, bbox_inches='tight')
    plt.show()

    return mas, drift_signals


# Run all experiments
if __name__ == "__main__":
    print("=" * 70)
    print("HAMILTONIAN LEARNING - COMPREHENSIVE VALIDATION")
    print("=" * 70)

    # 1. Run gradual drift detection
    print("\n1. Testing Gradual Drift Detection...")
    run_gradual_drift_experiment()

    # 2. Run multi-agent detection
    print("\n2. Testing Multi-Agent Drift Detection...")
    mas, signals = run_multi_agent_experiment()

    # 3. Validate Hamiltonian theory
    print("\n3. Validating Hamiltonian Theory...")
    from RuptureDetector import HamiltonianRuptureDetector
    from thu_drift_loader import create_sequential_stream

    loader, _, _ = create_sequential_stream(
        dataset_name="CNNS_Nonlinear_Sudden_ChocolateRotation",
        batch_size=1,
        return_rupture_positions=False
    )

    sample_X, sample_Y, _ = next(iter(loader))
    detector = HamiltonianRuptureDetector(input_dim=sample_X.shape[1])

    # Run a few steps
    for i, (X, Y, _) in enumerate(loader):
        if i >= 500:
            break
        X = X.squeeze()
        Y = Y.squeeze()
        detector.hamiltonian_step(X, Y, learning_rate=0.005, dt=0.01)

    validator = HamiltonianValidator(detector)
    validator.plot_theoretical_validation()

    boundary_check = validator.validate_costate_boundary_condition()
    print(f"\nBoundary Condition Check: {boundary_check}")

    print("\n" + "=" * 70)
    print("EXPERIMENTS COMPLETE")
    print("=" * 70)