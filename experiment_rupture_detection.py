import torch
import matplotlib.pyplot as plt
import numpy as np
from thu_drift_loader import create_sequential_stream
from RuptureDetector import HamiltonianRuptureDetector


def run_rupture_detection_experiment(dataset_name="CNNS_Nonlinear_Sudden_ChocolateRotation"):
    """
    Run Hamiltonian rupture detection on THU dataset.
    """

    print("=" * 70)
    print("HAMILTONIAN RUPTURE DETECTION EXPERIMENT")
    print(f"Dataset: {dataset_name}")
    print("=" * 70)

    # Load data ONCE
    loader, rupture_positions, drift_type = create_sequential_stream(
        dataset_name=dataset_name,
        batch_size=1,
        return_rupture_positions=True
    )

    sample_X, sample_Y, _ = next(iter(loader))
    input_dim = sample_X.shape[1]

    print(f"\nInput dimension: {input_dim}")
    print(f"Drift type: {drift_type}")
    print(f"True drift positions (to detect): {rupture_positions}")

    detector = HamiltonianRuptureDetector(
        input_dim=input_dim,
        hidden_dim=64,
        costate_decay=0.98
    )

    num_epochs = 2
    learning_rate = 0.005
    dt = 0.01

    # Tracking
    losses = []
    predictions = []
    truths = []
    costates = []
    states = []
    steps = []

    print("\nProcessing stream...")
    print("-" * 70)

    total_batches = len(loader)
    print(f"Total samples: {total_batches}")

    # Convert loader to list for multiple epochs
    all_data = []
    for X, Y, idx in loader:
        all_data.append((X.squeeze(), Y.squeeze(), idx.item()))

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        for batch_idx, (X, Y, idx) in enumerate(all_data):
            step = epoch * total_batches + batch_idx

            loss, pred = detector.hamiltonian_step(
                X, Y,
                learning_rate=learning_rate,
                dt=dt
            )

            losses.append(loss)
            predictions.append(pred.item() if hasattr(pred, 'item') else pred)
            truths.append(Y.item() if hasattr(Y, 'item') else Y)
            costates.append(detector.costate.item())
            states.append(detector.state.item())
            steps.append(step)

            # Print progress every 20000 steps
            if step % 20000 == 0 and step > 0:
                avg_costate = np.mean(np.abs(costates[-1000:])) if len(costates) >= 1000 else np.mean(np.abs(costates))
                print(f"  Step {step:6d} | Loss: {loss:.4f} | "
                      f"|costate|: {avg_costate:.6f} | State: {detector.state.item():.4f}")

    # Plot results with true drift positions
    plot_rupture_detection_results(steps, losses, costates, states,
                                   predictions, truths, rupture_positions, drift_type, total_batches, num_epochs)

    return detector, steps, costates, states, rupture_positions, drift_type


def plot_rupture_detection_results(steps, losses, costates, states,
                                   predictions, truths, rupture_positions, drift_type, total_batches, num_epochs):
    """Plot costate spikes at rupture points"""

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Plot 1: Loss over time
    ax = axes[0]
    ax.plot(steps, losses, 'b-', alpha=0.7, linewidth=0.5)
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Over Time')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale to see loss variations

    # Plot 2: Costate (rupture signal)
    ax = axes[1]
    ax.plot(steps, costates, 'purple', linewidth=0.8)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_ylabel('Costate p(t)')
    ax.set_title(f'Costate Evolution - Drift Type: {drift_type}')
    ax.grid(True, alpha=0.3)

    # Mark true rupture positions
    if rupture_positions is not None:
        for epoch in range(num_epochs):
            for pos in rupture_positions:
                pos_scaled = epoch * total_batches + pos
                if pos_scaled < len(steps):
                    ax.axvline(x=pos_scaled, color='red', linestyle='--', linewidth=1.5, alpha=0.8,
                               label='Drift' if epoch == 0 else '')
                    axes[2].axvline(x=pos_scaled, color='red', linestyle='--', linewidth=1.5, alpha=0.8)

    # Plot 3: State (rupture probability)
    ax = axes[2]
    ax.plot(steps, states, 'green', linewidth=0.8)
    ax.set_ylabel('State v(t)')
    ax.set_title('State Evolution (Rupture Probability)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Plot 4: Costate magnitude over time
    ax = axes[3]
    abs_costates = np.abs(costates)
    ax.plot(steps, abs_costates, 'orange', linewidth=0.8)
    if rupture_positions is not None:
        for epoch in range(num_epochs):
            for pos in rupture_positions:
                pos_scaled = epoch * total_batches + pos
                if pos_scaled < len(steps):
                    ax.axvline(x=pos_scaled, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.set_ylabel('|Costate|')
    ax.set_xlabel('Step')
    ax.set_title('Costate Magnitude (Rupture Signal)')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Hamiltonian Rupture Detection on THU Dataset ({drift_type} drift)\nRed lines: True drift positions',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'rupture_detection_results_{drift_type}.png', dpi=150, bbox_inches='tight')
    plt.show()


def analyze_costate_rupture_correlation(costates, rupture_positions, steps, drift_type, total_batches):
    """Quantitative analysis of costate spikes at true rupture points"""

    print("\n" + "=" * 70)
    print("COSTATE-RUPTURE CORRELATION ANALYSIS")
    print(f"Drift type: {drift_type}")
    print("=" * 70)

    if rupture_positions is None:
        print("No rupture positions provided for analysis")
        return

    abs_costates = np.abs(costates)

    # For THU dataset, drifts are at 25k, 50k, 75k within each epoch
    # Analyze windows around these positions
    spike_ratios = []

    print("\nAnalyzing costate behavior at true drift positions (25k, 50k, 75k):")
    print("-" * 70)

    for i, pos in enumerate(rupture_positions):
        window = 2000  # Look at 2000 samples before/after drift
        start, end = max(0, pos - window), min(len(costates), pos + window)

        # Get baseline from far away from drifts
        baseline_start = max(0, pos - 10000)
        baseline_end = max(0, pos - 5000)
        if baseline_end > baseline_start:
            baseline_vals = abs_costates[baseline_start:baseline_end]
            baseline = np.mean(baseline_vals) if len(baseline_vals) > 0 else np.mean(abs_costates)
        else:
            baseline = np.mean(abs_costates)

        # Get values around drift
        drift_vals = abs_costates[start:end]
        drift_avg = np.mean(drift_vals)
        drift_peak = np.max(drift_vals)

        ratio = drift_avg / baseline if baseline > 0 else 1
        spike_ratios.append(ratio)

        print(f"  Drift {i + 1} at {pos}:")
        print(f"    Baseline (before drift): {baseline:.6f}")
        print(f"    Costate during drift: avg={drift_avg:.6f}, peak={drift_peak:.6f}")
        print(f"    Spike ratio: {ratio:.2f}x\n")

    avg_ratio = np.mean(spike_ratios) if spike_ratios else 0

    print("=" * 70)
    print(f"\n📊 FINAL CONCLUSION:")
    print("-" * 70)
    if avg_ratio > 2.0:
        print(f"  ✅✅✅ EXCELLENT: Costate detects sudden drift! ({avg_ratio:.1f}x baseline)")
        print(f"  The Hamiltonian costate successfully identifies concept drift")
        print(f"  with high sensitivity.")
    elif avg_ratio > 1.5:
        print(f"  ✅ GOOD: Costate detects sudden drift ({avg_ratio:.1f}x baseline)")
        print(f"  The Hamiltonian costate shows clear spikes at drift positions.")
    elif avg_ratio > 1.2:
        print(f"  ⚠️  WEAK: Costate shows moderate response ({avg_ratio:.1f}x baseline)")
        print(f"  Consider adjusting hyperparameters or more training.")
    else:
        print(f"  ❌ NO DETECTION: Costate does not spike at drift positions ({avg_ratio:.1f}x baseline)")
        print(f"  Possible issues: costate decay too strong, learning rate too low,")
        print(f"  or network capacity insufficient.")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RUNNING THU DATASET EXPERIMENT")
    print("=" * 70)

    detector, steps, costates, states, ruptures, drift_type = run_rupture_detection_experiment(
        "CNNS_Nonlinear_Sudden_ChocolateRotation"
    )
    analyze_costate_rupture_correlation(costates, ruptures, steps, drift_type, 100000)