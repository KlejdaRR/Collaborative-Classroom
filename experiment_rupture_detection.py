# experiment_rupture_detection.py (UPDATED VERSION)
import torch
import matplotlib.pyplot as plt
import numpy as np
from thu_drift_loader import create_sequential_stream
from RuptureDetector import HamiltonianRuptureDetector
from gradual_drift_detector import HamiltonianGradualDriftDetector
from multi_agent_drift_detector import MultiAgentHamiltonianDetector
from hamiltonian_validation import HamiltonianValidator


def run_rupture_detection_experiment(dataset_name="CNNS_Nonlinear_Sudden_ChocolateRotation",
                                     use_enhanced=False,
                                     mode="standard"):
    """
    Run Hamiltonian rupture detection on THU dataset.

    Args:
        dataset_name: Name of THU dataset
        use_enhanced: Use enhanced gradual drift detector
        mode: "standard", "gradual", "multi_agent", or "all"
    """
    print("=" * 70)
    print("HAMILTONIAN RUPTURE DETECTION EXPERIMENT")
    print(f"Dataset: {dataset_name}")
    print(f"Mode: {mode}")
    print("=" * 70)

    # Load data
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

    # Select detector based on mode
    if mode == "gradual":
        detector = HamiltonianGradualDriftDetector(
            input_dim=input_dim,
            hidden_dim=64,
            costate_decay=0.99
        )
        print("\nUsing GRADUAL drift detector")
    elif mode == "multi_agent":
        # Multi-agent returns a different structure
        return run_multi_agent_integrated(loader, rupture_positions, drift_type, input_dim)
    elif mode == "all":
        # Run all modes and compare
        return run_all_modes_comparison(loader, rupture_positions, drift_type, input_dim)
    else:
        detector = HamiltonianRuptureDetector(
            input_dim=input_dim,
            hidden_dim=64,
            costate_decay=0.98
        )
        print("\nUsing STANDARD rupture detector")

    if mode != "multi_agent" and mode != "all":
        # Standard processing for single detector
        return run_detector_processing(detector, loader, rupture_positions, drift_type, mode)


def run_detector_processing(detector, loader, rupture_positions, drift_type, mode):
    """Process stream with a single detector"""
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
    dL_dv_history = []

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

            # Track costate and state (handle different detector types)
            if hasattr(detector, 'costate'):
                costates.append(detector.costate.item())
            elif hasattr(detector, 'costate_history') and len(detector.costate_history) > 0:
                costates.append(detector.costate_history[-1])
            else:
                costates.append(0.0)

            if hasattr(detector, 'state'):
                states.append(detector.state.item())
            else:
                states.append(0.5)

            steps.append(step)

            # Track dL_dv if available
            if hasattr(detector, 'dL_dv_history'):
                dL_dv_history.extend(detector.dL_dv_history[-1:] if detector.dL_dv_history else [0])

            # Print progress every 20000 steps
            if step % 20000 == 0 and step > 0:
                avg_costate = np.mean(np.abs(costates[-1000:])) if len(costates) >= 1000 else np.mean(np.abs(costates))
                print(f"  Step {step:6d} | Loss: {loss:.4f} | "
                      f"|costate|: {avg_costate:.6f} | State: {states[-1]:.4f}")

    # Plot results
    plot_rupture_detection_results(steps, losses, costates, states,
                                   predictions, truths, rupture_positions,
                                   drift_type, total_batches, num_epochs, mode)

    # Run theoretical validation if detector has required attributes
    if hasattr(detector, 'dL_dv_history') and len(detector.dL_dv_history) > 100:
        print("\n" + "=" * 70)
        print("THEORETICAL VALIDATION")
        print("=" * 70)
        validator = HamiltonianValidator(detector)
        correlation, p_value = validator.validate_hamiltonian_relationship()
        if correlation:
            print(f"  Hamiltonian relationship correlation: {correlation:.3f} (p={p_value:.3e})")
        boundary_check = validator.validate_costate_boundary_condition()
        print(f"  Boundary condition (p(T)→0): {boundary_check}")
        validator.plot_theoretical_validation()

    # Analyze costate-rupture correlation
    analyze_costate_rupture_correlation(costates, rupture_positions, steps,
                                        drift_type, total_batches, mode)

    return detector, steps, costates, states, rupture_positions, drift_type


def run_multi_agent_integrated(loader, rupture_positions, drift_type, input_dim):
    """Run multi-agent detection integrated with the experiment"""
    print("\nUsing MULTI-AGENT detector")

    mas = MultiAgentHamiltonianDetector(input_dim=input_dim, num_agents=5)

    # Tracking
    drift_signals = []
    drift_detections = []
    agent_agreements = []
    agent_costates_list = []
    steps = []

    total_batches = len(loader)
    print(f"Total samples: {total_batches}")
    print("\nProcessing stream...")
    print("-" * 70)

    for batch_idx, (X, Y, idx) in enumerate(loader):
        X = X.squeeze()
        Y = Y.squeeze()

        result = mas.detect_drift_ensemble(X, Y, learning_rate=0.005, dt=0.01)

        drift_signals.append(result['drift_signal'])
        drift_detections.append(1 if result['drift_detected'] else 0)
        agent_agreements.append(result['agent_agreement'])
        agent_costates_list.append(result['agent_costates'])
        steps.append(batch_idx)

        # Print at drift positions or periodically
        if batch_idx % 20000 == 0 and batch_idx > 0:
            print(f"  Step {batch_idx:6d} | Drift Signal: {result['drift_signal']:.4f} | "
                  f"Agreement: {result['agent_agreement']:.2f} | Type: {result['drift_type']}")

    # Plot multi-agent results
    plot_multi_agent_results(steps, drift_signals, drift_detections,
                             agent_agreements, agent_costates_list,
                             rupture_positions, drift_type, total_batches)

    # Analyze detection accuracy
    analyze_multi_agent_performance(drift_detections, drift_signals,
                                    rupture_positions, steps, total_batches)

    return mas, drift_signals, drift_detections


def plot_multi_agent_results(steps, drift_signals, drift_detections,
                             agent_agreements, agent_costates_list,
                             rupture_positions, drift_type, total_batches):
    """Plot multi-agent detection results"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Plot 1: Drift signal
    ax = axes[0]
    ax.plot(steps, drift_signals, 'orange', linewidth=0.5)
    if rupture_positions:
        for pos in rupture_positions:
            ax.axvline(x=pos, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.axvline(x=pos + total_batches, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.set_ylabel('Drift Signal')
    ax.set_title(f'Multi-Agent Drift Signal - {drift_type} drift')
    ax.grid(True, alpha=0.3)

    # Plot 2: Binary detections
    ax = axes[1]
    ax.plot(steps, drift_detections, 'g-', linewidth=0.5)
    if rupture_positions:
        for pos in rupture_positions:
            ax.axvline(x=pos, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=pos + total_batches, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Detection')
    ax.set_title('Binary Drift Detections')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    # Plot 3: Agent agreement (consensus)
    ax = axes[2]
    ax.plot(steps, agent_agreements, 'purple', linewidth=0.5)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Majority threshold')
    if rupture_positions:
        for pos in rupture_positions:
            ax.axvline(x=pos, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=pos + total_batches, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Agent Agreement')
    ax.set_title('Consensus Among Agents (High = Sudden Drift)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Costate diversity across agents
    ax = axes[3]
    # Show costate variance as proxy for diversity
    costate_vars = [np.std(costates) for costates in agent_costates_list]
    ax.plot(steps, costate_vars, 'brown', linewidth=0.5)
    if rupture_positions:
        for pos in rupture_positions:
            ax.axvline(x=pos, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=pos + total_batches, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Costate Variance')
    ax.set_title('Agent Diversity (High Variance = Disagreement = Drift)')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Multi-Agent Hamiltonian Drift Detection ({drift_type} drift)\nRed lines: True drift positions',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'multi_agent_results_{drift_type}.png', dpi=150, bbox_inches='tight')
    plt.show()


def analyze_multi_agent_performance(drift_detections, drift_signals,
                                    rupture_positions, steps, total_batches):
    """Analyze multi-agent detection performance"""
    print("\n" + "=" * 70)
    print("MULTI-AGENT DETECTION ANALYSIS")
    print("=" * 70)

    if not rupture_positions:
        print("No rupture positions for comparison")
        return

    # Analyze detection around true drifts
    detection_ratios = []
    for pos in rupture_positions:
        # Look at window around drift
        window_start = max(0, pos - 1000)
        window_end = min(len(drift_detections), pos + 1000)

        detections_in_window = drift_detections[window_start:window_end]
        detection_rate = sum(detections_in_window) / len(detections_in_window) if detections_in_window else 0

        detection_ratios.append(detection_rate)

        # Also check second epoch
        window_start_epoch2 = total_batches + pos - 1000
        window_end_epoch2 = min(len(drift_detections), total_batches + pos + 1000)
        if window_end_epoch2 > window_start_epoch2:
            detections_in_window2 = drift_detections[window_start_epoch2:window_end_epoch2]
            detection_rate2 = sum(detections_in_window2) / len(detections_in_window2) if detections_in_window2 else 0
            detection_ratios.append(detection_rate2)

    avg_detection_rate = np.mean(detection_ratios) if detection_ratios else 0

    print(f"\nDetection performance at true drift positions:")
    print(f"  Average detection rate within ±1000 samples: {avg_detection_rate:.1%}")

    if avg_detection_rate > 0.5:
        print(f"\n  ✅ Multi-agent system successfully detects drifts!")
        print(f"  The ensemble approach provides robust detection through")
        print(f"  agent diversity and consensus voting.")
    else:
        print(f"\n  ⚠️  Detection rate could be improved. Consider:")
        print(f"     - Increasing number of agents")
        print(f"     - Adjusting detection threshold")
        print(f"     - Tuning individual agent decay rates")


def run_all_modes_comparison(loader, rupture_positions, drift_type, input_dim):
    """Run all detection modes and compare results"""
    print("\n" + "=" * 70)
    print("RUNNING ALL MODES FOR COMPARISON")
    print("=" * 70)

    results = {}

    # Mode 1: Standard
    print("\n[1/3] Running Standard Detector...")
    detector_std = HamiltonianRuptureDetector(input_dim=input_dim, hidden_dim=64, costate_decay=0.98)
    _, steps_std, costates_std, _, _, _ = run_detector_processing(
        detector_std, loader, rupture_positions, drift_type, "standard_comparison"
    )
    results['standard'] = costates_std

    # Reset loader (reload to ensure fresh stream)
    loader, rupture_positions, drift_type = create_sequential_stream(
        dataset_name="CNNS_Nonlinear_Sudden_ChocolateRotation",
        batch_size=1,
        return_rupture_positions=True
    )

    # Mode 2: Gradual
    print("\n[2/3] Running Gradual Detector...")
    detector_grad = HamiltonianGradualDriftDetector(input_dim=input_dim, hidden_dim=64, costate_decay=0.99)
    _, steps_grad, costates_grad, _, _, _ = run_detector_processing(
        detector_grad, loader, rupture_positions, drift_type, "gradual_comparison"
    )
    results['gradual'] = costates_grad

    # Mode 3: Multi-agent
    print("\n[3/3] Running Multi-Agent Detector...")
    mas, signals, detections = run_multi_agent_integrated(loader, rupture_positions, drift_type, input_dim)
    results['multi_agent_signals'] = signals
    results['multi_agent_detections'] = detections

    # Comparison plot
    plot_mode_comparison(results, rupture_positions, drift_type)

    return results


def plot_mode_comparison(results, rupture_positions, drift_type):
    """Compare different detection modes"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Standard detector
    ax = axes[0]
    if 'standard' in results:
        ax.plot(results['standard'][:50000], 'purple', linewidth=0.5)
        if rupture_positions:
            for pos in rupture_positions:
                if pos < 50000:
                    ax.axvline(x=pos, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('|Costate|')
    ax.set_title('Standard Detector')
    ax.grid(True, alpha=0.3)

    # Gradual detector
    ax = axes[1]
    if 'gradual' in results:
        ax.plot(results['gradual'][:50000], 'orange', linewidth=0.5)
        if rupture_positions:
            for pos in rupture_positions:
                if pos < 50000:
                    ax.axvline(x=pos, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('|Costate|')
    ax.set_title('Gradual Detector')
    ax.grid(True, alpha=0.3)

    # Multi-agent
    ax = axes[2]
    if 'multi_agent_signals' in results:
        ax.plot(results['multi_agent_signals'][:50000], 'green', linewidth=0.5)
        if rupture_positions:
            for pos in rupture_positions:
                if pos < 50000:
                    ax.axvline(x=pos, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Drift Signal')
    ax.set_xlabel('Step')
    ax.set_title('Multi-Agent Detector')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Detection Mode Comparison ({drift_type} drift)\nRed lines: True drift positions',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('detection_modes_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_rupture_detection_results(steps, losses, costates, states,
                                   predictions, truths, rupture_positions,
                                   drift_type, total_batches, num_epochs, mode="standard"):
    """Plot costate spikes at rupture points (updated for mode)"""

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Plot 1: Loss over time
    ax = axes[0]
    ax.plot(steps, losses, 'b-', alpha=0.7, linewidth=0.5)
    ax.set_ylabel('Loss')
    ax.set_title(f'Training Loss Over Time - Mode: {mode}')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 2: Costate (rupture signal)
    ax = axes[1]
    ax.plot(steps, costates, 'purple', linewidth=0.8)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_ylabel('Costate p(t)')
    ax.set_title(f'Costate Evolution - {mode} detector, Drift: {drift_type}')
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

    # Plot 3: State
    ax = axes[2]
    ax.plot(steps, states, 'green', linewidth=0.8)
    ax.set_ylabel('State v(t)')
    ax.set_title('State Evolution')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Plot 4: Costate magnitude
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

    plt.suptitle(f'Hamiltonian Rupture Detection ({mode} mode, {drift_type} drift)\nRed lines: True drift positions',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'rupture_detection_results_{drift_type}_{mode}.png', dpi=150, bbox_inches='tight')
    plt.show()


def analyze_costate_rupture_correlation(costates, rupture_positions, steps, drift_type, total_batches, mode="standard"):
    """Quantitative analysis of costate spikes at true rupture points"""

    print("\n" + "=" * 70)
    print(f"COSTATE-RUPTURE CORRELATION ANALYSIS - Mode: {mode}")
    print(f"Drift type: {drift_type}")
    print("=" * 70)

    if rupture_positions is None:
        print("No rupture positions provided for analysis")
        return

    abs_costates = np.abs(costates)
    spike_ratios = []

    print("\nAnalyzing costate behavior at true drift positions (25k, 50k, 75k):")
    print("-" * 70)

    for i, pos in enumerate(rupture_positions):
        window = 2000
        start, end = max(0, pos - window), min(len(costates), pos + window)

        baseline_start = max(0, pos - 10000)
        baseline_end = max(0, pos - 5000)
        if baseline_end > baseline_start:
            baseline_vals = abs_costates[baseline_start:baseline_end]
            baseline = np.mean(baseline_vals) if len(baseline_vals) > 0 else np.mean(abs_costates)
        else:
            baseline = np.mean(abs_costates)

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
    print(f"\n📊 FINAL CONCLUSION ({mode} mode):")
    print("-" * 70)
    if avg_ratio > 2.0:
        print(f"  ✅✅✅ EXCELLENT: Costate detects sudden drift! ({avg_ratio:.1f}x baseline)")
    elif avg_ratio > 1.5:
        print(f"  ✅ GOOD: Costate detects sudden drift ({avg_ratio:.1f}x baseline)")
    elif avg_ratio > 1.2:
        print(f"  ⚠️  WEAK: Costate shows moderate response ({avg_ratio:.1f}x baseline)")
    else:
        print(f"  ❌ NO DETECTION: Costate does not spike at drift positions ({avg_ratio:.1f}x baseline)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Hamiltonian Rupture Detection')
    parser.add_argument('--mode', type=str, default='standard',
                        choices=['standard', 'gradual', 'multi_agent', 'all'],
                        help='Detection mode to run')
    parser.add_argument('--dataset', type=str,
                        default='CNNS_Nonlinear_Sudden_ChocolateRotation',
                        help='THU dataset name')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("RUNNING THU DATASET EXPERIMENT")
    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print("=" * 70)

    if args.mode == "standard":
        detector, steps, costates, states, ruptures, drift_type = run_rupture_detection_experiment(
            args.dataset, use_enhanced=False, mode="standard"
        )
    elif args.mode == "gradual":
        detector, steps, costates, states, ruptures, drift_type = run_rupture_detection_experiment(
            args.dataset, use_enhanced=True, mode="gradual"
        )
    elif args.mode == "multi_agent":
        loader, ruptures, drift_type = create_sequential_stream(
            dataset_name=args.dataset,
            batch_size=1,
            return_rupture_positions=True
        )
        sample_X, sample_Y, _ = next(iter(loader))
        run_multi_agent_integrated(loader, ruptures, drift_type, sample_X.shape[1])
    else:  # all
        loader, ruptures, drift_type = create_sequential_stream(
            dataset_name=args.dataset,
            batch_size=1,
            return_rupture_positions=True
        )
        sample_X, sample_Y, _ = next(iter(loader))
        run_all_modes_comparison(loader, ruptures, drift_type, sample_X.shape[1])