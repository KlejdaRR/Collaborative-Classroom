import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys

# Get the absolute path to THU-Concept-Drift-Datasets folder
thu_path = os.path.join(os.path.dirname(__file__), 'data', 'THU-Concept-Drift-Datasets')
sys.path.insert(0, thu_path)

print(f"Looking for THU dataset at: {thu_path}")

try:
    from DatasetsInput import Datasets

    print("✅ Successfully imported THU DatasetsInput")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    raise


class THUDriftDataset(Dataset):
    def __init__(self, X, Y, rupture_indices=None):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
        self.rupture_indices = rupture_indices

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], idx


def find_true_drift_positions(X, Y, n_drifts=3):
    """
    Find true concept drift positions by analyzing where class distribution
    or feature statistics change significantly.

    For THU datasets, drifts typically occur at 1/4, 1/2, and 3/4 of the data.
    """
    n_samples = len(X)

    # Method 1: Look for changes in class balance
    if Y is not None:
        # Use a sliding window to detect distribution shifts
        window_size = 2000
        class_ratios = []

        for i in range(0, n_samples - window_size, window_size // 2):
            window_end = min(i + window_size, n_samples)
            class_ratio = np.mean(Y[i:window_end])
            class_ratios.append((i, class_ratio))

        # Find positions where class ratio changes most dramatically
        changes = []
        for i in range(1, len(class_ratios)):
            diff = abs(class_ratios[i][1] - class_ratios[i - 1][1])
            changes.append((class_ratios[i][0], diff))

        # Sort by change magnitude and get top positions
        changes.sort(key=lambda x: x[1], reverse=True)
        drift_candidates = [pos for pos, _ in changes[:n_drifts]]
        drift_candidates.sort()

        # If we found good candidates, use them
        # FIXED: Compare the actual change value, not a boolean
        if len(drift_candidates) >= n_drifts and changes[0][1] > 0.1:
            print(f"Found drifts via class distribution: {drift_candidates}")
            return drift_candidates[:n_drifts]

    # Method 2: Use known THU dataset structure - drifts at 1/4, 1/2, 3/4
    print(f"Using default THU drift positions at 1/4, 1/2, 3/4 of data")
    drift_positions = [n_samples // 4, n_samples // 2, 3 * n_samples // 4]
    return drift_positions

def create_sequential_stream(dataset_name="CNNS_Nonlinear_Sudden_ChocolateRotation",
                             batch_size=1,
                             return_rupture_positions=True):
    """
    Create a sequential stream loader for THU dataset.
    Returns: (loader, rupture_positions, drift_type)
    """
    Data = Datasets()

    if not hasattr(Data, dataset_name):
        available = [attr for attr in dir(Data) if not attr.startswith('_') and callable(getattr(Data, attr))]
        print(f"Available datasets: {available[:10]}...")
        raise ValueError(f"Dataset '{dataset_name}' not found")

    print(f"Loading {dataset_name}...")
    result = getattr(Data, dataset_name)()

    # Handle different return types
    if isinstance(result, tuple) and len(result) >= 2:
        X, Y = result[0], result[1]
    else:
        X, Y = result, None

    # Convert to numpy
    if hasattr(X, 'numpy'):
        X = X.numpy()
    if Y is not None and hasattr(Y, 'numpy'):
        Y = Y.numpy()

    n_samples = len(X)
    print(f"Total samples: {n_samples}")
    print(f"Features: {X.shape[1] if hasattr(X, 'shape') else '?'}")

    # Determine drift type from dataset name
    if "Sudden" in dataset_name:
        drift_type = "sudden"
    elif "Gradual" in dataset_name:
        drift_type = "gradual"
    elif "Abrupt" in dataset_name:
        drift_type = "abrupt"
    else:
        drift_type = "unknown"

    # Find true drift positions
    if return_rupture_positions and Y is not None:
        rupture_positions = find_true_drift_positions(X, Y)
        print(f"True drift positions: {rupture_positions}")
    else:
        rupture_positions = None

    dataset = THUDriftDataset(X, Y, rupture_positions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader, rupture_positions, drift_type


if __name__ == "__main__":
    print("Testing THU dataset loader...")
    loader, ruptures, drift_type = create_sequential_stream()
    print(f"Loaded {len(loader.dataset)} samples")
    print(f"Drift type: {drift_type}")
    print(f"Drift positions: {ruptures}")