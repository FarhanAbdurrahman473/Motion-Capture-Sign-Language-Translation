"""
Data Preprocessing Module

Handles loading, validation, normalization, and preparation of hand landmark
datasets for machine learning training in the Sign Language Translation project.

Features:
- Load raw landmark data from collected gestures
- Validate landmark dimensions (63 features per hand)
- Coordinate normalization (translation + scale invariance)
- Label encoding for gesture classes
- Dataset builder for training pipeline
- Export processed dataset (.npy files)
- Dataset statistics generation

Usage:
    python -m src.preprocess

Integration:
    - src/capture.py   : Provides raw landmark data
    - src/train.py     : Consumes processed dataset
    - src/predict.py    : Consumes processed dataset
"""

import numpy as np
import pandas as pd
import os
import json
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from tqdm import tqdm

# =============================================================================
# CONSTANTS
# =============================================================================

# Gesture classes for MVP
LABELS = {
    "Halo": 0,
    "Makan": 1,
    "Minum": 2,
    "Tolong": 3,
    "TerimaKasih": 4
}

LABEL_TO_GESTURE = {v: k for k, v in LABELS.items()}

# Expected landmark configuration
NUM_LANDMARKS = 21
NUM_COORDINATES = 3  # x, y, z
EXPECTED_FEATURES = NUM_LANDMARKS * NUM_COORDINATES  # 63

# Default paths
DEFAULT_DATASET_DIR = "dataset"
DEFAULT_OUTPUT_DIR = "dataset/processed"


# =============================================================================
# LANDMARK VALIDATION
# =============================================================================

def validate_landmarks(landmarks: np.ndarray) -> Tuple[bool, str]:
    """
    Validate landmark data for ML training readiness.

    Checks:
    - Exactly 63 features (21 landmarks × 3 coordinates)
    - No missing values
    - No NaN or Inf values
    - Valid numerical range

    Args:
        landmarks: Array of landmark coordinates

    Returns:
        Tuple of (is_valid, message)
        - is_valid: True if data passes all checks
        - message: Status message or error description
    """
    # Check if empty
    if landmarks is None:
        return False, "Landmarks is None"

    # Convert to numpy array if needed
    if not isinstance(landmarks, np.ndarray):
        try:
            landmarks = np.array(landmarks)
        except Exception as e:
            return False, f"Cannot convert to array: {e}"

    # Check shape
    if len(landmarks) == 0:
        return False, "Empty landmark array"

    # Flatten for checking
    flat_landmarks = landmarks.flatten()

    # Check feature count
    if len(flat_landmarks) != EXPECTED_FEATURES:
        return False, f"Invalid feature count: {len(flat_landmarks)} (expected {EXPECTED_FEATURES})"

    # Check for NaN values
    if np.any(np.isnan(flat_landmarks)):
        return False, "Contains NaN values"

    # Check for Inf values
    if np.any(np.isinf(flat_landmarks)):
        return False, "Contains Inf values"

    # Check for reasonable coordinate range (should be normalized or ~0-1)
    # Allow some tolerance for unnormalized data
    abs_max = np.max(np.abs(flat_landmarks))
    if abs_max > 10000:
        return False, f"Coordinates out of reasonable range: {abs_max}"

    return True, "Valid"


def load_landmark_file(filepath: str) -> Optional[np.ndarray]:
    """
    Load landmark data from a single file.

    Supports formats:
    - .npy (NumPy binary)
    - .csv (comma-separated values)
    - .json (landmarks stored as list)

    Args:
        filepath: Path to landmark file

    Returns:
        Landmark array or None if loading fails
    """
    filepath = Path(filepath)

    if not filepath.exists():
        print(f"File not found: {filepath}")
        return None

    try:
        ext = filepath.suffix.lower()

        if ext == '.npy':
            return np.load(filepath)

        elif ext == '.csv':
            # CSV: each row is one sample, 63 columns
            df = pd.read_csv(filepath, header=None)
            return df.values

        elif ext == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            return np.array(data)

        else:
            print(f"Unsupported file format: {ext}")
            return None

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def load_raw_landmarks(dataset_dir: str = DEFAULT_DATASET_DIR) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load all raw landmark data from dataset directory.

    Expected structure:
        dataset/
        ├── Halo/
        │   ├── sample_001.npy
        │   └── ...
        ├── Makan/
        └── ...

    Args:
        dataset_dir: Root dataset directory

    Returns:
        Tuple of (landmarks_list, labels_list)
        - landmarks_list: List of landmark arrays
        - labels_list: List of corresponding labels
    """
    dataset_path = Path(dataset_dir)
    landmarks_list = []
    labels_list = []

    for gesture_name, label in LABELS.items():
        gesture_dir = dataset_path / gesture_name

        if not gesture_dir.exists():
            print(f"Warning: Directory not found: {gesture_dir}")
            continue

        # Find all landmark files
        npy_files = list(gesture_dir.glob("*.npy"))

        for file_path in tqdm(npy_files, desc=f"Loading {gesture_name}"):
            landmarks = load_landmark_file(file_path)

            if landmarks is not None:
                # Handle multiple samples in one file
                if len(landmarks.shape) == 2 and landmarks.shape[1] == EXPECTED_FEATURES:
                    for sample in landmarks:
                        landmarks_list.append(sample)
                        labels_list.append(label)
                elif len(landmarks.flatten()) == EXPECTED_FEATURES:
                    landmarks_list.append(landmarks.flatten())
                    labels_list.append(label)

    print(f"\nLoaded {len(landmarks_list)} samples from dataset")

    return landmarks_list, labels_list


# =============================================================================
# COORDINATE NORMALIZATION
# =============================================================================

def normalize_landmarks(landmarks: np.ndarray,
                        method: str = "wrist_and_scale") -> np.ndarray:
    """
    Normalize landmark coordinates for position and scale invariance.

    Two-step normalization:
    1. Translation Normalization: Subtract wrist landmark (index 0) as origin
    2. Scale Normalization: Divide by maximum absolute value

    Args:
        landmarks: Raw landmark array (63 features)
        method: Normalization method (currently only "wrist_and_scale")

    Returns:
        Normalized landmark array

    Raises:
        ValueError: If landmarks don't have exactly 63 features
    """
    # Convert to numpy array
    landmarks = np.array(landmarks, dtype=np.float64)

    # Validate dimensions
    if len(landmarks) != EXPECTED_FEATURES:
        raise ValueError(f"Expected {EXPECTED_FEATURES} features, got {len(landmarks)}")

    # Reshape to 21 landmarks × 3 coordinates
    landmarks = landmarks.reshape(NUM_LANDMARKS, NUM_COORDINATES)

    # === Translation Normalization ===
    # Use wrist landmark (landmark 0) as reference point
    wrist = landmarks[0]
    translated = landmarks - wrist

    # === Scale Normalization ===
    # Find maximum absolute value across all coordinates
    max_abs = np.max(np.abs(translated))

    # Avoid division by zero
    if max_abs > 1e-8:
        scaled = translated / max_abs
    else:
        scaled = translated

    # Flatten back to 63 features
    return scaled.flatten()


def normalize_dataset(landmarks_list: List[np.ndarray]) -> np.ndarray:
    """
    Normalize a list of landmark arrays.

    Args:
        landmarks_list: List of raw landmark arrays

    Returns:
        Normalized dataset array of shape (n_samples, 63)
    """
    normalized = []

    for landmarks in landmarks_list:
        normalized.append(normalize_landmarks(landmarks))

    return np.array(normalized)


# =============================================================================
# LABEL ENCODING
# =============================================================================

def encode_label(label: str) -> int:
    """
    Encode gesture label to numerical value.

    Args:
        label: Gesture name (e.g., "Halo", "Makan")

    Returns:
        Integer label (0-4)

    Raises:
        ValueError: If label is not recognized
    """
    label = label.strip()

    if label not in LABELS:
        raise ValueError(f"Unknown label: '{label}'. Available: {list(LABELS.keys())}")

    return LABELS[label]


def decode_label(label_id: int) -> str:
    """
    Decode numerical label back to gesture name.

    Args:
        label_id: Integer label (0-4)

    Returns:
        Gesture name

    Raises:
        ValueError: If label_id is out of range
    """
    if label_id not in LABEL_TO_GESTURE:
        raise ValueError(f"Unknown label ID: {label_id}")

    return LABEL_TO_GESTURE[label_id]


def encode_labels(labels: List[str]) -> np.ndarray:
    """
    Encode a list of gesture labels.

    Args:
        labels: List of gesture names

    Returns:
        Array of integer labels
    """
    return np.array([encode_label(label) for label in labels])


# =============================================================================
# DATASET BUILDER
# =============================================================================

def build_dataset(dataset_dir: str = DEFAULT_DATASET_DIR,
                  normalize: bool = True,
                  validate: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build complete dataset from raw landmark files.

    Pipeline:
    1. Load all landmark files from dataset directory
    2. Validate each sample
    3. Normalize coordinates (optional)
    4. Assign labels
    5. Return feature matrix and labels

    Args:
        dataset_dir: Root dataset directory
        normalize: Whether to normalize landmarks
        validate: Whether to validate landmarks

    Returns:
        Tuple of (X, y)
        - X: Feature matrix of shape (n_samples, 63)
        - y: Label array of shape (n_samples,)
    """
    # Load raw landmarks
    landmarks_list, labels_list = load_raw_landmarks(dataset_dir)

    if len(landmarks_list) == 0:
        print("Warning: No landmarks loaded")
        return np.array([]), np.array([])

    # Validate and filter
    valid_indices = []
    for idx, landmarks in enumerate(landmarks_list):
        is_valid, message = validate_landmarks(landmarks)

        if not is_valid:
            print(f"  Rejecting sample {idx}: {message}")
            continue

        valid_indices.append(idx)

    # Filter invalid samples
    landmarks_list = [landmarks_list[i] for i in valid_indices]
    labels_list = [labels_list[i] for i in valid_indices]

    print(f"Valid samples: {len(landmarks_list)}/{len(valid_indices)}")

    # Normalize if requested
    if normalize:
        print("Normalizing landmarks...")
        landmarks_list = [normalize_landmarks(lm) for lm in landmarks_list]

    # Convert to arrays
    X = np.array(landmarks_list)
    y = np.array(labels_list)

    return X, y


# =============================================================================
# DATASET EXPORT
# =============================================================================

def save_dataset(X: np.ndarray,
                 y: np.ndarray,
                 output_dir: str = DEFAULT_OUTPUT_DIR) -> Dict[str, str]:
    """
    Save processed dataset to files.

    Exports:
    - X.npy: Feature matrix
    - y.npy: Label array
    - dataset_info.json: Metadata
    - dataset.csv: Human-readable format (optional)

    Args:
        X: Feature matrix of shape (n_samples, 63)
        y: Label array of shape (n_samples,)
        output_dir: Output directory

    Returns:
        Dictionary mapping filename to filepath
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save X.npy
    X_path = output_path / "X.npy"
    np.save(X_path, X)
    print(f"Saved features: {X_path}")

    # Save y.npy
    y_path = output_path / "y.npy"
    np.save(y_path, y)
    print(f"Saved labels: {y_path}")

    # Save dataset_info.json
    info = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "classes": LABELS,
        "class_distribution": get_class_distribution(y)
    }

    info_path = output_path / "dataset_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Saved metadata: {info_path}")

    # Save CSV (human-readable)
    csv_path = output_path / "dataset.csv"
    df = pd.DataFrame(X)
    df['label'] = y
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    return {
        "X.npy": str(X_path),
        "y.npy": str(y_path),
        "dataset_info.json": str(info_path),
        "dataset.csv": str(csv_path)
    }


def load_processed_dataset(output_dir: str = DEFAULT_OUTPUT_DIR) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load processed dataset from files.

    Args:
        output_dir: Directory containing processed dataset

    Returns:
        Tuple of (X, y)

    Raises:
        FileNotFoundError: If dataset files not found
    """
    output_path = Path(output_dir)

    X_path = output_path / "X.npy"
    y_path = output_path / "y.npy"

    if not X_path.exists():
        raise FileNotFoundError(f"Features file not found: {X_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Labels file not found: {y_path}")

    X = np.load(X_path)
    y = np.load(y_path)

    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")

    return X, y


# =============================================================================
# DATASET STATISTICS
# =============================================================================

def get_class_distribution(y: np.ndarray) -> Dict[str, int]:
    """
    Calculate class distribution in dataset.

    Args:
        y: Label array

    Returns:
        Dictionary mapping gesture name to count
    """
    distribution = {}

    for label_id, count in zip(*np.unique(y, return_counts=True)):
        gesture_name = decode_label(label_id)
        distribution[gesture_name] = int(count)

    return distribution


def print_dataset_summary(X: np.ndarray,
                          y: np.ndarray,
                          title: str = "DATASET SUMMARY") -> Dict:
    """
    Print and return dataset statistics.

    Args:
        X: Feature matrix
        y: Label array
        title: Title for summary output

    Returns:
        Dictionary with dataset statistics
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_classes = len(np.unique(y))

    class_dist = get_class_distribution(y)
    min_samples = min(class_dist.values()) if class_dist else 0
    max_samples = max(class_dist.values()) if class_dist else 0
    is_balanced = (max_samples - min_samples) / max_samples <= 0.2 if max_samples > 0 else False

    # Calculate feature statistics
    feature_mean = np.mean(X, axis=0)
    feature_std = np.std(X, axis=0)
    feature_min = np.min(X, axis=0)
    feature_max = np.max(X, axis=0)

    summary = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "class_distribution": class_dist,
        "is_balanced": is_balanced,
        "feature_stats": {
            "mean": feature_mean,
            "std": feature_std,
            "min": feature_min,
            "max": feature_max
        }
    }

    # Print summary
    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50)
    print(f"Classes:     {n_classes}")
    print(f"Samples:     {n_samples}")
    print(f"Features:    {n_features}")
    print()
    print("Class Distribution:")
    for gesture, count in class_dist.items():
        bar = "█" * int(count / max_samples * 20) if max_samples > 0 else ""
        print(f"  {gesture:15s}: {count:4d} |{bar}")
    print()
    print(f"Balance:     {'✓ Balanced' if is_balanced else '⚠ Unbalanced'}")
    print(f"Feature Range: [{feature_min.min():.4f}, {feature_max.max():.4f}]")
    print("=" * 50)

    # Status message
    status = "✓ Valid and Ready for Training" if (
        n_samples > 0 and is_balanced
    ) else "⚠ Needs attention"

    print(f"\nStatus: {status}")

    return summary


# =============================================================================
# DATA AUGMENTATION (Optional Enhancement)
# =============================================================================

def augment_landmarks(landmarks: np.ndarray,
                      noise_std: float = 0.01,
                      scale_range: tuple = (0.9, 1.1)) -> List[np.ndarray]:
    """
    Generate augmented versions of landmarks.

    Augmentations:
    - Add Gaussian noise
    - Random scaling

    Args:
        landmarks: Original landmark array (63 features)
        noise_std: Standard deviation for Gaussian noise
        scale_range: Min and max scale factors

    Returns:
        List of augmented landmark arrays (including original)
    """
    import random

    augmented = []

    # Original
    augmented.append(landmarks)

    # With noise
    noise = np.random.normal(0, noise_std, landmarks.shape)
    augmented.append(landmarks + noise)

    # Scaled up
    scale_up = random.uniform(scale_range[0], scale_range[1])
    augmented.append(landmarks * scale_up)

    # Scaled down
    scale_down = random.uniform(scale_range[0], scale_range[1])
    augmented.append(landmarks / scale_down)

    return augmented


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main entry point for data preprocessing.

    Runs the complete preprocessing pipeline:
    1. Build dataset from raw landmarks
    2. Validate and normalize
    3. Save processed dataset
    4. Print summary statistics
    """
    import argparse

    parser = argparse.ArgumentParser(description="Data Preprocessing Pipeline")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_DIR,
                       help="Dataset directory")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR,
                       help="Output directory for processed dataset")
    parser.add_argument("--no-normalize", action="store_true",
                       help="Skip normalization")
    parser.add_argument("--load-only", action="store_true",
                       help="Load existing processed dataset")

    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  DATA PREPROCESSING PIPELINE")
    print("=" * 50)

    # Load or build dataset
    if args.load_only:
        print("\nLoading processed dataset...")
        X, y = load_processed_dataset(args.output)
    else:
        print(f"\nBuilding dataset from: {args.dataset}")
        X, y = build_dataset(
            args.dataset,
            normalize=not args.no_normalize,
            validate=True
        )

    # Check if dataset is empty
    if len(X) == 0:
        print("\n⚠ No data found. Please collect data first using:")
        print("   python -m src.collect_data")
        return

    # Print summary
    print_dataset_summary(X, y)

    # Save processed dataset
    if not args.load_only:
        print(f"\nSaving to: {args.output}")
        save_dataset(X, y, args.output)

    print("\n✓ Preprocessing complete!")


if __name__ == "__main__":
    main()