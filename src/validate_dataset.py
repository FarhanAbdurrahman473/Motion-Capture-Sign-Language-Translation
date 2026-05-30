"""
Dataset Validation Module

Validates the collected gesture dataset for:
- Number of samples per class
- Missing classes
- Empty folders
- Class imbalance
- Image quality checks
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json


class DatasetValidator:
    """Validates dataset integrity and quality."""

    CLASSES = ["Halo", "Makan", "Minum", "Tolong", "TerimaKasih"]

    # Quality thresholds
    BLUR_THRESHOLD = 80.0
    MIN_BRIGHTNESS = 25
    MAX_BRIGHTNESS = 225
    IMBALANCE_THRESHOLD = 0.1  # 10% maximum allowed difference

    def __init__(self, dataset_dir: str = "dataset"):
        """
        Initialize the validator.

        Args:
            dataset_dir: Path to dataset directory
        """
        self.dataset_dir = Path(dataset_dir)
        self.results = {}

    def _calculate_blur(self, image: np.ndarray) -> float:
        """Calculate blur score using Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _calculate_brightness(self, image: np.ndarray) -> float:
        """Calculate average brightness."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)

    def check_directory_structure(self) -> Dict[str, any]:
        """Check if all required directories exist."""
        missing = []
        empty = []
        present = []

        for class_name in self.CLASSES:
            class_dir = self.dataset_dir / class_name
            if not class_dir.exists():
                missing.append(class_name)
            elif not any(class_dir.iterdir()):
                empty.append(class_name)
            else:
                present.append(class_name)

        return {
            "missing": missing,
            "empty": empty,
            "present": present,
            "structure_valid": len(missing) == 0
        }

    def count_samples(self) -> Dict[str, int]:
        """Count samples in each class directory."""
        counts = {}
        for class_name in self.CLASSES:
            class_dir = self.dataset_dir / class_name
            if class_dir.exists():
                counts[class_name] = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
            else:
                counts[class_name] = 0
        return counts

    def validate_class_balance(self, counts: Dict[str, int]) -> Dict[str, any]:
        """Check if classes are balanced within threshold."""
        values = list(counts.values())

        if not values or max(values) == 0:
            return {
                "is_balanced": False,
                "min": 0,
                "max": 0,
                "difference": 0,
                "difference_percent": 100,
                "status": "No samples collected"
            }

        min_count = min(values)
        max_count = max(values)

        if max_count == 0:
            return {
                "is_balanced": False,
                "min": 0,
                "max": 0,
                "difference": 0,
                "difference_percent": 100,
                "status": "No samples"
            }

        difference = max_count - min_count
        difference_percent = (difference / max_count) * 100

        return {
            "is_balanced": difference_percent <= (self.IMBALANCE_THRESHOLD * 100),
            "min": min_count,
            "max": max_count,
            "difference": difference,
            "difference_percent": difference_percent,
            "status": "Balanced" if difference_percent <= (self.IMBALANCE_THRESHOLD * 100) else "Unbalanced"
        }

    def validate_image_quality(self, image_path: Path) -> Tuple[bool, str]:
        """
        Validate a single image for quality issues.

        Returns:
            Tuple of (is_valid, issue_description)
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return False, "Cannot read image"

            # Check blur
            blur_score = self._calculate_blur(image)
            if blur_score < self.BLUR_THRESHOLD:
                return False, f"Blurry (blur: {blur_score:.1f})"

            # Check brightness
            brightness = self._calculate_brightness(image)
            if brightness < self.MIN_BRIGHTNESS:
                return False, f"Too dark (brightness: {brightness:.1f})"
            if brightness > self.MAX_BRIGHTNESS:
                return False, f"Too bright (brightness: {brightness:.1f})"

            return True, "Valid"

        except Exception as e:
            return False, f"Error: {str(e)}"

    def validate_all_images(self) -> Dict[str, any]:
        """Validate all images in the dataset."""
        results = {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "issues": {
                "blurry": [],
                "dark": [],
                "bright": [],
                "error": []
            },
            "invalid_samples": []
        }

        for class_name in self.CLASSES:
            class_dir = self.dataset_dir / class_name
            if not class_dir.exists():
                continue

            for img_path in class_dir.glob("*.jpg"):
                results["total"] += 1
                is_valid, reason = self.validate_image_quality(img_path)

                if is_valid:
                    results["valid"] += 1
                else:
                    results["invalid"] += 1
                    results["invalid_samples"].append({
                        "path": str(img_path),
                        "reason": reason,
                        "class": class_name
                    })

                    # Categorize issue
                    if "Blurry" in reason:
                        results["issues"]["blurry"].append(str(img_path))
                    elif "dark" in reason:
                        results["issues"]["dark"].append(str(img_path))
                    elif "bright" in reason:
                        results["issues"]["bright"].append(str(img_path))
                    else:
                        results["issues"]["error"].append(str(img_path))

            # Also check PNG files
            for img_path in class_dir.glob("*.png"):
                results["total"] += 1
                is_valid, reason = self.validate_image_quality(img_path)

                if is_valid:
                    results["valid"] += 1
                else:
                    results["invalid"] += 1
                    results["invalid_samples"].append({
                        "path": str(img_path),
                        "reason": reason,
                        "class": class_name
                    })

        return results

    def validate_naming_convention(self) -> Dict[str, any]:
        """Check if files follow naming convention."""
        results = {
            "correct": [],
            "incorrect": []
        }

        for class_name in self.CLASSES:
            class_dir = self.dataset_dir / class_name
            if not class_dir.exists():
                continue

            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue

                # Expected pattern: <ClassName>_<index>.jpg
                expected_prefix = f"{class_name}_"
                if img_path.stem.startswith(expected_prefix):
                    results["correct"].append(str(img_path))
                else:
                    results["incorrect"].append({
                        "path": str(img_path),
                        "expected_pattern": f"{class_name}_XXX.jpg",
                        "actual_name": img_path.name
                    })

        return {
            "correct_count": len(results["correct"]),
            "incorrect_count": len(results["incorrect"]),
            "incorrect": results["incorrect"]
        }

    def validate_dataset(self, check_quality: bool = False) -> Dict[str, any]:
        """
        Run full dataset validation.

        Args:
            check_quality: Whether to check individual image quality

        Returns:
            Complete validation report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "dataset_dir": str(self.dataset_dir),
            "structure": self.check_directory_structure(),
            "sample_counts": self.count_samples(),
            "balance": self.validate_class_balance(self.count_samples()),
            "naming": self.validate_naming_convention(),
            "quality": None
        }

        if check_quality:
            report["quality"] = self.validate_all_images()

        self.results = report
        return report

    def print_report(self, report: Optional[Dict[str, any]] = None):
        """Print validation report in human-readable format."""
        if report is None:
            report = self.results

        print("\n" + "="*60)
        print("DATASET VALIDATION REPORT")
        print("="*60)
        print(f"\nTimestamp: {report['timestamp']}")
        print(f"Dataset: {report['dataset_dir']}")

        # Structure check
        print("\n--- Directory Structure ---")
        structure = report["structure"]
        if structure["structure_valid"]:
            print("✓ All required class directories present")
        else:
            print("✗ Missing directories:", structure["missing"])
            print("✗ Empty directories:", structure["empty"])

        # Sample counts
        print("\n--- Sample Counts ---")
        counts = report["sample_counts"]
        for cls, count in counts.items():
            print(f"  {cls:15s}: {count:4d}")

        # Balance check
        print("\n--- Class Balance ---")
        balance = report["balance"]
        print(f"  Min samples: {balance['min']}")
        print(f"  Max samples: {balance['max']}")
        print(f"  Difference: {balance['difference']} ({balance['difference_percent']:.1f}%)")
        print(f"  Status: {'✓ ' + balance['status'] if balance['is_balanced'] else '✗ ' + balance['status']}")

        # Naming convention
        print("\n--- Naming Convention ---")
        naming = report["naming"]
        print(f"  Correct: {naming['correct_count']}")
        print(f"  Incorrect: {naming['incorrect_count']}")

        if naming['incorrect_count'] > 0:
            print("  Issues:")
            for item in naming['incorrect'][:5]:  # Show first 5
                print(f"    - {item['path']}: expected '{item['expected_pattern']}'")

        # Quality check
        if report.get("quality"):
            print("\n--- Image Quality ---")
            quality = report["quality"]
            print(f"  Total images: {quality['total']}")
            print(f"  Valid: {quality['valid']}")
            print(f"  Invalid: {quality['invalid']}")

            if quality['invalid'] > 0:
                print("\n  Issue breakdown:")
                for issue_type, paths in quality['issues'].items():
                    if paths:
                        print(f"    {issue_type}: {len(paths)}")

        # Overall status
        print("\n--- Overall Status ---")
        is_valid = structure["structure_valid"] and balance["is_balanced"]
        status = "✓ READY FOR PREPROCESSING" if is_valid else "✗ NEEDS ATTENTION"
        print(status)

        print("="*60)

    def save_report(self, filepath: str, report: Optional[Dict[str, any]] = None):
        """Save validation report to JSON file."""
        if report is None:
            report = self.results

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Report saved to: {filepath}")

    def is_ready_for_training(self) -> Tuple[bool, List[str]]:
        """
        Check if dataset is ready for preprocessing/training.

        Returns:
            Tuple of (is_ready, list_of_issues)
        """
        issues = []

        # Check structure
        structure = self.check_directory_structure()
        if not structure["structure_valid"]:
            issues.append("Missing class directories")

        # Check minimum samples
        counts = self.count_samples()
        min_count = min(counts.values())
        if min_count < 100:
            issues.append(f"Minimum 100 samples per class required (current min: {min_count})")

        # Check balance
        balance = self.validate_class_balance(counts)
        if not balance["is_balanced"]:
            issues.append(f"Classes are unbalanced (difference: {balance['difference_percent']:.1f}%)")

        # Check naming
        naming = self.validate_naming_convention()
        if naming["incorrect_count"] > 0:
            issues.append(f"{naming['incorrect_count']} files don't follow naming convention")

        return len(issues) == 0, issues


def generate_dataset_report(dataset_dir: str = "dataset", output_path: str = "dataset_report.json",
                           check_quality: bool = False) -> Dict[str, any]:
    """
    Generate a complete dataset report.

    Args:
        dataset_dir: Path to dataset directory
        output_path: Path to save JSON report
        check_quality: Whether to check individual image quality

    Returns:
        Complete report dictionary
    """
    validator = DatasetValidator(dataset_dir)
    report = validator.validate_dataset(check_quality=check_quality)
    validator.print_report(report)
    validator.save_report(output_path, report)

    # Check readiness
    is_ready, issues = validator.is_ready_for_training()

    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Classes: {len(validator.CLASSES)}")
    print(f"Total Images: {report['sample_counts']}")

    total = sum(report['sample_counts'].values())
    print(f"\nTotal Samples: {total}")

    if is_ready:
        print("\n✓ Dataset is READY for preprocessing")
    else:
        print("\n✗ Dataset needs attention:")
        for issue in issues:
            print(f"  - {issue}")

    return report


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Dataset Validation Tool")
    parser.add_argument("--dataset", default="dataset",
                       help="Dataset directory path")
    parser.add_argument("--output", default="dataset_report.json",
                       help="Output report path")
    parser.add_argument("--check-quality", action="store_true",
                       help="Check individual image quality")
    parser.add_argument("--stats-only", action="store_true",
                       help="Show only basic statistics")

    args = parser.parse_args()

    validator = DatasetValidator(args.dataset)

    if args.stats_only:
        counts = validator.count_samples()
        balance = validator.validate_class_balance(counts)
        print("\n" + "="*40)
        print("DATASET STATISTICS")
        print("="*40)
        print(f"Classes: {len(validator.CLASSES)}")
        total = sum(counts.values())
        print(f"Total Samples: {total}\n")
        for cls, count in counts.items():
            print(f"  {cls:15s}: {count}")
        print(f"\nBalance: {balance['status']}")
        print("="*40)
    else:
        generate_dataset_report(args.dataset, args.output, args.check_quality)


if __name__ == "__main__":
    main()