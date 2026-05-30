"""
Dataset Report Generator

Generates comprehensive reports for the sign language gesture dataset,
including metadata about recording conditions and data collection process.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import platform


class DatasetReportGenerator:
    """Generates comprehensive dataset reports."""

    CLASSES = ["Halo", "Makan", "Minum", "Tolong", "TerimaKasih"]

    def __init__(self, dataset_dir: str = "dataset", collector_name: str = "Unknown"):
        """
        Initialize report generator.

        Args:
            dataset_dir: Path to dataset directory
            collector_name: Name of the data collector
        """
        self.dataset_dir = Path(dataset_dir)
        self.collector_name = collector_name
        self.report_data = {}

    def collect_metadata(self) -> Dict[str, any]:
        """Collect dataset metadata."""
        # Count samples per class
        sample_counts = {}
        for class_name in self.CLASSES:
            class_dir = self.dataset_dir / class_name
            if class_dir.exists():
                jpg_count = len(list(class_dir.glob("*.jpg")))
                png_count = len(list(class_dir.glob("*.png")))
                sample_counts[class_name] = jpg_count + png_count
            else:
                sample_counts[class_name] = 0

        total_samples = sum(sample_counts.values())

        # Analyze image properties
        image_info = self._analyze_images()

        # Calculate statistics
        stats = self._calculate_statistics(sample_counts)

        metadata = {
            "collection_date": datetime.now().strftime("%Y-%m-%d"),
            "collection_time": datetime.now().strftime("%H:%M:%S"),
            "data_collector": self.collector_name,
            "system_info": {
                "platform": platform.system(),
                "python_version": platform.python_version(),
            },
            "dataset_location": str(self.dataset_dir.absolute()),
            "classes": self.CLASSES,
            "total_classes": len(self.CLASSES),
            "samples_per_class": sample_counts,
            "total_samples": total_samples,
            "image_info": image_info,
            "statistics": stats
        }

        self.report_data = metadata
        return metadata

    def _analyze_images(self) -> Dict[str, any]:
        """Analyze sample images to gather information."""
        import cv2

        info = {
            "total_images": 0,
            "formats": {"jpg": 0, "png": 0},
            "sample_resolutions": [],
            "average_brightness": [],
            "has_images": False
        }

        for class_name in self.CLASSES:
            class_dir = self.dataset_dir / class_name
            if not class_dir.exists():
                continue

            # Sample some images to analyze
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            info["total_images"] += len(images)

            for img_path in images[:5]:  # Sample first 5 per class
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        h, w = img.shape[:2]
                        info["sample_resolutions"].append({"width": w, "height": h})
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        info["average_brightness"].append(float(gray.mean()))
                        info["has_images"] = True

                        ext = img_path.suffix.lower().lstrip('.')
                        if ext in info["formats"]:
                            info["formats"][ext] += 1
                        else:
                            info["formats"][ext] = info["formats"].get(ext, 0) + 1
                except Exception:
                    continue

        return info

    def _calculate_statistics(self, sample_counts: Dict[str, int]) -> Dict[str, any]:
        """Calculate dataset statistics."""
        values = list(sample_counts.values())

        if not values or max(values) == 0:
            return {
                "min_samples": 0,
                "max_samples": 0,
                "mean_samples": 0,
                "std_deviation": 0,
                "is_balanced": False,
                "balance_difference": 0,
                "balance_difference_percent": 100
            }

        import numpy as np

        min_val = min(values)
        max_val = max(values)
        mean_val = np.mean(values)
        std_val = np.std(values)

        balance_diff = max_val - min_val
        balance_percent = (balance_diff / max_val * 100) if max_val > 0 else 100

        return {
            "min_samples": min_val,
            "max_samples": max_val,
            "mean_samples": float(mean_val),
            "std_deviation": float(std_val),
            "is_balanced": balance_percent <= 10,
            "balance_difference": balance_diff,
            "balance_difference_percent": round(balance_percent, 2)
        }

    def check_recording_conditions(self) -> Dict[str, any]:
        """Check for recording condition notes based on image analysis."""
        conditions = {
            "lighting_variation": "Unknown",
            "background_consistency": "Unknown",
            "hand_position_variation": "Unknown",
            "recommended": True,
            "notes": []
        }

        # Analyze brightness variation as proxy for lighting conditions
        if self.report_data.get("image_info", {}).get("average_brightness"):
            brightness_values = self.report_data["image_info"]["average_brightness"]
            import numpy as np

            brightness_std = np.std(brightness_values)
            if brightness_std > 40:
                conditions["lighting_variation"] = "High variation"
                conditions["notes"].append("Good: Diverse lighting conditions captured")
            elif brightness_std > 20:
                conditions["lighting_variation"] = "Moderate variation"
                conditions["notes"].append("Lighting conditions are moderate")
            else:
                conditions["lighting_variation"] = "Low variation"
                conditions["notes"].append("Consider capturing in different lighting conditions")

        # Check for resolution consistency
        if self.report_data.get("image_info", {}).get("sample_resolutions"):
            resolutions = self.report_data["image_info"]["sample_resolutions"]
            if resolutions:
                first_res = resolutions[0]
                all_same = all(r["width"] == first_res["width"] and r["height"] == first_res["height"]
                             for r in resolutions)
                conditions["background_consistency"] = "Consistent" if all_same else "Variable"

        # Check sample counts for recommended minimum
        total = self.report_data.get("total_samples", 0)
        if total >= 1000:
            conditions["notes"].append("Excellent: 200+ samples per class achieved")
            conditions["recommended"] = True
        elif total >= 500:
            conditions["notes"].append("Good: 100+ samples per class achieved")
            conditions["recommended"] = True
        elif total >= 100 * len(self.CLASSES):
            conditions["notes"].append("Acceptable: Minimum 100 samples per class achieved")
            conditions["recommended"] = True
        else:
            conditions["notes"].append("Warning: Need more samples for better model training")
            conditions["recommended"] = False

        return conditions

    def generate_markdown_report(self, output_path: str = "DATASET_REPORT.md"):
        """Generate a markdown report."""
        metadata = self.collect_metadata()
        conditions = self.check_recording_conditions()

        report_lines = [
            "# Dataset Report\n",
            "---",
            f"\n**Generated:** {metadata['collection_date']} {metadata['collection_time']}",
            f"\n**Data Collector:** {metadata['data_collector']}",
            f"\n**Dataset Location:** `{metadata['dataset_location']}`",
            "\n---\n",
            "## Dataset Summary\n",
            f"| Property | Value |",
            "|---|---|",
            f"| Total Classes | {metadata['total_classes']} |",
            f"| Total Images | {metadata['total_samples']} |",
            f"| Collection Date | {metadata['collection_date']} |",
            "\n---\n",
            "## Samples Per Class\n",
            "\n| Class | Samples |",
            "|---|---|\n",
        ]

        for class_name, count in metadata['samples_per_class'].items():
            report_lines.append(f"| {class_name} | {count} |")

        report_lines.extend([
            "\n---\n",
            "## Statistics\n",
            f"- **Minimum samples:** {metadata['statistics']['min_samples']}",
            f"- **Maximum samples:** {metadata['statistics']['max_samples']}",
            f"- **Mean samples:** {metadata['statistics']['mean_samples']:.1f}",
            f"- **Standard deviation:** {metadata['statistics']['std_deviation']:.2f}",
            f"- **Balance difference:** {metadata['statistics']['balance_difference_percent']}%",
            "\n---\n",
            "## Recording Conditions\n",
            f"- **Lighting variation:** {conditions['lighting_variation']}",
            f"- **Background consistency:** {conditions['background_consistency']}",
            "\n**Notes:**\n",
        ])

        for note in conditions['notes']:
            report_lines.append(f"- {note}")

        report_lines.extend([
            "\n---\n",
            "## Image Information\n",
        ])

        if metadata['image_info']['has_images']:
            report_lines.extend([
                f"- **Image formats:** {', '.join([f'{k}: {v}' for k, v in metadata['image_info']['formats'].items()])}",
                f"- **Total images:** {metadata['image_info']['total_images']}",
            ])

            if metadata['image_info']['sample_resolutions']:
                res = metadata['image_info']['sample_resolutions'][0]
                report_lines.append(f"- **Sample resolution:** {res['width']}x{res['height']}")
        else:
            report_lines.append("*No images collected yet*")

        report_lines.extend([
            "\n---\n",
            "## Status\n",
        ])

        balance_status = "✓ BALANCED" if metadata['statistics']['is_balanced'] else "✗ UNBALANCED"
        ready_status = "✓ READY FOR PREPROCESSING" if conditions['recommended'] else "✗ NEEDS MORE DATA"

        report_lines.extend([
            f"\n- **Class balance:** {balance_status}",
            f"- **Dataset readiness:** {ready_status}",
            "\n---\n",
            "*Report generated by Dataset Report Generator*\n"
        ])

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"Report saved to: {output_path}")
        return output_path

    def generate_json_report(self, output_path: str = "dataset_report.json"):
        """Generate a JSON report."""
        metadata = self.collect_metadata()
        conditions = self.check_recording_conditions()

        report = {
            "report_type": "Dataset Collection Report",
            "metadata": metadata,
            "recording_conditions": conditions,
            "generated_at": datetime.now().isoformat()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        print(f"JSON report saved to: {output_path}")
        return report

    def print_summary(self):
        """Print a summary to console."""
        metadata = self.collect_metadata()
        conditions = self.check_recording_conditions()

        print("\n" + "="*60)
        print("DATASET REPORT")
        print("="*60)

        print(f"\nCollection Date: {metadata['collection_date']}")
        print(f"Data Collector: {metadata['data_collector']}")
        print(f"Location: {metadata['dataset_location']}")

        print("\n--- Classes ---")
        for cls, count in metadata['samples_per_class'].items():
            bar = "█" * (count // 10) + "░" * (20 - count // 10)
            print(f"  {cls:15s} [{bar}] {count}")

        print("\n--- Summary ---")
        print(f"  Total Classes: {metadata['total_classes']}")
        print(f"  Total Images: {metadata['total_samples']}")
        print(f"  Min samples: {metadata['statistics']['min_samples']}")
        print(f"  Max samples: {metadata['statistics']['max_samples']}")

        balance_status = "✓ BALANCED" if metadata['statistics']['is_balanced'] else "✗ UNBALANCED"
        ready_status = "✓ READY" if conditions['recommended'] else "✗ NEEDS DATA"

        print(f"\n--- Status ---")
        print(f"  Balance: {balance_status}")
        print(f"  Preprocessing: {ready_status}")

        if conditions['notes']:
            print("\n--- Notes ---")
            for note in conditions['notes']:
                print(f"  - {note}")

        print("\n" + "="*60)


def generate_report(dataset_dir: str = "dataset",
                   collector_name: str = "ML Team",
                   output_md: str = "DATASET_REPORT.md",
                   output_json: str = "dataset_report.json") -> Dict:
    """
    Generate comprehensive dataset reports.

    Args:
        dataset_dir: Path to dataset directory
        collector_name: Name of data collector
        output_md: Path for markdown report
        output_json: Path for JSON report

    Returns:
        Report data dictionary
    """
    generator = DatasetReportGenerator(dataset_dir, collector_name)

    # Generate both reports
    generator.generate_json_report(output_json)
    generator.generate_markdown_report(output_md)
    generator.print_summary()

    return generator.report_data


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Dataset Report Generator")
    parser.add_argument("--dataset", default="dataset",
                       help="Dataset directory path")
    parser.add_argument("--collector", default="ML Team",
                       help="Data collector name")
    parser.add_argument("--output-md", default="DATASET_REPORT.md",
                       help="Markdown output path")
    parser.add_argument("--output-json", default="dataset_report.json",
                       help="JSON output path")
    parser.add_argument("--summary-only", action="store_true",
                       help="Print summary only")

    args = parser.parse_args()

    generator = DatasetReportGenerator(args.dataset, args.collector)

    if args.summary_only:
        generator.print_summary()
    else:
        generator.generate_json_report(args.output_json)
        generator.generate_markdown_report(args.output_md)
        generator.print_summary()


if __name__ == "__main__":
    main()