"""
Data Collection Module

This module handles gesture data collection from webcam with real-time
quality validation for the Sign Language Translation project.

Features:
- Webcam capture with MediaPipe hand tracking
- Real-time quality validation (blur, brightness, visibility)
- Automatic image saving with proper naming convention
- Progress tracking for each class
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from datetime import datetime


class DataCollector:
    """Handles gesture data collection with quality validation."""

    # Gesture classes for MVP
    CLASSES = ["Halo", "Makan", "Minum", "Tolong", "TerimaKasih"]

    # Quality thresholds
    BLUR_THRESHOLD = 100.0
    MIN_BRIGHTNESS = 30
    MAX_BRIGHTNESS = 220
    MIN_HAND_VISIBILITY = 0.7  # Minimum fraction of hand landmarks visible
    MIN_HAND_AREA = 5000  # Minimum hand area in pixels

    def __init__(self, dataset_dir: str = "dataset"):
        """
        Initialize the data collector.

        Args:
            dataset_dir: Base directory for storing dataset
        """
        self.dataset_dir = Path(dataset_dir)
        self.current_class = None
        self.sample_counts = {cls: self._get_sample_count(cls) for cls in self.CLASSES}
        self.total_captured = 0
        self.total_rejected = 0

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Create dataset directories
        self._create_directories()

    def _create_directories(self):
        """Create dataset directories for each class."""
        for class_name in self.CLASSES:
            class_dir = self.dataset_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

    def _get_sample_count(self, class_name: str) -> int:
        """Get current sample count for a class."""
        class_dir = self.dataset_dir / class_name
        if class_dir.exists():
            return len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
        return 0

    def _calculate_blur(self, image: np.ndarray) -> float:
        """
        Calculate blur score using Laplacian variance.
        Higher variance = sharper image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _calculate_brightness(self, image: np.ndarray) -> float:
        """Calculate average brightness of the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)

    def _check_hand_visibility(self, hand_landmarks, image_shape: Tuple[int, int]) -> Tuple[bool, float]:
        """
        Check if hand is sufficiently visible and sized.

        Returns:
            Tuple of (is_visible, visibility_score)
        """
        if hand_landmarks is None:
            return False, 0.0

        h, w = image_shape[:2]

        # Get landmark coordinates
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]

        # Calculate bounding box
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # Calculate area
        hand_width = max_x - min_x
        hand_height = max_y - min_y
        hand_area = hand_width * hand_height * w * h

        # Check if all required landmarks are visible (not truncated)
        visible_count = sum(1 for lm in hand_landmarks.landmark if 0 <= lm.x <= 1 and 0 <= lm.y <= 1)
        visibility_ratio = visible_count / len(hand_landmarks.landmark)

        return hand_area >= self.MIN_HAND_AREA and visibility_ratio >= self.MIN_HAND_VISIBILITY, visibility_ratio

    def validate_frame(self, image: np.ndarray, hand_landmarks) -> Tuple[bool, str]:
        """
        Validate frame quality.

        Args:
            image: Input frame
            hand_landmarks: Detected hand landmarks

        Returns:
            Tuple of (is_valid, reason)
        """
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

        # Check hand visibility
        if hand_landmarks is None:
            return False, "No hand detected"

        is_visible, visibility_score = self._check_hand_visibility(hand_landmarks, image.shape)
        if not is_visible:
            return False, f"Hand not visible enough (score: {visibility_score:.2f})"

        return True, "Valid"

    def get_next_filename(self, class_name: str) -> str:
        """Get next filename for a class."""
        count = self.sample_counts[class_name] + 1
        return f"{class_name}_{count:03d}.jpg"

    def save_frame(self, image: np.ndarray, class_name: str) -> str:
        """
        Save frame to the appropriate class directory.

        Returns:
            Path to saved file
        """
        filename = self.get_next_filename(class_name)
        filepath = self.dataset_dir / class_name / filename
        cv2.imwrite(str(filepath), image)
        self.sample_counts[class_name] += 1
        self.total_captured += 1
        return str(filepath)

    def collect_samples(self, class_name: str, target_count: int = 200,
                       fps: int = 5, display: bool = True) -> Dict[str, int]:
        """
        Collect samples for a specific class.

        Args:
            class_name: Name of the gesture class
            target_count: Number of samples to collect
            fps: Frames per second to capture
            display: Whether to display the preview window

        Returns:
            Dictionary with sample counts per class
        """
        if class_name not in self.CLASSES:
            raise ValueError(f"Invalid class: {class_name}. Choose from {self.CLASSES}")

        self.current_class = class_name
        current_count = self.sample_counts[class_name]

        print(f"\n{'='*60}")
        print(f"Collecting samples for: {class_name}")
        print(f"Current samples: {current_count}/{target_count}")
        print(f"Press 'SPACE' to capture, 'Q' to quit, 'R' to restart count")
        print(f"{'='*60}\n")

        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot access webcam")

        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Initialize hand detection
        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        ) as hands:

            frame_interval = 1.0 / fps
            last_capture_time = time.time()
            capture_delay = 0.3  # Delay between captures to avoid duplicates

            while current_count < target_count:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Cannot read frame from webcam")
                    break

                # Flip frame horizontally for natural interaction
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect hands
                results = hands.process(rgb_frame)
                hand_landmarks = results.multi_hand_landmarks

                # Validate frame
                is_valid, reason = self.validate_frame(frame, hand_landmarks[0] if hand_landmarks else None)

                # Draw hand landmarks if detected
                if hand_landmarks:
                    for hl in hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hl,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )

                # Create status overlay
                status_color = (0, 255, 0) if is_valid else (0, 0, 255)
                status_text = "READY" if is_valid else reason

                # Draw UI elements
                cv2.rectangle(frame, (0, 0), (400, 120), (0, 0, 0), -1)
                cv2.putText(frame, f"Class: {class_name}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Progress: {current_count}/{target_count}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Status: {status_text}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

                # Show capture instructions
                cv2.putText(frame, "SPACE: Capture | Q: Quit | R: Reset", (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                if display:
                    cv2.imshow("Sign Language Data Collection", frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):  # Space to capture
                    current_time = time.time()
                    if is_valid and (current_time - last_capture_time) >= capture_delay:
                        filepath = self.save_frame(frame, class_name)
                        print(f"Captured: {filepath} (Total: {self.total_captured})")
                        current_count += 1
                        last_capture_time = current_time
                elif key == ord('r'):  # Reset count for current class
                    current_count = 0
                    print("Count reset!")

        cap.release()
        if display:
            cv2.destroyAllWindows()

        print(f"\nCollection complete for {class_name}: {current_count} samples")
        return self.sample_counts

    def batch_collect(self, target_per_class: int = 200, fps: int = 5) -> Dict[str, int]:
        """
        Interactively collect samples for all classes.

        Args:
            target_per_class: Target number of samples per class
            fps: Frames per second to capture

        Returns:
            Dictionary with sample counts per class
        """
        print("\n" + "="*60)
        print("BATCH DATA COLLECTION MODE")
        print("="*60)

        for class_name in self.CLASSES:
            remaining = target_per_class - self.sample_counts[class_name]
            if remaining > 0:
                print(f"\n[INFO] {class_name}: {self.sample_counts[class_name]} samples collected")
                print(f"[INFO] Need {remaining} more samples")
                self.collect_samples(class_name, target_per_class, fps)
            else:
                print(f"\n[INFO] {class_name}: Already has {self.sample_counts[class_name]} samples, skipping")

        return self.sample_counts

    def get_stats(self) -> Dict[str, any]:
        """Get current dataset statistics."""
        total = sum(self.sample_counts.values())
        min_count = min(self.sample_counts.values())
        max_count = max(self.sample_counts.values())

        return {
            "classes": len(self.CLASSES),
            "class_names": self.CLASSES,
            "samples_per_class": self.sample_counts,
            "total_samples": total,
            "min_samples": min_count,
            "max_samples": max_count,
            "is_balanced": (max_count - min_count) / max_count <= 0.1 if max_count > 0 else False,
            "total_captured": self.total_captured,
            "total_rejected": self.total_rejected
        }

    def print_stats(self):
        """Print current dataset statistics."""
        stats = self.get_stats()
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        print(f"Total Classes: {stats['classes']}")
        print(f"Total Samples: {stats['total_samples']}")
        print("\nSamples per class:")
        for cls, count in stats['samples_per_class'].items():
            print(f"  {cls:15s}: {count}")
        print(f"\nBalance Status: {'BALANCED' if stats['is_balanced'] else 'UNBALANCED'}")
        print(f"Total Captured: {stats['total_captured']}")
        print(f"Total Rejected: {stats['total_rejected']}")
        print("="*60)


def main():
    """Main entry point for data collection."""
    import argparse

    parser = argparse.ArgumentParser(description="Sign Language Data Collection")
    parser.add_argument("--class", dest="class_name", choices=DataCollector.CLASSES,
                       help="Specific class to collect data for")
    parser.add_argument("--target", type=int, default=200,
                       help="Target samples per class (default: 200)")
    parser.add_argument("--fps", type=int, default=5,
                       help="Capture rate in fps (default: 5)")
    parser.add_argument("--no-display", action="store_true",
                       help="Disable display window")
    parser.add_argument("--stats", action="store_true",
                       help="Show statistics and exit")

    args = parser.parse_args()

    collector = DataCollector()

    if args.stats:
        collector.print_stats()
        return

    if args.class_name:
        collector.collect_samples(args.class_name, args.target, args.fps, not args.no_display)
    else:
        collector.batch_collect(args.target, args.fps)

    collector.print_stats()


if __name__ == "__main__":
    main()