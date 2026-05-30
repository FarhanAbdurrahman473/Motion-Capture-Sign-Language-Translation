"""
Hand Tracking Module

This module handles real-time hand tracking using MediaPipe for the
Sign Language Translation project.

Features:
- Webcam capture with real-time hand detection
- MediaPipe Hands for 21-landmark hand tracking
- Landmark visualization with skeleton drawing
- FPS and hand count display
- Helper function for dataset compatibility (extract_landmarks)

Usage:
    python -m src.capture

For data collection, use: python -m src.collect_data
"""

import cv2
import mediapipe as mp
import time
from typing import List, Tuple, Optional


def extract_landmarks(hand_landmarks) -> List[float]:
    """
    Extract hand landmarks into a flat list for dataset compatibility.

    This function converts MediaPipe hand landmarks into a standardized
    format that will be used by:
    - preprocess.py (data normalization)
    - train.py (model training)
    - predict.py (gesture prediction)

    Args:
        hand_landmarks: MediaPipe hand_landmarks object containing 21 landmarks

    Returns:
        List of 63 float values: [x1, y1, z1, x2, y2, z2, ..., x21, y21, z21]
        Returns empty list if hand_landmarks is None
    """
    if hand_landmarks is None:
        return []

    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])

    return landmarks  # 63 features per hand


def get_hand_label(handedness) -> str:
    """
    Determine if detected hand is Left or Right.

    Args:
        handedness: MediaPipe handedness classification result

    Returns:
        String label: "Left" or "Right"
    """
    if handedness is None:
        return "Unknown"

    # Get the classification label
    label = handedness.classification[0].label
    # MediaPipe returns "Left" or "Right" - return as-is
    return label


def calculate_bounding_box(landmarks, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Calculate bounding box around detected hand landmarks.

    Args:
        landmarks: List of landmark points (each with x, y, z)
        frame_shape: Shape of the frame (height, width, channels)

    Returns:
        Tuple of (x_min, y_min, x_max, y_max) in pixel coordinates
    """
    h, w = frame_shape[:2]

    x_coords = [lm.x * w for lm in landmarks.landmark]
    y_coords = [lm.y * h for lm in landmarks.landmark]

    x_min = int(min(x_coords))
    y_min = int(min(y_coords))
    x_max = int(max(x_coords))
    y_max = int(max(y_coords))

    # Add padding
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)

    return x_min, y_min, x_max, y_max


class HandTracker:
    """Real-time hand tracking using MediaPipe."""

    def __init__(self,
                 static_image_mode: bool = False,
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.7):
        """
        Initialize MediaPipe Hands solution.

        Args:
            static_image_mode: If False, runs in real-time tracking mode
            max_num_hands: Maximum number of hands to detect (1-4)
            min_detection_confidence: Minimum detection confidence (0.0-1.0)
            min_tracking_confidence: Minimum tracking confidence (0.0-1.0)
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize MediaPipe Hands with specified parameters
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Drawing styles
        self.landmark_style = self.mp_drawing_styles.get_default_hand_landmarks_style()
        self.connection_style = self.mp_drawing_styles.get_default_hand_connections_style()

    def process_frame(self, frame: cv2.Mat) -> Tuple:
        """
        Process a single frame to detect hands.

        Args:
            frame: Input frame in BGR format

        Returns:
            Tuple of (results, annotated_frame)
            - results: MediaPipe hands detection results
            - annotated_frame: Frame with landmarks drawn
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame for hand detection
        results = self.hands.process(rgb_frame)

        # Draw hand landmarks on frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw connections and landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.landmark_style,
                    self.connection_style
                )

        return results, frame

    def get_landmarks(self, results) -> List:
        """
        Extract all detected hand landmarks from results.

        Args:
            results: MediaPipe hands detection results

        Returns:
            List of hand landmarks for each detected hand
        """
        if results.multi_hand_landmarks is None:
            return []

        return results.multi_hand_landmarks

    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()


def main():
    """
    Main entry point for hand tracking demonstration.

    Opens webcam and displays real-time hand tracking with:
    - 21 hand landmarks per detected hand
    - FPS counter
    - Hand count display
    - Optional: Hand labels (Left/Right), bounding boxes

    Controls:
        'q' - Quit application
        's' - Print landmark vectors to console
        'h' - Toggle hand labels
        'b' - Toggle bounding boxes
    """
    # Initialize hand tracker with specified parameters
    tracker = HandTracker(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return

    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # FPS calculation variables
    fps = 0
    frame_count = 0
    start_time = time.time()

    # Optional features
    show_hand_labels = True
    show_bounding_boxes = True

    print("\n" + "=" * 50)
    print("HAND TRACKING MODULE - MediaPipe")
    print("=" * 50)
    print("Controls:")
    print("  [q] - Quit")
    print("  [s] - Print landmarks to console")
    print("  [h] - Toggle hand labels")
    print("  [b] - Toggle bounding boxes")
    print("=" * 50 + "\n")

    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame from webcam")
                break

            # Flip frame horizontally for natural interaction (mirror effect)
            frame = cv2.flip(frame, 1)

            # Process frame for hand detection
            results, annotated_frame = tracker.process_frame(frame)

            # Get number of hands detected
            num_hands = 0
            hand_labels = []
            if results.multi_hand_landmarks:
                num_hands = len(results.multi_hand_landmarks)

                # Get hand labels if available
                if results.multi_handedness and show_hand_labels:
                    for handedness in results.multi_handedness:
                        label = get_hand_label(handedness)
                        hand_labels.append(label)

            # Draw bounding boxes and labels for each hand
            if results.multi_hand_landmarks and show_bounding_boxes:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Calculate bounding box
                    x_min, y_min, x_max, y_max = calculate_bounding_box(
                        hand_landmarks, frame.shape
                    )

                    # Draw bounding box
                    cv2.rectangle(annotated_frame,
                                (x_min, y_min),
                                (x_max, y_max),
                                (0, 255, 0), 2)

                    # Draw hand label
                    if show_hand_labels and idx < len(hand_labels):
                        label = hand_labels[idx]
                        cv2.putText(annotated_frame,
                                   label,
                                   (x_min, y_min - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   0.8, (0, 255, 0), 2)

            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()

            # Draw debug information on frame
            # Background rectangle for text
            cv2.rectangle(annotated_frame, (10, 10), (250, 80), (0, 0, 0), -1)

            # Display FPS
            cv2.putText(annotated_frame,
                       f"FPS: {int(fps)}",
                       (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 255), 2)

            # Display hand count
            cv2.putText(annotated_frame,
                       f"Hands: {num_hands}",
                       (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 0), 2)

            # Display controls hint
            cv2.putText(annotated_frame,
                       "q: Quit | s: Export | h: Labels | b: Boxes",
                       (10, annotated_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (200, 200, 200), 1)

            # Show frame
            cv2.imshow("Hand Tracking - MediaPipe", annotated_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                # Quit application
                print("\nQuitting hand tracking...")
                break

            elif key == ord('s'):
                # Export landmarks to console
                if results.multi_hand_landmarks:
                    print("\n" + "-" * 50)
                    print(f"EXPORTING LANDMARKS ({len(results.multi_hand_landmarks)} hand(s))")
                    print("-" * 50)

                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        # Get hand label
                        label = "Unknown"
                        if results.multi_handedness and idx < len(results.multi_handedness):
                            label = get_hand_label(results.multi_handedness[idx])

                        print(f"\nHand {idx + 1} ({label}):")

                        # Extract and display landmarks
                        landmarks = extract_landmarks(hand_landmarks)
                        print(f"  Total features: {len(landmarks)}")

                        # Print in formatted groups of 3 (x, y, z)
                        for i in range(0, len(landmarks), 9):
                            chunk = landmarks[i:i+9]
                            formatted = [f"{v:.4f}" for v in chunk]
                            print(f"  {formatted}")

                    print("-" * 50)
                else:
                    print("No hands detected - cannot export landmarks")

            elif key == ord('h'):
                # Toggle hand labels
                show_hand_labels = not show_hand_labels
                print(f"Hand labels: {'ON' if show_hand_labels else 'OFF'}")

            elif key == ord('b'):
                # Toggle bounding boxes
                show_bounding_boxes = not show_bounding_boxes
                print(f"Bounding boxes: {'ON' if show_bounding_boxes else 'OFF'}")

    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        print("Hand tracking closed successfully.")


if __name__ == "__main__":
    main()
