"""Inference module for object detection."""

import argparse
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

from config import Config
from model import ObjectDetectionModel
from utils import (
    load_image, resize_image, normalize_image, nms, draw_boxes,
    ensure_dir
)


class Detector:
    """Object detection inference class."""
    
    def __init__(self, model_path: str, config: Config):
        """
        Initialize detector with trained model.
        
        Args:
            model_path: Path to trained model
            config: Configuration object
        """
        self.model = tf.keras.models.load_model(model_path)
        self.config = config
        
        # COCO class names
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant'
        ] + ['class_{}'.format(i) for i in range(20, 80)]
        
        self.confidence_threshold = config.get('inference.confidence_threshold')
        self.iou_threshold = config.get('inference.iou_threshold')
        self.input_shape = tuple(config.get('model.input_shape'))
    
    def detect_image(self, image_path: str) -> List[Dict]:
        """
        Detect objects in image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of detections with keys: x1, y1, x2, y2, confidence, class_name, class_id
        """
        # Load and preprocess image
        image = load_image(image_path)
        original_h, original_w = image.shape[:2]
        
        # Resize and normalize
        prepared_image, scale = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        normalized_image = normalize_image(prepared_image)
        
        # Inference
        batch = np.expand_dims(normalized_image, axis=0)
        bbox_pred, conf_pred, class_pred = self.model.predict(batch, verbose=0)
        
        # Process predictions
        detections = self._process_predictions(
            bbox_pred[0], conf_pred[0], class_pred[0],
            original_h, original_w, scale
        )
        
        return detections
    
    def detect_video(self, video_path: str, output_path: str = None) -> None:
        """
        Detect objects in video file.
        
        Args:
            video_path: Path to video file
            output_path: Path to save output video (None for no output)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output video writer if needed
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {video_path}")
        print(f"Dimensions: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
        
        frame_count = 0
        pbar = tqdm(total=total_frames, desc="Processing frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            detections = self._detect_frame(frame)
            
            # Draw detections
            result_frame = draw_boxes(frame, detections)
            
            # Write to output video
            if out:
                out.write(result_frame)
            
            # Save first frame for preview
            if frame_count == 0:
                cv2.imwrite(str(Path(self.config.get('output.results_dir')) / 'frame_0_preview.jpg'),
                           result_frame)
            
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        if out:
            out.release()
            print(f"Output video saved: {output_path}")
        
        print(f"Processed {frame_count} frames")
    
    def detect_webcam(self, duration: int = 30):
        """
        Real-time object detection from webcam.
        
        Args:
            duration: Duration in seconds (0 for infinite)
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise ValueError("Failed to open webcam")
        
        print("Starting webcam detection (press 'q' to quit)...")
        print("FPS display will show inference speed")
        
        import time
        start_time = time.time()
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            detections = self._detect_frame(frame)
            
            # Draw detections
            result_frame = draw_boxes(frame, detections)
            
            # Calculate and display FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result_frame, f"Detections: {len(detections)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Object Detection', result_frame)
            
            # Check for quit or duration
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if duration > 0 and elapsed > duration:
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _detect_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in a single frame.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detections
        """
        original_h, original_w = frame.shape[:2]
        
        # Resize and normalize
        prepared_frame, scale = resize_image(frame, (self.input_shape[1], self.input_shape[0]))
        normalized_frame = normalize_image(prepared_frame)
        
        # Inference
        batch = np.expand_dims(normalized_frame, axis=0)
        bbox_pred, conf_pred, class_pred = self.model.predict(batch, verbose=0)
        
        # Process predictions
        detections = self._process_predictions(
            bbox_pred[0], conf_pred[0], class_pred[0],
            original_h, original_w, scale
        )
        
        return detections
    
    def _process_predictions(self, bbox_pred: np.ndarray, conf_pred: np.ndarray,
                            class_pred: np.ndarray, original_h: int, original_w: int,
                            scale: float) -> List[Dict]:
        """
        Process model predictions into detection boxes.
        
        Args:
            bbox_pred: Bounding box predictions
            conf_pred: Confidence predictions
            class_pred: Class predictions
            original_h: Original image height
            original_w: Original image width
            scale: Scale factor used in preprocessing
            
        Returns:
            List of detections
        """
        # Filter by confidence threshold
        conf_pred = conf_pred.flatten()
        
        if conf_pred < self.confidence_threshold:
            return []
        
        # Get class with highest confidence
        class_id = np.argmax(class_pred)
        class_confidence = class_pred[class_id]
        
        # Normalize bounding box coordinates
        x1, y1, x2, y2 = bbox_pred
        x1 = max(0, int(x1 / scale))
        y1 = max(0, int(y1 / scale))
        x2 = min(original_w, int(x2 / scale))
        y2 = min(original_h, int(y2 / scale))
        
        detection = {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'confidence': float(conf_pred),
            'class_id': str(class_id),
            'class_name': self.class_names[class_id] if class_id < len(self.class_names) else 'unknown'
        }
        
        return [detection]


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Object detection inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--input-image', type=str,
                       help='Path to input image')
    parser.add_argument('--input-video', type=str,
                       help='Path to input video')
    parser.add_argument('--webcam', action='store_true',
                       help='Use webcam for detection')
    parser.add_argument('--output-video', type=str,
                       help='Path for output video')
    parser.add_argument('--duration', type=int, default=30,
                       help='Duration for webcam detection (seconds)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    ensure_dir(config.get('output.results_dir'))
    
    # Create detector
    detector = Detector(args.model, config)
    
    # Run detection based on input type
    if args.input_image:
        print(f"Detecting objects in: {args.input_image}")
        detections = detector.detect_image(args.input_image)
        print(f"\nDetected {len(detections)} objects:")
        for det in detections:
            print(f"  - {det['class_name']}: {det['confidence']:.2%}")
    
    elif args.input_video:
        detector.detect_video(args.input_video, args.output_video)
    
    elif args.webcam:
        detector.detect_webcam(args.duration)
    
    else:
        print("Please specify --input-image, --input-video, or --webcam")


if __name__ == '__main__':
    main()
