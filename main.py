#!/usr/bin/env python3
"""Main entry point for object detection system."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from train import train_model
from detect import Detector
from config import Config
from utils import ensure_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Object Detection System - Train and inference command line tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model with dummy data
  python main.py train --dummy-data
  
  # Train model with custom data
  python main.py train --config config.yaml
  
  # Detect objects in image
  python main.py detect --model models/final_model.h5 --input-image test.jpg
  
  # Detect objects in video
  python main.py detect --model models/final_model.h5 --input-video test.mp4 --output-video output.mp4
  
  # Real-time detection from webcam
  python main.py detect --model models/final_model.h5 --webcam --duration 60
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train object detection model')
    train_parser.add_argument('--config', type=str, default='config.yaml',
                             help='Path to config file')
    train_parser.add_argument('--dummy-data', action='store_true',
                             help='Use dummy data for testing')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Run object detection inference')
    detect_parser.add_argument('--model', type=str, required=True,
                              help='Path to trained model')
    detect_parser.add_argument('--config', type=str, default='config.yaml',
                              help='Path to config file')
    detect_parser.add_argument('--input-image', type=str,
                              help='Path to input image')
    detect_parser.add_argument('--input-video', type=str,
                              help='Path to input video')
    detect_parser.add_argument('--webcam', action='store_true',
                              help='Use webcam for detection')
    detect_parser.add_argument('--output-video', type=str,
                              help='Path for output video (video input only)')
    detect_parser.add_argument('--duration', type=int, default=30,
                              help='Duration for webcam detection (seconds)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'train':
        config = Config(args.config)
        train_model(config, use_custom_data=not args.dummy_data)
    
    elif args.command == 'detect':
        config = Config(args.config)
        ensure_dir(config.get('output.results_dir'))
        
        detector = Detector(args.model, config)
        
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
            print("Error: Please specify --input-image, --input-video, or --webcam")
            sys.exit(1)


if __name__ == '__main__':
    main()
