"""Training script for object detection model."""

import argparse
import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from config import Config
from model import ObjectDetectionModel
from dataset import ObjectDetectionDataset, DatasetBuilder
from utils import ensure_dir


def setup_training(config: Config):
    """Set up training environment."""
    ensure_dir(config.get('output.model_dir'))
    ensure_dir(config.get('output.log_dir'))
    ensure_dir(config.get('output.results_dir'))
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)


def train_model(config: Config, use_custom_data: bool = True):
    """
    Train object detection model.
    
    Args:
        config: Configuration object
        use_custom_data: Whether to use custom dataset or create dummy data
    """
    print("=" * 60)
    print("Object Detection Model Training")
    print("=" * 60)
    
    setup_training(config)
    
    # Create or load dataset
    if not use_custom_data:
        print("\nCreating dummy dataset for demonstration...")
        DatasetBuilder.create_dummy_dataset(config.get('data.train_path'), num_images=20)
        DatasetBuilder.create_dummy_dataset(config.get('data.val_path'), num_images=5)
    else:
        # Check if custom data exists
        if not Path(config.get('data.train_path')).exists():
            print(f"\nWarning: Custom dataset not found at {config.get('data.train_path')}")
            print("Creating dummy dataset instead...")
            DatasetBuilder.create_dummy_dataset(config.get('data.train_path'), num_images=20)
    
    print("\nLoading dataset...")
    
    # COCO class names (simplified)
    class_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant'
    ] + ['class_{}'.format(i) for i in range(20, 80)]
    
    # Create datasets
    input_shape = tuple(config.get('model.input_shape'))
    train_dataset = ObjectDetectionDataset(
        config.get('data.train_path') + "/images",
        config.get('data.train_path') + "/annotations",
        image_size=(input_shape[1], input_shape[0]),
        class_names=class_names
    )
    
    # Build model
    print("\nBuilding model...")
    model = ObjectDetectionModel(
        input_shape=input_shape,
        num_classes=config.get('model.num_classes'),
        backbone=config.get('model.backbone')
    )
    model.build_model()
    model.compile_model(learning_rate=config.get('training.learning_rate'))
    model.summary()
    
    # Create data generator
    print("\nCreating data generator...")
    train_gen = train_dataset.get_data_generator(
        batch_size=config.get('training.batch_size'),
        shuffle=True
    )
    
    # Training callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=config.get('training.early_stopping_patience'),
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=Path(config.get('output.model_dir')) / 'best_model.h5',
            monitor='loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=config.get('output.log_dir'),
            histogram_freq=1
        )
    ]
    
    # Train model
    print("\nStarting training...")
    print(f"Epochs: {config.get('training.epochs')}")
    print(f"Batch size: {config.get('training.batch_size')}")
    print(f"Model: {config.get('model.backbone')}")
    
    history = model.train(
        train_gen,
        epochs=config.get('training.epochs'),
        callbacks=callbacks
    )
    
    # Save final model
    final_model_path = Path(config.get('output.model_dir')) / 'final_model.h5'
    model.save_model(str(final_model_path))
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Model saved to: {final_model_path}")
    print("=" * 60)
    
    return model, history


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train object detection model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--dummy-data', action='store_true',
                       help='Use dummy data for testing')
    
    args = parser.parse_args()
    
    config = Config(args.config)
    model, history = train_model(config, use_custom_data=not args.dummy_data)


if __name__ == '__main__':
    main()
