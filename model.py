"""Object detection model definition and training."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Optional
import numpy as np


class ObjectDetectionModel:
    """Object detection model using TensorFlow."""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (416, 416, 3),
                 num_classes: int = 80, backbone: str = "mobilenetv2"):
        """
        Initialize object detection model.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of object classes
            backbone: Backbone architecture (mobilenetv2, resnet50, efficientnet)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.backbone = backbone
        self.model = None
    
    def build_model(self) -> Model:
        """
        Build object detection model.
        
        Returns:
            Keras Model
        """
        # Get backbone
        if self.backbone == "mobilenetv2":
            backbone = keras.applications.MobileNetV2(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
        elif self.backbone == "resnet50":
            backbone = keras.applications.ResNet50(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
        elif self.backbone == "efficientnet":
            backbone = keras.applications.EfficientNetB0(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
        else:
            raise ValueError(f"Unknown backbone: {self.backbone}")
        
        # Freeze backbone weights initially
        backbone.trainable = False
        
        # Build detection head
        inputs = keras.Input(shape=self.input_shape)
        x = backbone(inputs, training=False)
        
        # Feature extraction and detection layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Detection outputs
        # Bounding boxes: x1, y1, x2, y2
        bbox_output = layers.Dense(4, name='bounding_boxes')(x)
        
        # Object confidence score
        confidence_output = layers.Dense(1, activation='sigmoid', name='confidence')(x)
        
        # Class predictions
        class_output = layers.Dense(self.num_classes, activation='softmax', name='classes')(x)
        
        outputs = [bbox_output, confidence_output, class_output]
        
        self.model = Model(inputs=inputs, outputs=outputs)
        return self.model
    
    def unfreeze_backbone(self, num_layers: Optional[int] = None) -> None:
        """
        Unfreeze backbone layers for fine-tuning.
        
        Args:
            num_layers: Number of layers to unfreeze from the end (None = unfreeze all)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        backbone = self.model.layers[1]  # Get backbone layer
        
        if num_layers is None:
            backbone.trainable = True
        else:
            for layer in backbone.layers[:-num_layers]:
                layer.trainable = False
            for layer in backbone.layers[-num_layers:]:
                layer.trainable = True
    
    def compile_model(self, learning_rate: float = 0.001) -> None:
        """
        Compile model with losses and optimizer.
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss={
                'bounding_boxes': 'mse',
                'confidence': 'binary_crossentropy',
                'classes': 'categorical_crossentropy'
            },
            loss_weights={
                'bounding_boxes': 1.0,
                'confidence': 0.5,
                'classes': 1.0
            },
            metrics={
                'bounding_boxes': ['mae'],
                'confidence': ['accuracy'],
                'classes': ['accuracy']
            }
        )
    
    def train(self, train_generator, val_generator=None, epochs: int = 50,
              callbacks: list = None) -> dict:
        """
        Train the model.
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Number of epochs
            callbacks: List of Keras callbacks
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss' if val_generator else 'loss',
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if val_generator else 'loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                )
            ]
        
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
    
    def save_model(self, save_path: str) -> None:
        """
        Save model to disk.
        
        Args:
            save_path: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.model.save(save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str) -> None:
        """
        Load model from disk.
        
        Args:
            load_path: Path to load model from
        """
        self.model = keras.models.load_model(load_path)
        print(f"Model loaded from {load_path}")
    
    def summary(self) -> None:
        """Print model summary."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        self.model.summary()


def create_simple_detector():
    """Create a simple end-to-end object detection model."""
    inputs = keras.Input(shape=(416, 416, 3))
    
    # Convolutional blocks
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Flatten and dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Outputs
    bbox = layers.Dense(4, name='bbox')(x)
    confidence = layers.Dense(1, activation='sigmoid', name='confidence')(x)
    classes = layers.Dense(80, activation='softmax', name='classes')(x)
    
    model = Model(inputs=inputs, outputs=[bbox, confidence, classes])
    return model
