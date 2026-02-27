"""
Example usage of the Object Detection System
"""

# Example 1: Training with custom configuration
# ==================================================

from src.config import Config
from src.train import train_model

# Load configuration
config = Config('config.yaml')

# Train model
model, history = train_model(config, use_custom_data=True)


# Example 2: Inference on images
# ==================================================

from src.detect import Detector

# Create detector
detector = Detector('models/final_model.h5', config)

# Detect objects in image
detections = detector.detect_image('test.jpg')

print(f"Found {len(detections)} objects:")
for det in detections:
    print(f"  - {det['class_name']}: {det['confidence']:.2%}")


# Example 3: Custom dataset loading
# ==================================================

from src.dataset import ObjectDetectionDataset

# Load dataset
class_names = ['person', 'car', 'dog', 'cat']
dataset = ObjectDetectionDataset(
    image_dir='data/train/images',
    annotation_dir='data/train/annotations',
    image_size=(416, 416),
    class_names=class_names
)

# Get data generator
train_gen = dataset.get_data_generator(
    batch_size=32,
    shuffle=True
)

# Iterate through batches
for batch_x, batch_y in train_gen:
    # batch_x: image batch
    # batch_y: target tensors
    print(f"Batch shape: {batch_x.shape}")
    break


# Example 4: Model evaluation
# ==================================================

from src.evaluate import ObjectDetectionEvaluator

# Sample predictions and ground truth
predictions = [
    {'x1': 10, 'y1': 20, 'x2': 100, 'y2': 150, 'confidence': 0.95, 'class_id': 0},
    {'x1': 150, 'y1': 200, 'x2': 300, 'y2': 350, 'confidence': 0.85, 'class_id': 1}
]

ground_truth = [
    {'x1': 15, 'y1': 25, 'x2': 105, 'y2': 155, 'class_id': 0},
    {'x1': 160, 'y1': 210, 'x2': 310, 'y2': 360, 'class_id': 1}
]

# Calculate metrics
iou = ObjectDetectionEvaluator.calculate_iou(
    (predictions[0]['x1'], predictions[0]['y1'], predictions[0]['x2'], predictions[0]['y2']),
    (ground_truth[0]['x1'], ground_truth[0]['y1'], ground_truth[0]['x2'], ground_truth[0]['y2'])
)

print(f"IoU: {iou:.4f}")

# Calculate precision-recall
precisions, recalls = ObjectDetectionEvaluator.precision_recall_curve(
    predictions, ground_truth
)

ap = ObjectDetectionEvaluator.average_precision(precisions, recalls)
print(f"Average Precision: {ap:.4f}")


# Example 5: Advanced model configuration
# ==================================================

from src.model import ObjectDetectionModel
import tensorflow as tf

# Create custom model
model = ObjectDetectionModel(
    input_shape=(416, 416, 3),
    num_classes=80,
    backbone='resnet50'
)

# Build and compile model
model.build_model()
model.compile_model(learning_rate=0.001)

# View model architecture
model.summary()

# Unfreeze backbone for fine-tuning
model.unfreeze_backbone(num_layers=20)


# Example 6: Configuration management
# ==================================================

# Access configuration values
learning_rate = config.get('training.learning_rate')
batch_size = config.get('training.batch_size')
model_dir = config.get('output.model_dir')

# Modify configuration
config.set('training.epochs', 50)
config.set('model.backbone', 'efficientnet')

# Save modified configuration
config.save('config_custom.yaml')


# Example 7: Utility functions
# ==================================================

from src.utils import (
    load_image, resize_image, normalize_image,
    nms, calculate_iou, draw_boxes, plot_boxes
)
import numpy as np

# Load and preprocess image
image = load_image('test.jpg')
resized, scale = resize_image(image, (416, 416))
normalized = normalize_image(resized)

# Apply Non-Maximum Suppression
boxes = np.array([[10, 20, 100, 150], [15, 25, 105, 155], [200, 200, 300, 300]])
scores = np.array([0.95, 0.90, 0.5])

keep_indices = nms(boxes, scores, iou_threshold=0.5)

# Draw boxes on image
detections = [
    {'x1': 10, 'y1': 20, 'x2': 100, 'y2': 150, 'confidence': 0.95, 'class_name': 'person'}
]

result = draw_boxes(image, detections)

# Visualize with matplotlib
plot_boxes(result, detections, title="Detections")
