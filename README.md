# Object Detection System

A comprehensive object detection system built with TensorFlow/Keras supporting custom model training and inference on images, videos, and real-time webcam feeds.

## Features

- **Multiple Backbones**: MobileNetV2, ResNet50, EfficientNet
- **Custom Training**: Train on your own dataset
- **Flexible Inference**: Images, videos, and real-time webcam detection
- **Easy Configuration**: YAML-based configuration management
- **Production Ready**: Modular design with proper error handling

## Project Structure

```
The-void/
├── main.py              # Main entry point
├── requirements.txt     # Python dependencies
├── config.yaml         # Configuration file
├── README.md           # This file
├── src/                # Source code
│   ├── config.py       # Configuration management
│   ├── dataset.py      # Dataset loading and preprocessing
│   ├── model.py        # Model definition
│   ├── train.py        # Training script
│   ├── detect.py       # Inference module
│   ├── evaluate.py     # Evaluation metrics
│   └── utils.py        # Utility functions
├── data/               # Training data
│   ├── train/         # Training images and annotations
│   ├── val/           # Validation images and annotations
│   └── test/          # Test images
├── models/            # Trained models
└── outputs/           # Detection results and logs
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to customize training and inference parameters:

```yaml
# Model settings
model:
  backbone: "mobilenetv2"  # Options: mobilenetv2, resnet50, efficientnet
  input_shape: [416, 416, 3]
  num_classes: 80

# Training settings
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001

# Inference settings
inference:
  confidence_threshold: 0.5
  iou_threshold: 0.4
```

## Quick Start

### Training

Train with dummy data (for testing):
```bash
python main.py train --dummy-data
```

Train with custom data:
```bash
python main.py train --config config.yaml
```

### Inference

Detect objects in an image:
```bash
python main.py detect --model models/final_model.h5 --input-image test.jpg
```

Detect objects in a video:
```bash
python main.py detect --model models/final_model.h5 --input-video test.mp4 --output-video output.mp4
```

Real-time detection from webcam:
```bash
python main.py detect --model models/final_model.h5 --webcam --duration 30
```

## Dataset Format

The system supports Pascal VOC XML format for annotations:

```xml
<?xml version="1.0"?>
<annotation>
  <object>
    <name>person</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>150</ymin>
      <xmax>300</xmax>
      <ymax>450</ymax>
    </bndbox>
  </object>
</annotation>
```

## Model Architecture

- **Backbone**: Transfer learning from pre-trained models (MobileNetV2, ResNet50, EfficientNet)
- **Detection Head**: Custom layers for bounding box regression and classification
- **Outputs**:
  - Bounding boxes: `[x1, y1, x2, y2]`
  - Confidence score: Object presence probability  
  - Class predictions: 80-class softmax output

## Evaluation Metrics

- **IoU (Intersection over Union)**: Box overlap measurement
- **Precision-Recall Curve**: True positive and false positive trade-off
- **Average Precision (AP)**: Area under precision-recall curve
- **mAP**: Mean average precision across all classes

## Troubleshooting

### Out of Memory Error
- Reduce `batch_size` in `config.yaml`
- Use smaller backbone (MobileNetV2)
- Reduce input image size

### Low Accuracy
- Increase training data
- Train for more epochs
- Adjust detection thresholds
- Enable data augmentation

## Advanced Usage

### Custom Training

```python
from src.train import train_model
from src.config import Config

config = Config('config.yaml')
model, history = train_model(config, use_custom_data=True)
```

### Custom Dataset Loading

```python
from src.dataset import ObjectDetectionDataset

dataset = ObjectDetectionDataset(
    image_dir='data/train/images',
    annotation_dir='data/train/annotations',
    image_size=(416, 416),
    class_names=['person', 'car', 'dog']
)

train_gen = dataset.get_data_generator(batch_size=32, shuffle=True)
```

## Dependencies

- TensorFlow 2.15+
- NumPy
- OpenCV
- Pillow
- Matplotlib
- scikit-learn