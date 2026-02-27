"""Dataset loading and preprocessing for object detection."""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import tensorflow as tf
from tqdm import tqdm


class ObjectDetectionDataset:
    """Dataset loader for object detection tasks."""
    
    def __init__(self, image_dir: str, annotation_dir: str, image_size: Tuple[int, int] = (416, 416),
                 class_names: Optional[List[str]] = None):
        """
        Initialize dataset loader.
        
        Args:
            image_dir: Directory containing images
            annotation_dir: Directory containing annotations
            image_size: Target image size (width, height)
            class_names: List of class names
        """
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.image_size = image_size
        self.class_names = class_names or []
        
        self.images = sorted(self.image_dir.glob("*.jpg")) + sorted(self.image_dir.glob("*.png"))
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get image and annotations.
        
        Args:
            idx: Index of image
            
        Returns:
            Image array and annotation array
        """
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotations
        annotation_path = self.annotation_dir / img_path.stem / ".xml"
        boxes = self._parse_xml_annotation(annotation_path)
        
        # Resize image
        original_h, original_w = image.shape[:2]
        image = cv2.resize(image, self.image_size)
        
        # Normalize image
        image = image.astype(np.float32) / 255.0
        
        # Normalize boxes to resized image
        scale_x = self.image_size[0] / original_w
        scale_y = self.image_size[1] / original_h
        
        normalized_boxes = []
        for box in boxes:
            x1 = int(box['x1'] * scale_x)
            y1 = int(box['y1'] * scale_y)
            x2 = int(box['x2'] * scale_x)
            y2 = int(box['y2'] * scale_y)
            class_id = box['class_id']
            
            normalized_boxes.append([x1, y1, x2, y2, class_id])
        
        # Create target tensor
        target = self._create_target_tensor(normalized_boxes)
        
        return image, target
    
    def _parse_xml_annotation(self, xml_path: Path) -> List[Dict]:
        """
        Parse XML annotation file (Pascal VOC format).
        
        Args:
            xml_path: Path to XML file
            
        Returns:
            List of box dictionaries with keys: x1, y1, x2, y2, class_id
        """
        boxes = []
        
        if not xml_path.exists():
            return boxes
        
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for obj in root.findall('object'):
            class_name_elem = obj.find('name')
            if class_name_elem is None or class_name_elem.text is None:
                continue
            class_name = class_name_elem.text
            class_id = self.class_names.index(class_name) if class_name in self.class_names else 0
            
            bbox = obj.find('bndbox')
            if bbox is None:
                continue
            
            xmin_elem = bbox.find('xmin')
            ymin_elem = bbox.find('ymin')
            xmax_elem = bbox.find('xmax')
            ymax_elem = bbox.find('ymax')
            
            if any(elem is None or elem.text is None for elem in [xmin_elem, ymin_elem, xmax_elem, ymax_elem]):
                continue
            
            box = {
                'x1': int(xmin_elem.text),
                'y1': int(ymin_elem.text),
                'x2': int(xmax_elem.text),
                'y2': int(ymax_elem.text),
                'class_id': class_id
            }
            boxes.append(box)
        
        return boxes
    
    def _create_target_tensor(self, boxes: List[list], max_boxes: int = 100) -> np.ndarray:
        """
        Create target tensor for training.
        
        Args:
            boxes: List of boxes [x1, y1, x2, y2, class_id]
            max_boxes: Maximum number of boxes per image
            
        Returns:
            Target tensor of shape (max_boxes, 5)
        """
        target = np.zeros((max_boxes, 5), dtype=np.float32)
        
        for i, box in enumerate(min(len(boxes), max_boxes)):
            target[i] = box
        
        return target
    
    def get_data_generator(self, batch_size: int = 32, shuffle: bool = True):
        """
        Create TensorFlow data generator.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            tf.data.Dataset
        """
        indices = np.arange(len(self))
        
        if shuffle:
            np.random.shuffle(indices)
        
        def generator():
            for idx in indices:
                image, target = self[idx]
                yield image, target
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(self.image_size[1], self.image_size[0], 3), dtype=tf.float32),
                tf.TensorSpec(shape=(100, 5), dtype=tf.float32)
            )
        )
        
        dataset = dataset.batch(batch_size)
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=100)
        
        return dataset


class DatasetBuilder:
    """Builder for creating datasets from raw data."""
    
    @staticmethod
    def create_dummy_dataset(output_dir: str, num_images: int = 10,
                           image_size: Tuple[int, int] = (640, 480)) -> None:
        """
        Create dummy dataset for testing.
        
        Args:
            output_dir: Output directory for dataset
            num_images: Number of images to create
            image_size: Image size (width, height)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        img_dir = output_path / "images"
        ann_dir = output_path / "annotations"
        img_dir.mkdir(exist_ok=True)
        ann_dir.mkdir(exist_ok=True)
        
        for i in range(num_images):
            # Create random image
            image = np.random.randint(0, 256, (*image_size[::-1], 3), dtype=np.uint8)
            
            # Draw some random objects
            for _ in range(np.random.randint(1, 4)):
                x1 = np.random.randint(0, image_size[0] - 50)
                y1 = np.random.randint(0, image_size[1] - 50)
                x2 = x1 + np.random.randint(50, 200)
                y2 = y1 + np.random.randint(50, 200)
                color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Save image
            cv2.imwrite(str(img_dir / f"image_{i:04d}.jpg"), image)
            
            # Save dummy annotation
            ann_file = ann_dir / f"image_{i:04d}.txt"
            ann_file.write_text("")
        
        print(f"Created dummy dataset with {num_images} images in {output_path}")
