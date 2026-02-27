"""Utility functions for object detection system."""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array (BGR format from OpenCV)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return img


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> Tuple[np.ndarray, float]:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        target_size: Target size (width, height)
        
    Returns:
        Resized image and scale factor
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scale to fit image in target size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to target size
    padded = np.zeros((target_h, target_w, 3), dtype=resized.dtype)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded, scale


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    return image.astype(np.float32) / 255.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Convert normalized image back to [0, 255] range.
    
    Args:
        image: Normalized image
        
    Returns:
        Denormalized image
    """
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """
    Apply Non-Maximum Suppression to remove overlapping boxes.
    
    Args:
        boxes: Array of bounding boxes [x1, y1, x2, y2]
        scores: Confidence scores for each box
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Indices of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([])
    
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(sorted_indices) > 0:
        current = sorted_indices[0]
        keep.append(current)
        
        if len(sorted_indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes[current]
        remaining_boxes = boxes[sorted_indices[1:]]
        
        iou = calculate_iou(current_box, remaining_boxes)
        
        # Keep only boxes with IoU below threshold
        sorted_indices = sorted_indices[1:][iou < iou_threshold]
    
    return np.array(keep)


def calculate_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Calculate Intersection over Union (IoU).
    
    Args:
        box: Single box [x1, y1, x2, y2]
        boxes: Multiple boxes [N, 4]
        
    Returns:
        IoU values for each box
    """
    # Calculate intersection
    x1_min = np.maximum(box[0], boxes[:, 0])
    y1_min = np.maximum(box[1], boxes[:, 1])
    x2_max = np.minimum(box[2], boxes[:, 2])
    y2_max = np.minimum(box[3], boxes[:, 3])
    
    intersection = np.maximum(0, x2_max - x1_min) * np.maximum(0, y2_max - y1_min)
    
    # Calculate union
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    
    iou = intersection / np.maximum(union, 1e-8)
    return iou


def draw_boxes(image: np.ndarray, boxes: List[dict], color: Tuple[int, int, int] = (0, 255, 0),
               thickness: int = 2) -> np.ndarray:
    """
    Draw bounding boxes on image.
    
    Args:
        image: Input image
        boxes: List of box dictionaries with keys: x1, y1, x2, y2, confidence, class_name
        color: Box color (BGR)
        thickness: Line thickness
        
    Returns:
        Image with drawn boxes
    """
    result = image.copy()
    
    for box in boxes:
        x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
        confidence = box.get('confidence', 0)
        class_name = box.get('class_name', 'Object')
        
        # Draw rectangle
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        label = f"{class_name} {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(result, (x1, y1 - label_size[1] - 5), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(result, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return result


def plot_boxes(image: np.ndarray, boxes: List[dict], title: str = "Detection Results") -> None:
    """
    Plot image with bounding boxes using matplotlib.
    
    Args:
        image: Input image (BGR format from OpenCV)
        boxes: List of box dictionaries
        title: Plot title
    """
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.set_title(title)
    
    for box in boxes:
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        width = x2 - x1
        height = y2 - y1
        
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2,
                                edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        confidence = box.get('confidence', 0)
        class_name = box.get('class_name', 'Object')
        ax.text(x1, y1 - 10, f"{class_name} {confidence:.2f}",
               color='red', fontsize=10, bbox=dict(facecolor='yellow', alpha=0.7))
    
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
