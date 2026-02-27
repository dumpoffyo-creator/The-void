"""Model evaluation and metrics calculation."""

import numpy as np
from typing import List, Dict, Tuple
import tensorflow as tf


class ObjectDetectionEvaluator:
    """Evaluator for object detection metrics."""
    
    @staticmethod
    def calculate_iou(box1: Tuple, box2: Tuple) -> float:
        """
        Calculate Intersection over Union (IoU) between two boxes.
        
        Args:
            box1: Box as (x1, y1, x2, y2)
            box2: Box as (x1, y1, x2, y2)
            
        Returns:
            IoU value between 0 and 1
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        intersect_xmin = max(x1_min, x2_min)
        intersect_ymin = max(y1_min, y2_min)
        intersect_xmax = min(x1_max, x2_max)
        intersect_ymax = min(y1_max, y2_max)
        
        if intersect_xmax < intersect_xmin or intersect_ymax < intersect_ymin:
            return 0.0
        
        intersect_area = (intersect_xmax - intersect_xmin) * (intersect_ymax - intersect_ymin)
        
        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - intersect_area
        
        iou = intersect_area / union_area if union_area > 0 else 0.0
        return iou
    
    @staticmethod
    def precision_recall_curve(predictions: List[Dict], ground_truth: List[Dict],
                              iou_threshold: float = 0.5) -> Tuple[List[float], List[float]]:
        """
        Calculate precision-recall curve.
        
        Args:
            predictions: List of predicted detections
            ground_truth: List of ground truth detections
            iou_threshold: IoU threshold for matching
            
        Returns:
            Precision and recall values
        """
        # Sort predictions by confidence
        sorted_preds = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        tp = 0  # True positives
        fp = 0  # False positives
        used_gt = set()
        
        precisions = []
        recalls = []
        
        for pred in sorted_preds:
            pred_box = (pred['x1'], pred['y1'], pred['x2'], pred['y2'])
            
            # Find best matching ground truth
            best_iou = iou_threshold
            best_idx = -1
            
            for i, gt in enumerate(ground_truth):
                if i in used_gt:
                    continue
                
                if pred.get('class_id') != gt.get('class_id'):
                    continue
                
                gt_box = (gt['x1'], gt['y1'], gt['x2'], gt['y2'])
                iou = ObjectDetectionEvaluator.calculate_iou(pred_box, gt_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            # Update TP/FP
            if best_idx >= 0:
                tp += 1
                used_gt.add(best_idx)
            else:
                fp += 1
            
            # Calculate precision and recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / len(ground_truth) if len(ground_truth) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        return precisions, recalls
    
    @staticmethod
    def average_precision(precisions: List[float], recalls: List[float]) -> float:
        """
        Calculate Average Precision (AP) from precision-recall curve.
        
        Args:
            precisions: List of precision values
            recalls: List of recall values
            
        Returns:
            Average precision value
        """
        if len(precisions) == 0:
            return 0.0
        
        # Sort by recall
        sorted_indices = np.argsort(recalls)
        recalls = np.array(recalls)[sorted_indices]
        precisions = np.array(precisions)[sorted_indices]
        
        # Interpolate precision
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))
        
        # Calculate area under curve
        ap = np.sum(np.diff(recalls) * precisions[:-1])
        return ap
    
    @staticmethod
    def mean_average_precision(predictions_per_image: List[List[Dict]],
                              ground_truth_per_image: List[List[Dict]],
                              iou_threshold: float = 0.5) -> float:
        """
        Calculate Mean Average Precision (mAP).
        
        Args:
            predictions_per_image: List of predictions for each image
            ground_truth_per_image: List of ground truth for each image
            iou_threshold: IoU threshold for matching
            
        Returns:
            mAP value
        """
        aps = []
        
        for preds, gts in zip(predictions_per_image, ground_truth_per_image):
            precisions, recalls = ObjectDetectionEvaluator.precision_recall_curve(
                preds, gts, iou_threshold
            )
            
            if len(precisions) > 0:
                ap = ObjectDetectionEvaluator.average_precision(precisions, recalls)
                aps.append(ap)
        
        mAP = np.mean(aps) if len(aps) > 0 else 0.0
        return mAP


class MetricsCallback(tf.keras.callbacks.Callback):
    """Keras callback for custom metrics."""
    
    def __init__(self, val_data=None):
        """
        Initialize callback.
        
        Args:
            val_data: Validation data for evaluation
        """
        super().__init__()
        self.val_data = val_data
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at end of each epoch."""
        if logs is None:
            logs = {}
        
        # Add custom metrics to logs here
        # Example: logs['custom_metric'] = custom_value
