import numpy as np
import torch

def iou(pred, target, n_classes = 21):
    """
    Calculate the Intersection over Union (IoU) for predictions.

    Args:
        pred (tensor): Predicted output from the model.
        target (tensor): Ground truth labels.
        n_classes (int, optional): Number of classes. Default is 21.

    Returns:
        float: Mean IoU across all classes.
    """

    mask = np.arange(n_classes)[:, None, None]
    intersection = np.sum((pred == mask) & (target == mask), axis=(1, 2))
    union = np.sum((pred == mask) | (target == mask), axis=(1, 2))
    iou_per_class = np.where(union > 0, intersection / union, 0)
    return np.mean(iou_per_class)

def pixel_acc(pred, target):
    """
    Calculate pixel-wise accuracy between predictions and targets.

    Args:
        pred (tensor): Predicted output from the model.
        target (tensor): Ground truth labels.

    Returns:
        float: Pixel-wise accuracy.
    """
    return np.sum(pred == target) / pred.size
