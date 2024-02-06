import numpy as np
import torch
from torchmetrics import JaccardIndex

from . import theme


def find_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        theme.print("WARNING: using CPU!!")
        return torch.device("cpu")


jaccard = JaccardIndex(task="multiclass", num_classes=21)


def compute_iou(pred, target, n_classes=21):
    """
    Calculate the Intersection over Union (IoU) for predictions.

    Args:
    ----
        pred (tensor): Predicted output from the model.
        target (tensor): Ground truth labels.
        n_classes (int, optional): Number of classes. Default is 21.

    Returns:
    -------
        float: Mean IoU across all classes.
    """

    # pred = pred.data.cpu().numpy()
    # target = target.data.cpu().numpy()  # (16, 224, 224)
    # pred = np.argmax(pred, axis=1)  # (16, 224, 224)
    pred = torch.argmax(pred, dim=1).cpu()
    return jaccard(pred, target.cpu())

    iou = 0.0
    for c in range(n_classes):
        true_mask = target == c
        predicted_mask = pred == c
        intersection = np.sum(true_mask & predicted_mask)
        union = np.sum(true_mask | predicted_mask)
        if union == 0:
            continue
        iou += intersection / union
    return iou / n_classes


def compute_pixel_accuracy(pred, target):
    """
    Calculate pixel-wise accuracy between predictions and targets.

    Args:
    ----
        pred (tensor): Predicted output from the model.
        target (tensor): Ground truth labels.

    Returns:
    -------
        float: Pixel-wise accuracy.
    """
    pred = pred.data.cpu().numpy()
    target = target.data.cpu().numpy()
    pred = np.argmax(pred, axis=1)
    assert pred.shape == target.shape
    return np.sum(pred == target) / pred.size
