import numpy as np


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
    pred = pred.data.cpu().numpy()
    target = target.data.cpu().numpy()  # (16, 224, 224)
    pred = np.argmax(pred, axis=1)  # (16, 224, 224)

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
