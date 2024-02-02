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

    intersection = np.sum(pred == target, axis=(1, 2))  # (16,)
    union = np.sum(pred != target, axis=(1, 2)) + intersection  # (16,)
    return np.mean(intersection / union)

    # mask = np.arange(n_classes)[:, None, None]
    #
    # intersection = np.sum((pred == mask) & (target == mask), axis=(1, 2))
    # union = np.sum((pred == mask) | (target == mask), axis=(1, 2))
    # iou_per_class = np.where(union > 0, intersection / union, 0)
    # return np.mean(iou_per_class)


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
