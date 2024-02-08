import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
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
    target = target.squeeze().data.cpu()  # (16, 224, 224)
    # pred = np.argmax(pred, axis=1)  # (16, 224, 224)
    pred = torch.argmax(pred, dim=1).cpu()
    return jaccard(pred, target)

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
    target = target.data.squeeze().cpu().numpy()
    pred = np.argmax(pred, axis=1)
    assert pred.shape == target.shape, (pred.shape, target.shape)
    return np.sum(pred == target) / pred.size


def display_images(image_tensors, mask_tensors):
    if image_tensors is None and mask_tensors is None:
        raise Exception("Must pass something in")
    if mask_tensors is None:
        num_images = len(image_tensors)
    else:
        num_images = len(mask_tensors)

    # Set up the subplots in a 4x4 grid (adjust as needed)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(num_images):
        if image_tensors is not None:
            # Convert torch tensors to numpy arrays
            img_np = F.to_pil_image(image_tensors[i])
            # Overlay the segmentation mask
            axes[i].imshow(np.asarray(img_np))
        if mask_tensors is not None:
            mask_np = F.to_pil_image(mask_tensors[i])
            axes[i].imshow(
                np.asarray(mask_np), cmap="tab20", alpha=0.5, vmin=0, vmax=21
            )

        axes[i].axis("off")  # Optional: Turn off axis labels
        axes[i].set_title(f"Image {i + 1}")

    plt.tight_layout()
    plt.show()
