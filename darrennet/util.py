import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import warnings
import torchvision.transforms.functional as F
from torchmetrics import JaccardIndex
from torchvision.transforms import v2
from .main import load_dataset

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

def makeImagesPlotReady(image):
    batch, depth, height, width = image.shape
    reshape_img = image[0]

    # Get the rgb channels and reshape the image to height x width x depth
    r, g, b = reshape_img
    reshape_img = np.array([[[r[i][j], g[i][j], b[i][j]] for j in range(width)] for i in range(height)])
    return reshape_img

def convertMaskToRGB(msk, palette):
    b, d, h, w = msk.shape
    mask_rgb = np.zeros((1, 3, h, w), dtype=np.uint8)

    # Loop through every pixel of the mask and set the corresponding RGB values
    for i in range(h):
        for j in range(w):
            mask_rgb[0][0][i][j], mask_rgb[0][1][i][j], mask_rgb[0][2][i][j] = palette[int(msk[0][0][i][j])]
    return mask_rgb

def returnToString(arr, noToClass):
    string = ""
    for i in arr:
        string += noToClass[i]
        string += ", "
    return string[:-2]

def export_model(fcn_model, device, inputs):
    fcn_model.eval()
    fcn_model.to(device)
    inputs = inputs.to(device)
    with torch.no_grad():
        output_image = fcn_model(inputs)
    fcn_model.train()
    return output_image

def compare_images(model_paths, img_idx=0, img_cnt=1):
    num_to_class = {0:"background", 1:"aeroplane", 2:"bicycle", 3:"bird", 4:"boat", 5:"bottle",
                 6:"bus", 7:"car", 8:"cat", 9:"chair", 10:"cow", 11:"dining table", 12:"dog",
                 13:"horse", 14:"motorbike", 15:"person", 16:"potted plant", 17:"sheep",
                 18:"sofa", 19:"train", 20:"tv/monitor"}

    rgb = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128)
           ,(128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128)
           ,(64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]

    rgb = {i : rgb[i] for i in range(len(num_to_class))}

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(*mean_std),
        ]
    )
    mask_transform = v2.Compose(
        [
            v2.Lambda(lambda x: x.resize((224, 224))),
            v2.ToImage(),
        ]
    )

    train, val, test = load_dataset(None, input_transform, mask_transform, size=1)
    if img_idx < 0 or img_idx > len(test) - 1: img_idx = 0

    first_batch = iter(test)
    for _ in range(img_idx): next(first_batch)

    # Create models here
    models = [torch.load(os.path.join("models/" + m)) for m in model_paths]
    model_cnt = len(models) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_paths.insert(0, "Ground Truth")
    for _ in range(img_cnt):
        img, msk = next(first_batch)
        true_classes = [num_to_class[i] for i in torch.unique(msk).numpy()]
        print("Ground True Classes:", true_classes,"\n")

        output_images = [convertMaskToRGB(msk, rgb)]
        for i in range(model_cnt - 1):
            output_image = export_model(models[i], device, img)

            output_image = output_image.to("cpu").detach()
            output_image = torch.argmax(output_image, dim=1)
            print(f"Model \"{model_paths[i + 1]}\" Predicted Classes:", returnToString(torch.unique(output_image).numpy(), num_to_class))

            output_image = output_image[None,:,:,:]
            mask_rgb = convertMaskToRGB(output_image, rgb)
            output_images.append(mask_rgb)

        fig, axs = plt.subplots(model_cnt // 4 + 1, 4, figsize=(12, 6))
        axs = axs.flatten()
        for c in range(model_cnt):
            axs[c].imshow(makeImagesPlotReady(img))
            axs[c].imshow(makeImagesPlotReady(output_images[c]), alpha = 0.6)
            axs[c].set_title(model_paths[c])
            axs[c].axis('off')
        plt.show()