import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from . import voc
from .paths import DATA_DIR
from .theme import console

root = str(DATA_DIR)


def download_data():
    torchvision.datasets.VOCSegmentation(
        root=root, year="2007", download=True, image_set="train"
    )
    torchvision.datasets.VOCSegmentation(
        root=root, year="2007", download=True, image_set="val"
    )
    torchvision.datasets.VOCSegmentation(
        root=root, year="2007", download=True, image_set="test"
    )


def load_dataset(augment_transform, input_transform, target_transform):
    train_dataset = voc.VOC(
        "train", input_transform, target_transform, augmentation=augment_transform
    )
    val_dataset = voc.VOC("val", input_transform, target_transform, augmentation=None)
    test_dataset = voc.VOC("test", input_transform, target_transform, augmentation=None)

    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

    return train_loader, val_loader, test_loader


def get_frequency_spectrum():
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # Final transform for input
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
            # v2.Lambda(
            #     lambda img: torch.from_numpy(np.array(img, dtype=np.int32)).long()
            # ),
            v2.ToImage(),
            # Change shape from (16,1,224,224) -> (16,224,224)
            # v2.Lambda(lambda x: torch.squeeze(x, dim=0)),
        ]
    )
    train, val, test = load_dataset(None, input_transform, mask_transform)
    all_train = np.concatenate([label for _, label in train]).flatten()
    unique_values, counts = np.unique(all_train, return_counts=True)

    freq = dict(zip(unique_values, counts))
    counts = np.array([freq[i] for i in range(21)])
    console.print("Frequency of each class")
    console.print(freq)
    total = counts.sum()

    inv_freq = total / counts
    console.print("Incorrect guess penalty (the inverse of relative frequency):")
    console.print(list(inv_freq))
    console.print(list(inv_freq / inv_freq.max()))

    weights = 1.0 / np.log(1.02 + counts / total)

    console.print("Log calculated weights:")
    console.print(list(weights))
