# Just run once using `python download.py`

import torchvision
from torch.utils.data import DataLoader

from . import voc
from .paths import DATA_DIR

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


def load_dataset(input_transform, target_transform):
    train_dataset = voc.VOC(
        "train", transform=input_transform, target_transform=target_transform
    )
    val_dataset = voc.VOC(
        "val", transform=input_transform, target_transform=target_transform
    )
    test_dataset = voc.VOC(
        "test", transform=input_transform, target_transform=target_transform
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

    return train_loader, val_loader, test_loader
