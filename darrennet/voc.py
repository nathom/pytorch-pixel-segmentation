import torch
from PIL import Image
from torch.utils import data
from torchvision.transforms import v2

from .paths import DATA_DIR

num_classes = 21
ignore_label = 255
root = DATA_DIR

"""
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
"""


# Feel free to convert this palette to a map
palette = [
    0,
    0,
    0,
    128,
    0,
    0,
    0,
    128,
    0,
    128,
    128,
    0,
    0,
    0,
    128,
    128,
    0,
    128,
    0,
    128,
    128,
    128,
    128,
    128,
    64,
    0,
    0,
    192,
    0,
    0,
    64,
    128,
    0,
    192,
    128,
    0,
    64,
    0,
    128,
    192,
    0,
    128,
    64,
    128,
    128,
    192,
    128,
    128,
    0,
    64,
    0,
    128,
    64,
    0,
    0,
    192,
    0,
    128,
    192,
    0,
    0,
    64,
    128,
]  # 3 values- R,G,B for every class. First 3 values for class 0, next 3 for
# class 1 and so on......


def make_dataset(mode):
    """
    Creates a list of tuples (image_path, mask_path) for a given dataset mode.

    Asserts that the mode is one of 'train', 'val', or 'test'.
    Based on the mode, it reads corresponding image and mask paths from the VOC dataset.
    For each image, it pairs its path with the corresponding mask path.

    TODO: Make similar for val and test set.

    Args:
    ----
        mode (str): The mode of the dataset, either 'train', 'val', or 'test'.

    Returns:
    -------
        list of tuples: Each tuple contains paths (image_path, mask_path).
    """
    assert mode in ["train", "val", "test"]
    items = []
    voc_2007 = root / "VOCdevkit" / "VOC2007"
    img_dir = voc_2007 / "JPEGImages"
    mask_path = voc_2007 / "SegmentationClass"
    if mode == "train":
        img_names = voc_2007 / "ImageSets" / "Segmentation" / "train.txt"
    elif mode == "val":
        img_names = voc_2007 / "ImageSets" / "Segmentation" / "val.txt"
    else:
        img_names = voc_2007 / "ImageSets" / "Segmentation" / "test.txt"

    with open(img_names) as file:
        items = [
            (img_dir / f"{img}.jpg", mask_path / f"{img}.png")
            for img in map(str.strip, file.readlines())
        ]

    return items


class VOC(data.Dataset):
    """
    A custom dataset class for VOC dataset.
    Maintain the structure of this so that it is easily compatible with Pytorch's dataloader.

    - Resizes images and masks to a specified width and height.
    - Implements methods to get dataset items and dataset length.

    - TIP: You may add an additional argument for common transformation for both the image and mask
           to help with data augmentation in part 4.c

    Args:
    ----
        mode (str): Mode of the dataset ('train', 'val', etc.).
        transform (callable, optional): Transform to be applied to the images.
        target_transform (callable, optional): Transform to be applied to the masks.
    """

    def __init__(
        self,
        mode: str,
        transform: v2.Transform | None = None,
        target_transform: v2.Transform | None = None,
        augmentation: v2.Transform | None = None,
    ):
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise RuntimeError("Found 0 images, please check the data set")
        self.mode = mode
        self.augmentation = augmentation
        self.transform = transform
        self.target_transform = target_transform
        self.width = 224
        self.height = 224

    def __getitem__(self, index: int):
        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        mask = Image.open(mask_path).resize((224, 224))

        if self.mode == "train" and self.augmentation is not None:
            img, mask = self.augmentation(img, mask)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            mask: torch.Tensor = self.target_transform(mask)

        mask[mask == ignore_label] = 0

        return img, mask

    def __len__(self):
        return len(self.imgs)
