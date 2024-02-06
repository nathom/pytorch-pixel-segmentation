import click
import numpy as np
import torch
from click_help_colors import HelpColorsCommand, HelpColorsGroup
from rich.traceback import install
from torch import nn
from torchvision.transforms import v2

from . import theme
from .basic_fcn import FCN
from .dataset import download_data, load_dataset
from .train import model_test, model_train

install(show_locals=True)


epochs = 20
n_class = 21


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def init_weights(model):
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(model.weight.data)
        assert model.bias is not None
        torch.nn.init.normal_(model.bias.data)  # xavier not applicable for biases


def find_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        theme.print("WARNING: using CPU!!")
        return torch.device("cpu")


@click.group(
    cls=HelpColorsGroup,
    help_headers_color="blue",
    help_options_color="bright_red",
    help_options_custom_colors={"--help": "bright_green"},
)
@click.option("-s", "--serious", help="Turn on Serious Darren mode.", is_flag=True)
def main(serious):
    """Welcome to DarrenNet, the world's most advanced CNN for pixel segmentation."""
    theme.set_serious(serious)


@main.command(cls=HelpColorsCommand)
def download():
    """Download and save the dataset."""
    theme.print("Downloading dataset...")
    download_data()


@click.option("-a", "--augment", help="Choose <=4 from {a,v,h,r}")
@main.command(cls=HelpColorsCommand)
def cook(augment):
    """Train the model."""
    device = find_device()
    fcn_model = FCN(n_class=n_class)
    fcn_model.apply(init_weights)

    augment_transforms: list[v2.Transform] = [v2.ToImage()]
    if augment is not None:
        if "a" in augment:
            theme.print("Affine transform selected.")
            augment_transforms.append(v2.RandomAffine((-20, 20), translate=(0, 0.2)))
        if "v" in augment:
            theme.print("Vertical flip transform selected.")
            augment_transforms.append(v2.RandomVerticalFlip(0.5))
        if "h" in augment:
            theme.print("Horizontal flip transform selected.")
            augment_transforms.append(v2.RandomHorizontalFlip(0.5))
        if "r" in augment:
            theme.print("Resized Crop transform selected.")
            augment_transforms.append(v2.RandomResizedCrop((224, 224)))

    augment_transform = v2.Compose(augment_transforms)

    optimizer = torch.optim.Adam(params=fcn_model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    fcn_model = fcn_model.to(device)
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
            v2.ToImage(),
            v2.Resize((224, 224)),
            # Change shape from (16,1,224,224) -> (16,224,224)
            v2.Lambda(lambda x: torch.squeeze(x, dim=0)),
        ]
    )

    train_loader, val_loader, test_loader = load_dataset(
        augment_transform,
        input_transform,
        mask_transform,
    )

    theme.print("Training network...")
    model_train(
        fcn_model, optimizer, criterion, device, train_loader, val_loader, epochs
    )
    model_test(fcn_model, criterion, test_loader, device)


@main.command(cls=HelpColorsCommand)
def insight():
    """Run inference on the model."""
    theme.print("Loading network and running inference...")


if __name__ == "__main__":
    main()
