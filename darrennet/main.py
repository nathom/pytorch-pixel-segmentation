import click
import numpy as np
import torch
import torchvision.transforms as standard_transforms
from click_help_colors import HelpColorsCommand, HelpColorsGroup
from rich.traceback import install
from torch import nn

from . import theme
from .basic_fcn import FCN
from .dataset import download_data, load_dataset
from .train import model_test, model_train

install(show_locals=True)

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
input_transform = standard_transforms.Compose(
    [standard_transforms.ToTensor(), standard_transforms.Normalize(*mean_std)]
)

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


@main.command(cls=HelpColorsCommand)
def cook():
    """Train the model."""
    device = find_device()
    fcn_model = FCN(n_class=n_class)
    fcn_model.apply(init_weights)

    # TODO determine which device to use (cuda or cpu)
    # Check that MPS is available
    optimizer = torch.optim.Adam(params=fcn_model.parameters(), lr=0.001)

    criterion = torch.nn.CrossEntropyLoss()

    fcn_model = fcn_model.to(device)
    train_loader, val_loader, test_loader = load_dataset(
        input_transform, MaskToTensor()
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
