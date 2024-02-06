import click
import numpy as np
import torch
from click_help_colors import HelpColorsCommand, HelpColorsGroup
from rich.traceback import install
from torch import nn
from torchvision.transforms import v2

from . import theme
from .basic_fcn import FCN
from .dataset import download_data, get_frequency_spectrum, load_dataset
from .train import model_test, model_train
from .util import find_device

install(show_locals=True)


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def init_weights(model):
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(model.weight.data)
        assert model.bias is not None
        torch.nn.init.normal_(model.bias.data)  # xavier not applicable for biases


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
def info():
    get_frequency_spectrum()


@click.option("-w", "--weight", help="Weight loss by inverse frequency.", is_flag=True)
@click.option("-a", "--augment", help="Choose <=4 from {a,v,h,r}")
@main.command(cls=HelpColorsCommand)
def cook(weight, augment):
    """Train the model."""
    epochs = 20
    n_class = 21
    device = find_device()
    fcn_model = FCN(n_class=n_class)
    fcn_model.apply(init_weights)

    if weight:
        theme.print("Weighting loss by inverse class frequency")
        # calculated with dataset:get_freqency_spectrum
        weights = torch.Tensor(
            [
                0.002995969342219151,
                0.47902764537654907,
                0.5324441987423122,
                0.25992938107232816,
                0.4927520784480921,
                0.3255679657459964,
                0.11598093326643251,
                0.30993979538475974,
                0.12336882555439917,
                0.20796595530283946,
                0.29532387888079725,
                0.17547388957632715,
                0.16052417758703308,
                0.26889475704663635,
                0.2918671161786431,
                0.025474723651872998,
                0.7314641941710704,
                1.0,
                0.13692179197839105,
                0.1928081677593714,
                0.17702469844916716,
            ]
        ).to(device)
        learning_rate = 0.01
    else:
        learning_rate = 0.001
        weights = None

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

    optimizer = torch.optim.Adam(params=fcn_model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    fcn_model = fcn_model.to(device)
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # Final transform for input
    input_transform = v2.Compose(
        [
            v2.ToImage(),
            # v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(*mean_std),
        ]
    )

    mask_transform = v2.Compose(
        [
            v2.ToImage(),
            # v2.Resize((224, 224)),
            # Change shape from (1,224,224) -> (224, 224)
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
