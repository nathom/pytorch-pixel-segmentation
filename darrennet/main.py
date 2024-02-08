import json
import os

import click
import numpy as np
import segmentation_models_pytorch as smp
import torch
from click_help_colors import HelpColorsCommand, HelpColorsGroup
from rich.traceback import install
from torch import nn
from torchvision.transforms import v2

from . import theme
from .basic_fcn import FCN
from .dataset import download_data, get_frequency_spectrum, load_dataset
from .erfnet import ERF
from .train import model_test, model_train
from .unet import UNet
from .unet_resnet import UNetResnet
from .util import display_images, find_device

models_dir = "./models"
os.makedirs(models_dir, exist_ok=True)

install(show_locals=False)
torch.manual_seed(42)


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
    """Print frequency spectrum for class weights."""
    get_frequency_spectrum()


@click.option("-a", "--augment", help="Choose <=4 from {a,v,h,r}")
@click.option("-l", "--load", help="Load model")
@main.command(cls=HelpColorsCommand)
def data(augment, load):
    aug = get_augment_transforms(augment)
    # aug = None
    train_loader, val_loader, test_loader = load_dataset(aug, None, None)
    for inputs, labels in train_loader:
        display_images(inputs, labels)
        break
    for inputs, labels in val_loader:
        display_images(inputs, labels)
        break
    for inputs, labels in test_loader:
        display_images(inputs, labels)
        break


def validate_filepath(ctx, param, value):
    path1 = os.path.join(models_dir, value + ".pkl")
    path2 = os.path.join(models_dir, value + ".json")
    if value and os.path.exists(path1) or os.path.exists(path2):
        raise click.BadParameter(
            f"Item '{value}' already exists. Please choose a non-existing filepath."
        )
    return value


@click.option("-a", "--augment", help="Choose <=4 from {a,v,h,r}")
@click.option("-e", "--erfnet", help="Use ERFNet", is_flag=True)
@click.option("-u", "--unet", help="Use UNet", is_flag=True)
@click.option(
    "-smp",
    "--smp-module",
    help="Use a pretrained model from the SMP package.",
    type=str,
)
@click.option("-nc", "--no-cosine", help="Don't use cosine LR", is_flag=True)
@click.option(
    "-ur",
    "--unet-resnet",
    help="Use UNet with pretrained resnet backbone",
    is_flag=True,
)
@click.option(
    "-s",
    "--save",
    help="Saves model to directory with specified name.",
    callback=validate_filepath,
)
@click.option("--epochs", type=int, default=150, help="Number of epochs")
@main.command(cls=HelpColorsCommand)
def cook(augment, erfnet, save, unet, unet_resnet, smp_module, no_cosine, epochs):
    """Train the model."""
    n_class = 21
    device = find_device()

    if erfnet:
        theme.print("Using ERFNet")
        fcn_model = ERF(num_classes=n_class, input_channels=3)
        learning_rate = 5e-4
    elif smp_module == "unet++":
        theme.print("Using smp UNet++")
        fcn_model = smp.create_model("unetplusplus", classes=21)
        learning_rate = 1e-4
    elif smp_module == "unet":
        theme.print("Using smp unet")
        fcn_model = smp.create_model("unet", classes=21)
        learning_rate = 1e-3
    elif smp_module == "manet":
        theme.print("Using smp manet")
        fcn_model = smp.create_model("manet", classes=21)
        learning_rate = 1e-3
    elif smp_module == "fpn":
        theme.print("Using smp FPN")
        fcn_model = smp.create_model("unetplusplus", classes=21)
        learning_rate = 1e-3
    elif smp_module == "linknet":
        theme.print("Using smp LinkNet")
        fcn_model = smp.create_model("linknet", classes=21)
        learning_rate = 1e-2
    elif unet:
        theme.print("Using UNet")
        fcn_model = UNet(3, n_class)
        learning_rate = 1e-3
    elif unet_resnet:
        theme.print("using UNet with pretrained ResNet backbone")
        fcn_model = UNetResnet(n_class)
        learning_rate = 1e-3
    else:
        theme.print("Using FCN")
        fcn_model = FCN(n_class=n_class)
        fcn_model.apply(init_weights)
        learning_rate = 0.001

    theme.print("Weighting loss by inverse class frequency")
    # calculated with dataset:get_freqency_spectrum
    weights = torch.Tensor(
        [
            1.7765756464776548,
            41.14605024958389,
            41.923407019122614,
            35.610010988370455,
            41.358926183434754,
            37.852584879629106,
            26.134633569759732,
            37.38219759224805,
            26.90527861619218,
            33.178439919075075,
            36.909468575338494,
            31.208512027144945,
            30.14330606533797,
            35.96210892771767,
            36.792639486832435,
            9.879025266255747,
            43.95102624926591,
            45.53468859040118,
            28.199209349833556,
            32.31202739148328,
            31.31266311222871,
        ]
    ).to(device)

    augment_transform = get_augment_transforms(augment)
    optimizer = torch.optim.Adam(params=fcn_model.parameters(), lr=learning_rate)
    theme.print(f"Learning rate: {learning_rate}, weight decay: {1e-4}")
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    criterion = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE)

    if not no_cosine:
        theme.print("Using cosine LR")
        cos_opt = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        cos_opt = None

    fcn_model = fcn_model.to(device)
    mean, std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Final transform for input
    input_transform = v2.Compose(
        [
            v2.ToImage(),
            # v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std),
        ]
    )

    mask_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.long),
            # v2.ToImage(),
            # v2.Resize((224, 224)),
            # Change shape from (1,224,224) -> (224, 224)
            # v2.Lambda(lambda x: torch.squeeze(x, dim=0)),
        ]
    )

    train_loader, val_loader, test_loader = load_dataset(
        augment_transform,
        input_transform,
        mask_transform,
    )

    theme.print("Training network...")
    validation_losses, validation_ious, validation_accs = None, None, None
    try:
        validation_losses, validation_ious, validation_accs = model_train(
            fcn_model,
            optimizer,
            criterion,
            device,
            train_loader,
            val_loader,
            epochs,
            cos_opt,
        )

        theme.print("Testing network...")
        model_test(
            fcn_model,
            criterion,
            test_loader,
            device,
        )

    finally:
        if save is not None:
            path = os.path.join(models_dir, save + ".pkl")
            print(f"Saving model to {path}")
            torch.save(fcn_model, path)
            data_path = os.path.join(models_dir, save + ".json")
            with open(data_path, "w") as f:
                json.dump(
                    {
                        "learning_rate": learning_rate,
                        "augment": augment,
                        "erfnet": erfnet,
                        "save": save,
                        "unet": unet,
                        "unet_resnet": unet_resnet,
                        "smp_module": smp_module,
                        "no_cosine": no_cosine,
                        "epochs": epochs,
                        "loss": validation_losses,
                        "ious": validation_ious,
                        "accs": validation_accs,
                    },
                    f,
                    indent=4,
                )


@click.option("-l", "--load", help="Loads cached model.")
@click.option("-d", "--display", is_flag=True)
@main.command(cls=HelpColorsCommand)
def insight(load, display):
    """Run inference on the model."""
    path = os.path.join(models_dir, load + ".pkl")
    model = torch.load(path)

    theme.print("Loading network and running inference...")
    device = find_device()
    criterion = torch.nn.CrossEntropyLoss()
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    inverse_normalize = v2.Normalize(
        mean=[-mean[i] / std[i] for i in range(3)], std=[1 / std[i] for i in range(3)]
    )
    input_transform = v2.Compose(
        [
            v2.ToImage(),
            # v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std),
        ]
    )

    train_loader, _, test_loader = load_dataset(None, input_transform, MaskToTensor())

    model_test(model, criterion, test_loader, device)

    if display:
        for inputs, labels in test_loader:
            labels = v2.ToDtype(torch.uint8)(labels)
            inputs2 = inverse_normalize(inputs)
            outputs = model(inputs.to(device)).cpu().detach().numpy()
            outputs = np.argmax(outputs, axis=1).astype(np.uint8)
            display_images(inputs2.cpu(), outputs)


def get_augment_transforms(augment: str | None):
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

    augment_transforms.append(v2.ToPILImage())
    return v2.Compose(augment_transforms)


if __name__ == "__main__":
    main()
