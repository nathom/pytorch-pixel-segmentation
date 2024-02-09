import json
import os

import click
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from click_help_colors import HelpColorsCommand, HelpColorsGroup
from PIL import Image
from rich.traceback import install
from torch import nn
from torchvision.transforms import v2

from . import theme
from .basic_fcn import FCN
from .dataset import download_data, get_frequency_spectrum, load_dataset
from .erfnet_imagenet import ERFNet
from .train import model_test, model_train
from .unet import UNet
from .unet_resnet import UNetResnet
from .util import compare_images, display_images, find_device

models_dir = "./models"
os.makedirs(models_dir, exist_ok=True)

install(show_locals=False)
torch.manual_seed(42)


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def init_weights(model):
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.ConvTranspose2d):
        # torch.nn.init.xavier_uniform_(model.weight.data)
        torch.nn.init.kaiming_uniform_(
            model.weight.data, mode="fan_in", nonlinearity="relu"
        )
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
    for inputs, labels in val_loader:
        display_images(inputs, labels)
        break
    for inputs, labels in test_loader:
        display_images(inputs, labels)
        break


def validate_filepath(ctx, param, value):
    if value is None:
        theme.print("WARNING: NOT SAVING MODEL!")
        return value
    path1 = os.path.join(models_dir, value + ".pkl")
    path2 = os.path.join(models_dir, value + ".json")
    if value and os.path.exists(path1) or os.path.exists(path2):
        raise click.BadParameter(
            f"Item '{value}' already exists. Please choose a non-existing filepath."
        )
    return value


def validate_filepath_exists(ctx, param, value):
    if value is None:
        return value
    path1 = os.path.join(models_dir, value + ".pkl")
    path2 = os.path.join(models_dir, value + ".json")
    if value and not (os.path.exists(path1) and os.path.exists(path2)):
        raise click.BadParameter(f"Item '{value}' must exist.")
    return value


@click.option("-a", "--augment", help="Choose <=4 from {a,v,h,r}")
@click.option("-e", "--erfnet", help="Use ERFNet", is_flag=True)
@click.option("-ep", "--erfnet-pretrained", help="Use ERFNet pretrained", is_flag=True)
@click.option("-fp", "--fcn-pretrained", help="Use FCN pretrained", is_flag=True)
@click.option("-u", "--unet", help="Use UNet", is_flag=True)
@click.option(
    "-smp",
    "--smp-module",
    help="Use a pretrained model from the SMP package.",
    type=str,
)
@click.option("-nc", "--no-cosine", help="Don't use cosine LR", is_flag=True)
@click.option("-nw", "--no-weights", help="Don't weighting for classes", is_flag=True)
@click.option(
    "-ur",
    "--unet-resnet",
    help="Use UNet with pretrained resnet backbone",
    is_flag=True,
)
@click.option(
    "--dice",
    help="Use dice loss",
    is_flag=True,
)
@click.option(
    "-s",
    "--save",
    help="Saves model to directory with specified name.",
    callback=validate_filepath,
)
@click.option(
    "-l",
    "--load",
    help="Load model and train it",
    callback=validate_filepath_exists,
)
@click.option("--epochs", type=int, default=150, help="Number of epochs")
@click.option("--patience", type=int, default=70, help="Number of max bad epochs")
@main.command(cls=HelpColorsCommand)
def cook(
    load,
    augment,
    erfnet_pretrained,
    erfnet,
    save,
    unet,
    unet_resnet,
    smp_module,
    no_cosine,
    no_weights,
    epochs,
    fcn_pretrained,
    dice,
    patience,
):
    """Train the model."""
    n_class = 21
    device = find_device()

    if load:
        path = os.path.join(models_dir, load + ".pkl")
        fcn_model = torch.load(path)
        learning_rate = 1e-3
    elif erfnet_pretrained:
        theme.print("Using ERFNet")
        pretrained_enc = ERFNet(1000)
        pretrained_model_path = "./data/erfnet_encoder_pretrained.pth.tar"
        state_dict = torch.load(pretrained_model_path, map_location=device)[
            "state_dict"
        ]
        new_state_dict = {}
        for k, v in state_dict.items():
            nk = k.replace("module.", "")
            new_state_dict[nk] = v
        pretrained_enc.load_state_dict(new_state_dict)
        pretrained_enc = next(pretrained_enc.children()).encoder

        fcn_model = ERFNet(
            num_classes=n_class, input_channels=3, encoder=pretrained_enc
        )
        learning_rate = 5e-4
    elif erfnet:
        fcn_model = ERFNet(num_classes=n_class, input_channels=3)
        learning_rate = 5e-4
    elif smp_module == "unet++":
        theme.print("Using smp UNet++")
        fcn_model = smp.create_model("unet", classes=21)
        learning_rate = 5e-4
    elif smp_module == "unet":
        theme.print("Using smp UNet")
        fcn_model = smp.create_model("unetplusplus", classes=21)
        learning_rate = 1e-4
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
        learning_rate = 1e-4
    elif fcn_pretrained:
        theme.print("Using FCN with resnet backbone")
        fcn_model = FCN(n_class=n_class, resnet_backbone=True)
        learning_rate = 0.001
    else:
        theme.print("Using FCN")
        fcn_model = FCN(n_class=n_class)
        fcn_model.apply(init_weights)
        learning_rate = 0.001

    # calculated with dataset:get_freqency_spectrum
    if not no_weights:
        theme.print("Weighting loss by inverse class frequency")
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
    else:
        weights = None

    augment_transform = get_augment_transforms(augment)
    optimizer = torch.optim.Adam(params=fcn_model.parameters(), lr=learning_rate)
    theme.print(f"Learning rate: {learning_rate}, weight decay: {1e-4}")
    if dice:
        criterion = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE)
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

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
    validation_losses, training_losses, validation_ious, validation_accs = (
        None,
        None,
        None,
        None,
    )
    try:
        (
            validation_losses,
            training_losses,
            validation_ious,
            validation_accs,
        ) = model_train(
            fcn_model,
            optimizer,
            criterion,
            device,
            train_loader,
            val_loader,
            epochs,
            cos_opt,
            patience,
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
                        "erfnet": erfnet_pretrained,
                        "save": save,
                        "unet": unet,
                        "unet_resnet": unet_resnet,
                        "smp_module": smp_module,
                        "no_cosine": no_cosine,
                        "epochs": epochs,
                        "loss": validation_losses,
                        "training_loss": training_losses,
                        "ious": validation_ious,
                        "accs": validation_accs,
                    },
                    f,
                    indent=4,
                )


@click.option("-d", "--display", is_flag=True)
@click.option("-c", "--compare")
@click.option("-i", "--img", type=str)
@click.option("-l", "--load", help="Loads cached model.")
@main.command(cls=HelpColorsCommand)
def insight(load, display, compare, img):
    """Run inference on the model."""

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    inverse_normalize = v2.Normalize(
        mean=[-mean[i] / std[i] for i in range(3)],
        std=[1 / std[i] for i in range(3)],
    )
    input_transform = v2.Compose(
        [
            v2.ToImage(),
            # v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std),
        ]
    )

    if load is not None:
        path = os.path.join(models_dir, load + ".pkl")
        model = torch.load(path)

        theme.print("Loading network and running inference...")
        device = find_device()
        criterion = torch.nn.CrossEntropyLoss()
        train_loader, _, test_loader = load_dataset(
            None, input_transform, MaskToTensor()
        )

        model_test(model, criterion, test_loader, device)

    if display:
        if img is not None:
            img_obj = Image.open(img).convert("RGB").resize((224, 224))
            inputs = input_transform(img_obj)
            inputs = inputs.unsqueeze(0)
            print(inputs.shape)
            inputs2 = inverse_normalize(inputs)
            outputs = model(inputs.to(device)).cpu().detach().numpy()
            outputs = np.argmax(outputs, axis=1).astype(np.uint8)
            display_images(inputs2.cpu(), outputs)

        for inputs, labels in test_loader:
            print(inputs.shape)
            labels = v2.ToDtype(torch.uint8)(labels)
            inputs2 = inverse_normalize(inputs)
            outputs = model(inputs.to(device)).cpu().detach().numpy()
            outputs = np.argmax(outputs, axis=1).astype(np.uint8)
            display_images(inputs2.cpu(), outputs)

    if compare:
        files = []
        for file in os.listdir(os.getcwd() + "/models"):
            if file.endswith(".pkl"):
                files.append(file)
        if img is not None:
            theme.print(f"Showing custom image {img}")
            img_obj = Image.open(img).convert("RGB").resize((224, 224))
            inputs = input_transform(img_obj)
            inputs = inputs.unsqueeze(0)
            compare_images(files, int(compare), 1, inputs)
        else:
            compare_images(files, int(compare), 1)


@click.option("-l", "--load", help="Loads cached model.")
@main.command(cls=HelpColorsCommand)
def plot(
    load,
):
    path = os.path.join(models_dir, load + ".json")
    with open(path) as f:
        info = json.load(f)

    plot_training_metrics(
        info["training_loss"],
        info["loss"],
        len(info["loss"]) - 1,
        info["ious"],
        info["accs"],
    )
    # {
    #     "learning_rate": learning_rate,
    #     "augment": augment,
    #     "erfnet": erfnet_pretrained,
    #     "save": save,
    #     "unet": unet,
    #     "unet_resnet": unet_resnet,
    #     "smp_module": smp_module,
    #     "no_cosine": no_cosine,
    #     "epochs": epochs,
    #     "loss": validation_losses,
    #     "training_loss": training_losses,
    #     "ious": validation_ious,
    #     "accs": validation_accs,
    # },


def plot_training_metrics(
    loss_train, loss_val, early_stop_epoch, iou_val, accuracy_val
):
    # Create a figure and subplots
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))

    epoch_range = np.linspace(0, len(loss_val), len(loss_train))
    # Plot Training and Validation Loss with Early Stop Indicator
    axs[0].plot(epoch_range, loss_train, label="Training Loss", color="blue")
    axs[0].plot(
        range(1, len(loss_val) + 1), loss_val, label="Validation Loss", color="orange"
    )
    axs[0].axvline(
        x=early_stop_epoch + 1, color="red", linestyle="--", label="Early Stop"
    )
    axs[0].scatter(
        early_stop_epoch, loss_val[early_stop_epoch], color="red", marker="x"
    )
    axs[0].set_title("Training and Validation Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Plot Validation IOU
    axs[1].plot(iou_val, label="Validation IOU", color="green")
    axs[1].set_title("Validation IOU")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("IOU")
    axs[1].legend()

    # Plot Validation Accuracy
    axs[2].plot(accuracy_val, label="Validation Accuracy", color="purple")
    axs[2].set_title("Validation Accuracy")
    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("Accuracy")
    axs[2].legend()

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


def get_augment_transforms(augment: str | None):
    augment_transforms: list[v2.Transform] = [v2.ToImage()]
    if augment is not None:
        if "a" in augment:
            theme.print("Affine transform selected.")
            augment_transforms.append(
                v2.RandomAffine(degrees=(-20, 20), shear=(-20, 20))
            )
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
