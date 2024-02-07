import numpy as np
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from . import util
from .paths import CURRENT_MODEL_PATH

console = Console()


# TODO Get class weights
def get_class_weights():
    # TODO for Q4.c || Caculate the weights for the classes
    raise NotImplementedError


def model_train(
    model,
    optimizer,
    criterion,
    device,
    train_loader,
    validation_loader,
    epochs: int,
    cos_opt,
):
    """
    Train a deep learning model using mini-batches.

    - Perform forward propagation in each epoch.
    - Compute loss and conduct backpropagation.
    - Update model weights.
    - Evaluate model on validation set for mIoU score.
    - Save model state if mIoU score improves.
    - Implement early stopping if necessary.

    Returns:
    -------
        None.
    """

    best_iou_score = 0.0
    bad_epochs = 0
    patience = 6

    progress = Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    with progress as prog:
        epoch_bar = prog.add_task("All Epochs", total=epochs)
        for epoch in range(epochs):
            train_bar = prog.add_task(f"Epoch {epoch}.", total=len(train_loader))
            loss = None
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

                optimizer.zero_grad()
                # Compute loss
                loss = criterion(outputs, labels)
                # Backpropagate
                loss.backward()
                # Update weights
                optimizer.step()

                prog.update(
                    train_bar,
                    advance=1,
                    description=f"Epoch {epoch}, IOU: n/a, Acc: n/a, Loss: {loss.item():.2f}",
                )

            cos_opt.step()
            current_miou_score, pixel_acc, loss = evaluate_validation(
                model, criterion, epoch, validation_loader, device
            )

            if current_miou_score > best_iou_score:
                best_iou_score = current_miou_score
                torch.save(model.state_dict(), CURRENT_MODEL_PATH)
                bad_epochs = 0
                # save the best model
            else:
                bad_epochs += 1

            if bad_epochs >= patience:
                console.print(f"Patience ({patience}) exceed, ending training.")
                break

            assert loss is not None
            prog.update(
                train_bar,
                description=f"Epoch {epoch}, IOU: {current_miou_score:.2f}, Acc: {100*pixel_acc:.2f}% Loss: {loss:.2f}",
            )
            prog.update(
                epoch_bar,
                advance=1,
                description=f"All Epochs, IOU: {current_miou_score:.2f}, Patience: {bad_epochs}",
            )


# TODO
def evaluate_validation(model, criterion, epoch, val_loader, device):
    """
    Validate the deep learning model on a validation dataset.

    - Set model to evaluation mode.
    - Disable gradient calculations.
    - Iterate over validation data loader:
        - Perform forward pass to get outputs.
        - Compute loss and accumulate it.
        - Calculate and accumulate mean Intersection over Union (IoU) scores and pixel accuracy.
    - Print average loss, IoU, and pixel accuracy for the epoch.
    - Switch model back to training mode.

    Args:
    ----
        epoch (int): The current epoch number.

    Returns:
    -------
        tuple: Mean IoU score and mean loss for this validation epoch.
    """
    model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    losses = []
    mean_iou_scores = []
    accuracy = []


    with torch.no_grad():
        for input, label in val_loader:
            input = input.to(device)
            label = label.to(device)

            output = model(input)
            loss = criterion(output, label)

            losses.append(loss.item())
            iou = util.compute_iou(output, label)
            mean_iou_scores.append(iou)

            acc = util.compute_pixel_accuracy(output, label)
            accuracy.append(acc)

    # print(f"Loss at epoch: {epoch} is {np.mean(losses)}")
    # print(f"IoU at epoch: {epoch} is {np.mean(mean_iou_scores)}")
    # print(f"Pixel acc at epoch: {epoch} is {np.mean(accuracy)}")

    model.train()

    return np.mean(mean_iou_scores), np.mean(accuracy), loss


# TODO
def model_test(model, criterion, test_loader, device):
    """
    Test the deep learning model using a test dataset.

    - Load the model with the best weights.
    - Set the model to evaluation mode.
    - Iterate over the test data loader:
        - Perform forward pass and compute loss.
        - Accumulate loss, IoU scores, and pixel accuracy.
    - Print average loss, IoU, and pixel accuracy for the test data.
    - Switch model back to training mode.

    Returns:
    -------
        None. Outputs average test metrics to the console.
    """

    model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    losses = []
    mean_iou_scores = []
    accuracy = []

    total_loss = 0.0
    total_iou = 0.0
    total_pixel_acc = 0.0
    total_samples = 0

    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing

        for input, label in test_loader:
            input = input.to(device)
            label = label.to(device)
            output = model(input)
            loss = criterion(output, label)
            losses.append(loss.item())
            iou = util.compute_iou(output, label)
            mean_iou_scores.append(iou)

            acc = util.compute_pixel_accuracy(output, label)
            accuracy.append(acc)

    print(f"Average Loss: {np.mean(losses)}")
    print(f"Average IoU Score: {np.mean(mean_iou_scores)}")
    print(f"Average Pixel Accuracy: {np.mean(accuracy)}")

    model.train()  # TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!


def export_model(fcn_model, device, inputs):
    """
    Export the output of the model for given inputs.

    - Set the model to evaluation mode.
    - Load the model with the best saved weights.
    - Perform a forward pass with the model to get output.
    - Switch model back to training mode.

    Args:
    ----
        inputs: Input data to the model.

    Returns:
    -------
        Output from the model for the given inputs.
    """

    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    path = "models/best.pth"

    # TODO Then Load your best model using saved_model_path
    try:
        checkpoint = torch.load(path, map_location=device)
        fcn_model.load_state_dict(checkpoint["model_state_dict"])
        print("Best model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {path}. Please check the path.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    fcn_model.to(device)
    inputs = inputs.to(device)

    with torch.no_grad():
        output_image = fcn_model(inputs)

    # TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!
    fcn_model.train()

    return output_image
