import time

import numpy as np
import torch
from rich.progress import Progress

from . import util
from .paths import CURRENT_MODEL_PATH


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

    with Progress() as prog:
        epoch_bar = prog.add_task("Epochs", total=epochs)
        for epoch in range(epochs):
            ts = time.time()
            train_bar = prog.add_task("Train", total=len(train_loader))
            for iter, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()

                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, labels)

                # Backpropagate
                loss.backward()

                # Update weights
                optimizer.step()

                if iter % 10 == 0:
                    print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
                prog.update(train_bar, advance=1)

            print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))

            current_miou_score, _ = evaluate_validation(
                model, criterion, epoch, validation_loader
            )

            if current_miou_score > best_iou_score:
                best_iou_score = current_miou_score
                torch.save(model.state_dict(), CURRENT_MODEL_PATH)
                # save the best model
            prog.update(epoch_bar, advance=1)


# TODO
def evaluate_validation(model, criterion, epoch, val_loader):
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
            output = model(input)
            loss = criterion(output, label)
            losses.append(loss.item())
            iou = util.compute_iou(output, label)
            mean_iou_scores.append(iou)

            acc = util.compute_pixel_accuracy(output, label)
            accuracy.append(acc)

    print(f"Loss at epoch: {epoch} is {np.mean(losses)}")
    print(f"IoU at epoch: {epoch} is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc at epoch: {epoch} is {np.mean(accuracy)}")

    model.train()

    return np.mean(mean_iou_scores), np.mean(accuracy)


# TODO
def model_test(model, criterion, test_loader):
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

    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing
        for input, label in test_loader:
            output = model(input)
            loss = criterion(output, label)
            losses.append(loss.item())
            iou = util.compute_iou(output, label)
            mean_iou_scores.append(iou)

            acc = util.compute_pixel_accuracy(output, label)
            accuracy.append(acc)

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

    saved_model_path = "Fill Path To Best Model"
    # TODO Then Load your best model using saved_model_path

    inputs = inputs.to(device)

    output_image = fcn_model(inputs)

    fcn_model.train()  # TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return output_image
