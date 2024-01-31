import gc
import time

import numpy as np
import torch
import torchvision.transforms as standard_transforms
import voc
from basic_fcn import FCN
from torch import nn
from torch.utils.data import DataLoader


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)  # xavier not applicable for biases


# TODO Get class weights
def get_class_weights():
    # TODO for Q4.c || Caculate the weights for the classes
    raise NotImplementedError


mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
input_transform = standard_transforms.Compose(
    [standard_transforms.ToTensor(), standard_transforms.Normalize(*mean_std)]
)

target_transform = MaskToTensor()

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

epochs = 20

n_class = 21

fcn_model = FCN(n_class=n_class)
fcn_model.apply(init_weights)

# TODO determine which device to use (cuda or cpu)
# Check that MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")

elif torch.cuda.is_available():
    device = torch.device("cuda")

else:
    if not torch.backends.mps.is_built():
        raise Exception(
            "MPS not available because the current PyTorch install was not "
            "built with MPS enabled."
        )
    else:
        raise Exception(
            "MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine."
        )


optimizer = torch.optim.Adam(params=fcn_model.parameters(), lr=0.001)

criterion = torch.nn.CrossEntropyLoss()

fcn_model = FCN.to(device)


def train():
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

    for epoch in range(epochs):
        ts = time.time()
        for iter, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = fcn_model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backpropagate
            loss.backward()

            # Update weights
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))

        current_miou_score = val(epoch)

        if current_miou_score > best_iou_score:
            best_iou_score = current_miou_score
            # save the best model


# TODO
def val(epoch):
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
    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing
        for iter, (input, label) in enumerate(val_loader):
            pass

    print(f"Loss at epoch: {epoch} is {np.mean(losses)}")
    print(f"IoU at epoch: {epoch} is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc at epoch: {epoch} is {np.mean(accuracy)}")

    fcn_model.train()  # TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(mean_iou_scores)


# TODO
def modelTest():
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

    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing
        for iter, (input, label) in enumerate(test_loader):
            pass

    fcn_model.train()  # TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!


def export_model(inputs):
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


if __name__ == "__main__":
    val(0)  # show the accuracy before training
    train()
    modelTest()

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()
