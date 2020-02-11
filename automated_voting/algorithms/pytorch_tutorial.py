import math
import pickle
import gzip
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import optim
import torch.nn.functional as F

""" Refactored code from pytorch_nn_tutorial.ipynb """

################################################################################
# Data
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

################################################################################
# 2. Model
# Logistic Regression with one linear (wx + b) layer, with input size (784, 10)
class MnistLogistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)

# Basic CNN with 3 conv layers, 3 relu and 1 pooling layer
class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))

# 2.2 Define loss function for batches
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

################################################################################
# 3. Learning
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):

    print("-------------------------------------------------------")
    for epoch in range(epochs):

        # Training
        model.train()
        t_losses, t_nums = zip(
            *[loss_batch(model, loss_func, xb, yb, opt) for xb, yb in train_dl]
        )

        train_loss = np.sum(np.multiply(t_losses, t_nums)) / np.sum(t_nums)

        # Evaluation
        model.eval()
        with torch.no_grad():
            v_losses, v_nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(v_losses, v_nums)) / np.sum(v_nums)

        print(f"Epoch: {epoch}, Validation Loss: {val_loss}, Train loss: {train_loss}")
        print("-------------------------------------------------------")

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

################################################################################
# 4. Run everything!
# 4.1 Test Logistic Regression
def main(model="Logistic"):
    # 0. Parameters
    epochs = 10
    lr = 0.5
    bs = 64

    # 1. Data
    path = Path("../notebooks/data") / "mnist"
    fname = "mnist.pkl.gz"

    # 1.1 Get data from gzip file, import train and validation datasets
    with gzip.open((path / fname).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

    # 1.2 Create TensorDataset to wrap the train/val datasets -> First convert to tensor!
    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid)
    )
    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)

    # 1.3 DataLoader will help us yield the x and y batches cleanly according to batch size bs
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs)

    # 2. Model
    # 2.1 Define Model
    if model == "Logistic":
        model = MnistLogistic()
    else:  # if model == "CNN"
        model = MnistCNN()

    # 2.2 Define loss function -> For now, we will use cross entropy
    loss_func = F.cross_entropy

    # 2.3 Define optimizer
    opt = optim.SGD(model.parameters(), lr=lr)

    # 3. Train!
    fit(epochs, model, loss_func, opt, train_dl, valid_dl)

    # 4. Validate predictions
    preds = model(x_valid)
    print("Accuracy:", accuracy(preds, y_valid).item())


if __name__ == "__main__":
    main(model="CNN")


