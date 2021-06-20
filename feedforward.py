# tool imports
import pandas as pd
import numpy as np
import cv2
import os

# torch imports
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# feedforward nn
class NN(nn.Module):
    def __init__(self, input_size, no_classes):
        super(NN, self).__init__()
        self.input_layer = nn.Linear(input_size, 50)
        self.hidden_layer = nn.Linear(50, no_classes)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.hidden_layer(x)

        return x

if __name__ == "__main__":
    # data paths
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    full_path = "data/full_data.csv"
    save_path = "model/final_checkpoint.pth.tar"

    # constants
    no_rows = 0
    no_columns = 0

    # data preperation
    train_frame = pd.read_csv(train_path)
    test_frame = pd.read_csv(test_path)
    full_frame = pd.read_csv(full_path)
    no_columns = len(train_frame.columns)
    no_train_values = train_frame.shape[0]
    no_test_values = test_frame.shape[0]

    # data loader
    def load_data(row, frame = "train"):
        return train_frame.iloc[row].values if frame == "train" else test_frame.iloc[row].values

    # pytorch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyperparameters
    input_size = no_columns - 1
    in_channels = 1
    no_classes = 10
    learning_rate = 1e-3
    batch_size = 64
    no_epochs = 4
    load_model = True

    # train dataset
    list_x = []
    list_y = []

    for i in range(no_train_values):
        list_x.append(load_data(i, "train")[1:])
        list_y.append(load_data(i, "train")[0])

    tensor_x = torch.Tensor(list_x).to(torch.float32) # float
    tensor_y = torch.Tensor(list_y).to(torch.int64) # long
    train_dataset = TensorDataset(tensor_x, tensor_y)

    # test dataset
    list_x = []
    list_y = []

    for i in range(no_test_values):
        list_x.append(load_data(i, "test")[1:])
        list_y.append(load_data(i, "test")[0])

    tensor_x = torch.Tensor(list_x).to(torch.float32) # float
    tensor_y = torch.Tensor(list_y).to(torch.int64) # long

    test_dataset = TensorDataset(tensor_x, tensor_y)

    # data loaders
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(dataset = test_dataset, batch_size = 1, shuffle = True)
    test_loader = train_loader

    # initializations
    model = NN(input_size = input_size, no_classes = no_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    total_loss = 0
    no_batches = 0

    # train or load
    if os.path.exists(save_path) and load_model:
        # load model
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # train loop
        for epoch in range(no_epochs):
            total_loss = 0
            no_batches = 0
            for batch_index, (x, y_truth) in enumerate(test_loader):
                x = x.reshape(x.shape[0], -1)
                no_batches += 1

                # forward pass
                y_pred = model(x)
                loss = criterion(y_pred, y_truth)
                total_loss += loss

                # backward pass
                optimizer.zero_grad()
                loss.backward()

                # optimizer step
                optimizer.step()

            print("Epoch: {}| Loss: {}".format(epoch + 1, (total_loss / no_batches)))

        # save the model
        final_checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        torch.save(final_checkpoint, save_path)
        print("Saved Model: {}".format(save_path))

    # accuracy loop
    no_correct = 0
    no_samples = 0
    model.eval() # testing mode
    with torch.no_grad():
        for batch_index, (x, y_truth) in enumerate(train_loader):
            x = x.reshape(x.shape[0], -1)
            y_pred = model(x)
            _, pred = y_pred.max(dim = 1)

            no_correct += (pred == y_truth).sum()
            no_samples += y_truth.shape[0]

        model.train() # training mode
        print("Accuracy: {}%".format(round((float(no_correct) / no_samples) * 100, 2)))
