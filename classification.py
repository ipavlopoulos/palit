import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm
import pandas as pd
import skimage
import random
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import figure
from sklearn.metrics import *
from skimage import transform
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim



line_data = pd.read_pickle("palit_lines.csv")
print("Images num:", line_data.shape[0])
line_data = line_data.explode("img")
print("Lines num:", line_data.shape[0])

# removing empty & focusing on 4 centuries
line_data = line_data[line_data.img.notna()]
line_data = line_data[line_data.date.isin({1,2,3,4})]

# cleaning
x_dim = line_data.img.apply(lambda x: x.shape[0]) # mean 50
y_dim = line_data.img.apply(lambda x: x.shape[1]) # mean 300
line_data = line_data[(x_dim>50)&(y_dim>300)]
print("Lines num after preprocessing:", line_data.shape[0])

# resizing
line_data.img = line_data.img.apply(lambda img: transform.resize(img, (50, 300)))


"""# Modeling"""
print("Modeling...")
# SPLIT
#test_ix  = line_data.uid%5==0
#trainset = line_data[~test_ix]
#testset = line_data[test_ix]
#valset, testset = train_test_split(testset, test_size=testset.shape[0]//2, random_state=2023)
trainset, testset = train_test_split(line_data, test_size=0.1, random_state=39)
trainset, valset = train_test_split(trainset, test_size=testset.shape[0], random_state=39)

# allowing lines from the same manuscript in the validation
print("Train, Val, Test:", trainset.shape[0], valset.shape[0], testset.shape[0])


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, 3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(3, 3)
        self.fc1 = nn.Linear(128*1584, 1024, bias=True)
        self.fc2 = nn.Linear(1024, 128, bias=True)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 4) # regression output

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.unsqueeze(1))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        x = self.fc3(x)
        return x

def validate(model, dataloader, device="cpu"):
    predictions, gold_labels = [], []
    with torch.no_grad():
        for id_batch, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.argmax(1).cpu().numpy())
            gold_labels.extend(labels.cpu().numpy())
    return predictions, gold_labels


class ImageDataset(TensorDataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx].img.astype(np.float32)
        label = self.dataframe.iloc[idx].date-1 # 0-4 for 1-4CE
        if self.transform is not None:
            sample = self.transform(sample)
            sample = sample.reshape(sample.shape[1], sample.shape[2]) # grayscale
        return sample, label

def train(model, dataloader, optimizer, criterion, val_dataloader=None, patience=3, val_metric = mean_absolute_error, n_print=10, N_EPOCHS=4, device="cpu"):
    losses, mae_losses, val_losses, train_losses = [], [], [], []
    dataset_size = len(dataloader.dataset)
    lowest_error = 100000
    best_epoch = -1
    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")

        # Loop over batches in an epoch using DataLoader
        for id_batch, (inputs, labels) in enumerate(dataloader):
            model.train()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print the loss for this batch (every n batches, plus the # examples)
            if id_batch % n_print == 0:
                loss, current = loss.item(), (id_batch + 1) * len(inputs)
                losses.append(loss)
                mae = nn.L1Loss()
                mae_loss = mae(outputs.argmax(1).float(), labels)
                mae_losses.append(mae_loss)
                print(f"loss: {loss:.2f} (@all: {np.mean(losses):.2f}) mae: {mae_loss:.2f}  [{current:>2d}/{dataset_size:>2d}]")

        if val_dataloader is not None:
            if (patience>0) and (epoch-best_epoch>patience):
                print("Max patience reached.")
                print("Loading the best checkpoint...")
                model.load_state_dict(torch.load("cnn_clf.pt"))
                print("Exiting...")
                break
            predictions, labels = validate(model, val_dataloader, device=device)
            val_error = val_metric(predictions, labels)
            train_predictions, train_labels = validate(model, dataloader, device=device)
            train_error = val_metric(train_predictions, train_labels)
            val_losses.append(val_error)
            train_losses.append(train_error)
            if (patience>0) and (val_error<lowest_error):
                lowest_error = val_error
                best_epoch = epoch
                torch.save(model.state_dict(), "cnn_clf.pt")
            print(f"Train loss: {train_error:.2f}, Val loss: {val_error:.2f}, Best Epoch: {best_epoch+1} (Val loss: {lowest_error:.2f} - var: {np.var(predictions):.2f})")
    return losses, mae_losses, train_losses, val_losses


fragmenting = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(3), # turn
    transforms.RandomResizedCrop((50, 300), scale=(0.75, 1.0), ratio=(1,6)), # crop n scale
    transforms.GaussianBlur(3), # blur
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, value=1),
])

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = Net().to(device)
    mse_loss, mae_loss, train_losses, val_losses = train(net, DataLoader(ImageDataset(trainset, transform=fragmenting), batch_size=32, shuffle=True, drop_last=True),
                                                         optimizer=optim.Adam(net.parameters(), lr=1e-3), #, weight_decay=1e-5),
                                                         criterion=nn.CrossEntropyLoss(), # nn.MSELoss()
                                                         N_EPOCHS=200,
                                                         device=device,
                                                         val_metric=mean_absolute_error,
                                                         patience=40,
                                                         val_dataloader=DataLoader(ImageDataset(valset), batch_size=1)
                                                         )

    print(f"Parameters: {nn.utils.parameters_to_vector(net.parameters()).numel():.1f}")
    net.eval()

    ev = trainset.sample(100) # evaluating on train
    predictions, labels = validate(net, DataLoader(ImageDataset(ev), batch_size=1), device=device)
    print("Train loss:", nn.L1Loss()(torch.Tensor(predictions), torch.Tensor(labels)).numpy())

    # evaluation on test
    predictions, labels = validate(net, DataLoader(ImageDataset(testset), batch_size=1), device=device)
    print("Test loss", nn.L1Loss()(torch.Tensor(predictions), torch.Tensor(labels)).numpy())
    testset["cnn"] = predictions
    testset.to_pickle("clftest.csv")
