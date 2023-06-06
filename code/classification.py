import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import skimage
from skimage import transform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import *
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import confusion_matrix
import xgboost as xgb 

from tqdm.notebook import tqdm
import os
import random

class CNNCLF(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.CNN = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
        )
        
        self.MLP = nn.Sequential(
            nn.Linear(41472, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.1, inplace=True),
            
            nn.Linear(1024, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.1, inplace=True),
            
            nn.Linear(512, n_classes, bias=True)
        )
        
    def forward(self, x):
        x = self.CNN(x.unsqueeze(1))
        x = torch.flatten(x, 1)
        x = self.MLP(x)
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


# the model predicts the average - high variance
def validate_clf(model, dataloader, device="cpu", criterion = nn.CrossEntropyLoss()):
    prediction_list, labels_list = [], []
    model.eval()
    loss = 0
    with torch.no_grad():
        for id_batch, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            model.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            labels = np.argmax(labels.cpu().detach().numpy(), axis=1)
            predictions = np.argmax(outputs.cpu().detach().numpy(), axis=1)
            prediction_list.append(predictions)
            labels_list.append(labels)
    p = np.concatenate(prediction_list)
    g = np.concatenate(labels_list)
    acc = accuracy_score(g, p)
    return p, g, loss, acc

def train_clf(model, dataloader, optimizer, criterion=nn.CrossEntropyLoss(), val_dataloader=None, patience=3, N_EPOCHS=100, device="cpu"):
    losses, mae_losses, val_losses, train_losses = [], [], [], []
    dataset_size = len(dataloader.dataset)
    lowest_error = 100000
    best_epoch = -1
    
    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loss = 0
        # Loop over batches in an epoch using DataLoader
        for id_batch, (inputs, labels) in enumerate(dataloader):
            model.train()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if val_dataloader is not None:
            if (patience>0) and (epoch-best_epoch>patience-1) and epoch > 50:
                print("Max patience reached, now loading the best checkpoint...")
                model.load_state_dict(torch.load("checkpoint.pt"))
                print("Exiting...")
                break
            predictions, labels, val_loss, val_acc = validate_clf(model, val_dataloader, device=device)
            val_losses.append(val_loss)
            train_losses.append(train_loss)
            if (patience>0) and (val_loss<lowest_error):
                lowest_error = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), "checkpoint.pt")
            print(f"Train loss: {train_loss:.2f}, Val loss: {val_loss:.2f}, Val acc: {val_acc:.2f}") 
            print(f"Best Epoch: {best_epoch+1} (Val loss: {lowest_error:.2f})") 
            print(f"[val var: {np.var(predictions):.2f}, min-max: {min(predictions):.2f}-{max(predictions):.2f}]")
    model.eval()
    return model, train_losses, val_losses

class ImageDataset(TensorDataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.labels = np.array(self.dataframe['date'])
        self.labels = np.reshape(self.labels, (self.labels.shape[0], 1))
        self.labels = OneHotEncoder(sparse=False).fit_transform(self.labels)
        
    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        sample, date = self.dataframe.iloc[idx].img.astype(np.float32), self.labels[idx]
        if self.transform is not None:
            sample = self.transform(sample)
            sample = sample.reshape(sample.shape[1], sample.shape[2]) # grayscale
        return sample, date # 1CE-4CE

augment = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(3), # turn
    transforms.RandomResizedCrop((50, 300), scale=(0.75, 1.0), ratio=(1, 6)), # crop n scale (add scale to control this)
    #transforms.RandomCrop((30, 30)),
    transforms.GaussianBlur(3), # blur
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, value=0.5), # erase
])

cc_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop((50, 300)),
    transforms.ToTensor(),
])

rs_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((50,300)),
    transforms.ToTensor(),
])

white_fragment_ablation = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((50,300)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=1, value=0), # erase
])

black_fragment_ablation = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((50,300)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=1, value=1), # erase
])
