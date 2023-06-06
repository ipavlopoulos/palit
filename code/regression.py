# training our CNN
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(41472, 1024, bias=True) # 108928 if the 2nd ConvNet is active
        self.fc2 = nn.Linear(1024, 512, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.fc3 = nn.Linear(512, 1, bias=True) # regression output

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.unsqueeze(1))))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = self.dropout(x) # our augmentation is competitive to dropout at the input (yet locally coherent)
        x = self.fc3(x)
        return x

# the model predicts the average - high variance
def validate(model, dataloader, device="cpu"):
    predictions, gold_labels = [], []
    with torch.no_grad():
        for id_batch, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            model.eval()
            model.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy()[:, 0])
            gold_labels.extend(labels.cpu())
    return predictions, gold_labels

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
            inputs, labels = inputs.to(device), labels.float().flatten().to(device)
            outputs = model(inputs).flatten()
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print the loss for this batch (every n batches, plus the # examples)
            if id_batch % n_print == 0:
                loss, current = loss.item(), (id_batch + 1) * len(inputs)
                losses.append(loss)
                mae = nn.L1Loss()
                mae_loss = mae(outputs, labels)
                mae_losses.append(mae_loss)
                print(f"loss: {loss:.2f} (@all: {np.mean(losses):.2f}) mae: {mae_loss:.2f}  [{current:>2d}/{dataset_size:>2d}]")

        if val_dataloader is not None:
            if (patience>0) and (epoch-best_epoch>patience-1):
                print("Max patience reached, now loading the best checkpoint...")
                model.load_state_dict(torch.load("checkpoint.pt"))
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
                torch.save(model.state_dict(), "checkpoint.pt")
            print(f"Train loss: {train_error:.2f}, Val loss: {val_error:.2f}, Best Epoch: {best_epoch+1} (Val loss: {lowest_error:.2f}) [val var: {np.var(predictions):.2f}, min-max: {min(predictions):.2f}-{max(predictions):.2f}]")
    return losses, mae_losses, train_losses, val_losses

class ImageDataset(TensorDataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        sample, date = self.dataframe.iloc[idx].img.astype(np.float32), self.dataframe.iloc[idx].date
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

