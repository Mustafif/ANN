import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SubsetRandomSampler

from ann import ForwardModel

device = torch.device("cuda" if torch.cuda.is_available() else "mps:0"  if torch.backends.mps.is_available() else "cpu")

class SimDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.copy().reset_index(drop=True)
        self.data = self.data[self.data["sigma"] > 1e-8] # clean dataset
        self.base = [
            "S0","m","r","T","callput","alpha","beta","omega","gamma","lambda"
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        base_vals =  [row[b] for b in self.base]
        X = torch.tensor(base_vals, dtype=torch.float32)
        target = row["sigma"]
        Y = torch.tensor(target, dtype=torch.float32)
        return X, Y

def train_test_split(data, test_size=0.3, random_state=42):
    train_data, test_data = sklearn.model_selection.train_test_split(
        data, test_size=test_size, random_state=random_state, shuffle=True
    )
    return SimDataset(train_data), SimDataset(test_data)

def train_val_split(dataset, val_size=0.2, random_state=42):
    # Get indices of the full dataset
    indices = list(range(len(dataset)))

    # Split indices into train and validation
    train_indices, val_indices = sklearn.model_selection.train_test_split(
        indices, test_size=val_size, random_state=random_state
    )

    # Create samplers for train and validation
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    return train_sampler, val_sampler

def train_model(model: nn.Module, train_loader, val_loader, criterion, optimizer, device, epochs):
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr = optimizer.param_groups[0]["lr"],
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy="cos"
    )

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_X, batch_Y in train_loader:
            x, y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()

            output = model(x.float())
            target = y.float().unsqueeze(1)

            loss = criterion(output, target)
            loss.backward()

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss/len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                x, y  = batch_X.to(device), batch_Y.to(device)
                output = model(x.float())
                target = y.float().unsqueeze(1)
                loss = criterion(output, target)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(
            "Epoch {}: Train Loss {:.4f} Val Loss {:.4f}".format(
                epoch + 1, avg_train_loss, avg_val_loss
            )
        )

    return model, train_losses, val_losses

def eval_model(model: nn.Module, test_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0  # Initialize total loss counter
    predictions = []  # List to store model predictions
    targets = []  # List to store targets

    with torch.no_grad():  # Disable gradient computation for evaluation
        for X, Y in test_loader:  # Loop through batches
            X, Y = X.to(device), Y.to(device)  # Move data to device
            output = model(X)  # Get model predictions
            # Reshape target to match output dimensions
            target = Y.float().unsqueeze(1)
            loss = criterion(output, target)  # Calculate loss for batch
            total_loss += loss.item()  # Add batch loss to total

            # Store predictions and targets as flattened arrays
            predictions.extend(
                output.cpu().numpy().flatten()
            )  # Convert predictions to numpy array
            targets.extend(Y.cpu().numpy().flatten())  # Convert targets to numpy array

    avg_loss = total_loss / len(
        test_loader
    )  # Calculate average loss across all batches
    return avg_loss, np.array(predictions), np.array(targets)

def main():
    data = pd.read_csv("datasets/varying_garch_parallel_1000.csv")

    train_data, test_data = train_test_split(data)
    num_workers = 6
    lr = 0.001
    weight_decay = 1e-6
    batch_size = 32
    epochs = 1000
    dropout_rate=0.0

    train_sampler, val_sampler = train_val_split(train_data, val_size=0.2)
    test_sampler = RandomSampler(test_data)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    model = ForwardModel(dropout_rate=dropout_rate).to(device)
    criterion = nn.HuberLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    trained_model, tl, vl = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, epochs=epochs
    )
    # Evaluation
    train_loss, train_pred, train_target = eval_model(
        trained_model, train_loader, criterion, device
    )
    test_loss, test_pred, test_target = eval_model(
        trained_model, test_loader, criterion, device
    )

    print(train_loss, test_loss)


if __name__ == "__main__":
    main()
