import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SubsetRandomSampler
from torch_optimizer import Lookahead

from ann import ForwardModel

module = ForwardModel
dataset = "Duan2_50K/dataset_duan.csv"
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps:0"
    if torch.backends.mps.is_available()
    else "cpu"
)
dlayer = False


class SimDataset(Dataset):
    def __init__(self, dataframe):
        # Filter and reset index
        df = dataframe.copy().reset_index(drop=True)
        df = df[df["V"] > 0.5]  # clean dataset

        base_cols = [
            "S0",
            "m",
            "r",
            "T",
            "corp",
            "alpha",
            "beta",
            "omega",
            "gamma",
            "lambda",
        ]

        # 1. Base Features (Vectorized)
        base_vals = df[base_cols].values.astype(np.float64)

        # 2. Log Features (Vectorized)
        # Add epsilon to avoid log(0)
        eps = 1e-8
        log_vals = np.column_stack(
            [
                np.log(df["alpha"].values + eps),
                np.log(df["beta"].values + eps),
                np.log(df["omega"].values + eps),
                np.log(df["gamma"].values + eps),
                np.log(df["lambda"].values + eps),
            ]
        ).astype(np.float64)

        # 3. Concatenate and Convert to Tensor
        # Result shape: (N, 15)
        self.X = torch.tensor(np.hstack([base_vals, log_vals]), dtype=torch.float64)
        self.Y = torch.tensor(df["sigma"].values, dtype=torch.float64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Direct tensor indexing is orders of magnitude faster than pandas iloc
        return self.X[idx], self.Y[idx]


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


def train_model(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    base_opt,
    device,
    epochs,
    val_loader,
):
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=optimizer.param_groups[0]["lr"],
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy="cos",
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
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation if val_loader is provided
        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_Y in val_loader:
                    x, y = batch_X.to(device), batch_Y.to(device)
                    output = model(x.float())
                    target = y.float().unsqueeze(1)
                    loss = criterion(output, target)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            print(
                f"Epoch {epoch + 1}: Train Loss {avg_train_loss:.4f} Val Loss {avg_val_loss:.4f}"
            )
        else:
            print(f"Epoch {epoch + 1}: Train Loss {avg_train_loss:.4f}")

    return model, train_losses, val_losses


def eval_model(model: nn.Module, test_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0  # Initialize total loss counter
    predictions = []  # List to store model predictions
    targets = []  # List to store targets
    losses = []  # List to store individual losses

    with torch.no_grad():  # Disable gradient computation for evaluation
        for X, Y in test_loader:  # Loop through batches
            X, Y = X.to(device), Y.to(device)  # Move data to device
            output = model(X.float())  # Get model predictions
            # Reshape target to match output dimensions
            target = Y.float().unsqueeze(1)
            loss = criterion(output, target)  # Calculate loss for batch
            total_loss += loss.item()  # Add batch loss to total
            losses.append(loss.item())

            # Store predictions and targets as flattened arrays
            predictions.extend(
                output.cpu().numpy().flatten()
            )  # Convert predictions to numpy array
            targets.extend(Y.cpu().numpy().flatten())  # Convert targets to numpy array

    avg_loss = total_loss / len(
        test_loader
    )  # Calculate average loss across all batches
    return avg_loss, np.array(predictions), np.array(targets), np.array(losses)


def main():
    data = pd.read_csv(dataset)

    train_data, test_data = train_test_split(data)
    num_workers = 6
    lr = 0.001
    weight_decay = 1e-6
    batch_size = 32
    epochs = 100
    dropout_rate = 0.0

    # Option 2: Train final model with train/val split
    # print("\nTraining final model with validation split...")
    train_sampler, val_sampler = train_val_split(train_data, val_size=0.2)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_sampler = RandomSampler(test_data)
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = module(dropout_rate=dropout_rate, dlayer=dlayer).to(device)
    criterion = nn.HuberLoss().to(device)
    base_opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    optimizer = Lookahead(base_opt, k=5, alpha=0.5)

    trained_model, tl, vl = train_model(
        model,
        train_loader,
        criterion,
        base_opt,
        optimizer,
        # base_opt,
        device,
        epochs,
        val_loader,
    )

    # Plot training and validation losses for final model
    plt.figure(figsize=(10, 6))
    plt.plot(tl, label="Training Loss")
    # if vl:  # Only plot validation loss if it exists
    #     plt.plot(vl, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Final Model: Training and Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    dataset_name = os.path.basename(dataset).split("/")[0].split(".")[0]
    # Build a safe filename from the dataset path (strip directories and extension)
    filename = f"final_model_loss_plot_{dataset_name}_with_{'dlayer' if dlayer else 'out_dlayer'}.png"
    # Save before show (so the file is written even if show blocks or closes the figure)
    plt.savefig(filename)
    # plt.show()
    plt.close()

    # Evaluation
    train_loss, train_pred, train_target, train_losses = eval_model(
        trained_model, train_loader, criterion, device
    )
    test_loss, test_pred, test_target, test_losses = eval_model(
        trained_model, test_loader, criterion, device
    )

    # save the trained model
    # torch.save(
    #     trained_model.state_dict(),
    #     f"trained_model_{dataset_name}_with_{'dlayer' if dlayer else 'out_dlayer'}.pth",
    # )
    print("\n" + "=" * 60)
    print("Final Model Evaluation")
    print("=" * 60)
    print("In-Sample Stats:")
    disp_stats(
        train_losses,
        train_pred,
        train_target,
        f"Train_{dataset_name}_with_{'dlayer' if dlayer else 'out_dlayer'}",
    )
    print("\nOut-of-Sample Stats:")
    disp_stats(
        test_losses,
        test_pred,
        test_target,
        f"Test_{dataset_name}_with_{'dlayer' if dlayer else 'out_dlayer'}",
    )


def disp_stats(losses, pred, true, name):
    mean = np.mean(losses)
    corr = np.corrcoef(pred, true)
    ninety_fifth = np.percentile(losses, 95)
    ninety_ninth = np.percentile(losses, 99)
    min_val = np.min(losses)
    max_val = np.max(losses)

    print(f"Mean: {mean:.12f}")
    print(f"Correlation: {corr[0, 1]:.12f}")
    print(f"95th Percentile: {ninety_fifth:.6f}")
    print(f"99th Percentile: {ninety_ninth:.6f}")
    print(f"Min: {min_val:.12f}")
    print(f"Max: {max_val:.12f}")

    df = pd.DataFrame(
        {
            "mean": [mean],
            "corr": [corr[0, 1]],
            "ninety_fifth": [ninety_fifth],
            "ninety_ninth": [ninety_ninth],
            "min": [min_val],
            "max": [max_val],
        }
    )

    df.to_csv(f"{name}_stats.csv", index=True)
    print(f"Stats saved to {name}_stats.csv")


if __name__ == "__main__":
    main()
