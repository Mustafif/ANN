import datetime
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
dataset = "Duan2_100K/dataset_duan.csv"
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps:0"
    if torch.backends.mps.is_available()
    else "cpu"
)


class SimDataset(Dataset):
    def __init__(self, dataframe):
        df = dataframe.copy().reset_index(drop=True)

        # clean dataset, options > 0.5
        df = df[df["V"] > 0.5]

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

        base_vals = df[base_cols].values.astype(np.float64)

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

        self.X = torch.tensor(np.hstack([base_vals, log_vals]), dtype=torch.float64)
        self.Y = torch.tensor(df["sigma"].values, dtype=torch.float64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def train_test_split(data, test_size=0.3, random_state=42):
    train_data, test_data = sklearn.model_selection.train_test_split(
        data, test_size=test_size, random_state=random_state, shuffle=True
    )
    return SimDataset(train_data), SimDataset(test_data)


def train_val_split(dataset, val_size=0.2, random_state=42):
    indices = list(range(len(dataset)))
    train_indices, val_indices = sklearn.model_selection.train_test_split(
        indices, test_size=val_size, random_state=random_state
    )
    return SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices)


def train_model(
    model, train_loader, criterion, optimizer, base_opt, device, epochs, val_loader
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
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

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


def eval_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    losses = []

    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            output = model(X.float())
            target = Y.float().unsqueeze(1)
            loss = criterion(output, target)
            total_loss += loss.item()
            losses.append(loss.item())
            predictions.extend(output.cpu().numpy().flatten())
            targets.extend(Y.cpu().numpy().flatten())

    return (
        total_loss / len(test_loader),
        np.array(predictions),
        np.array(targets),
        np.array(losses),
    )


def compute_stats(losses, pred, true):
    corr = np.corrcoef(pred, true)[0, 1]
    return {
        "Mean Loss": np.mean(losses),
        "Correlation": corr,
        "95th Percentile": np.percentile(losses, 95),
        "99th Percentile": np.percentile(losses, 99),
        "Min Loss": np.min(losses),
        "Max Loss": np.max(losses),
    }


def run_variant(
    dlayer,
    train_data,
    test_data,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    device,
    epochs,
    lr,
    weight_decay,
    out_dir,
):
    label = "dlayer" if dlayer else "no_dlayer"
    print(f"\n{'=' * 60}")
    print(f"Training: {label}")
    print(f"{'=' * 60}")

    model = module(dropout_rate=0.0, dlayer=dlayer).to(device)
    base_opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    optimizer = Lookahead(base_opt, k=5, alpha=0.5)

    trained_model, tl, vl = train_model(
        model, train_loader, criterion, optimizer, base_opt, device, epochs, val_loader
    )

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(tl, label="Training Loss")
    if vl:
        plt.plot(vl, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss Curve — {label}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"loss_curve_{label}.png"))
    plt.close()

    # Eval
    _, train_pred, train_target, train_losses = eval_model(
        trained_model, train_loader, criterion, device
    )
    _, test_pred, test_target, test_losses = eval_model(
        trained_model, test_loader, criterion, device
    )

    train_stats = compute_stats(train_losses, train_pred, train_target)
    test_stats = compute_stats(test_losses, test_pred, test_target)

    return label, train_stats, test_stats


def main():
    data = pd.read_csv(dataset)

    # Hyperparams
    lr = 0.001
    weight_decay = 1e-6
    batch_size = 32
    epochs = 200
    num_workers = 6

    # Output folder
    import os

    # dataset is a path like "path/to/folder/file.csv"
    dataset_dir = os.path.dirname(dataset)  # "path/to/folder"
    dataset_name = os.path.basename(dataset_dir)  # "folder"

    out_dir = f"report_{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)

    # Data split (shared across both variants for fair comparison)
    train_data, test_data = train_test_split(data)
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
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        sampler=RandomSampler(test_data),
        num_workers=num_workers,
        pin_memory=True,
    )

    criterion = nn.HuberLoss().to(device)

    # Run both variants
    rows = []
    for dlayer in [False, True]:
        label, train_stats, test_stats = run_variant(
            dlayer,
            train_data,
            test_data,
            train_loader,
            val_loader,
            test_loader,
            criterion,
            device,
            epochs,
            lr,
            weight_decay,
            out_dir,
        )
        rows.append({"Variant": label, "Split": "In-Sample", **train_stats})
        rows.append({"Variant": label, "Split": "Out-of-Sample", **test_stats})

    # Build and save combined report CSV
    report_df = pd.DataFrame(rows)
    # Reorder so Variant + Split come first
    col_order = [
        "Variant",
        "Split",
        "Mean Loss",
        "Correlation",
        "95th Percentile",
        "99th Percentile",
        "Min Loss",
        "Max Loss",
    ]
    report_df = report_df[col_order]

    csv_path = os.path.join(out_dir, "report.csv")
    report_df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 60}")
    print("Combined Report")
    print("=" * 60)
    print(report_df.to_string(index=False))
    print(f"\nReport saved to: {csv_path}")
    print(f"Plots saved to:  {out_dir}/")


if __name__ == "__main__":
    main()
