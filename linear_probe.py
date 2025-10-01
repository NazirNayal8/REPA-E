import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import h5py
import wandb
import random
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm

from gmm_fit import reduce_dimensionality
from sklearn.model_selection import train_test_split


class NormalDataset(Dataset):

    def __init__(self, data, labels):
        self.data = data

        new_labels = []
        for l in labels:
            new_labels.extend([int(l.decode("utf-8")[7:-1])])

        self.labels = torch.tensor(new_labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


parser = argparse.ArgumentParser(description="Linear Probe Training")
parser.add_argument(
    "--data-path",
    type=str,
    required=True,
    help="Path to the folder containing the data",
)
parser.add_argument(
    "--latents-name", type=str, required=True, help="Name of the latents dataset"
)
parser.add_argument(
    "--batch-size", type=int, default=256, help="Batch size for training"
)
parser.add_argument(
    "--n-iter", type=int, default=500, help="Number of training iterations"
)
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument(
    "--dim-reduction",
    type=str,
    choices=["PCA", "UMAP"],
    default="PCA",
    help="Dimensionality reduction method",
)
parser.add_argument(
    "--target-dim", type=int, default=100, help="Target dimensionality after reduction"
)

parser.add_argument(
    "--num-workers", type=int, default=4, help="Number of DataLoader workers"
)


def train(
    train_dataset,
    val_dataset,
    input_dim,
    num_classes,
    batch_size,
    n_iter,
    lr,
    num_workers=4,
):
    model = LinearProbe(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    train_losses, val_losses = [], []
    val_accuracies = []

    for epoch in tqdm(range(n_iter)):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            inputs = batch[0]
            labels = batch[1]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        avg_train_loss = total_loss / len(train_dataloader.dataset)

        model.eval()
        total_val_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in val_dataloader:
                inputs = batch[0]
                labels = batch[1]
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
        avg_val_loss = total_val_loss / len(val_dataloader.dataset)

        val_accuracy = correct / len(val_dataloader.dataset)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
            }
        )

        print(
            f"Epoch [{epoch+1}/{n_iter}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
        )

    return model, train_losses, val_losses, val_accuracies


if __name__ == "__main__":

    args = parser.parse_args()

    latents_path = os.path.join(args.data_path, args.latents_name + "_processed.h5")

    with h5py.File(latents_path, "r") as h5f:
        data = h5f["data"][:]
        labels = h5f["labels"][:]

    # random seeding for everything
    seed = 42
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    data = reduce_dimensionality(
        data, method=args.dim_reduction, n_components=args.target_dim
    )

    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data = (data - mean) / (std + 1e-8)

    # data = torch.from_numpy(data).float()

    train_data, val_data, train_labels, val_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )  #

    train_dataset = NormalDataset(train_data, train_labels)
    val_dataset = NormalDataset(val_data, val_labels)

    # Initialize wandb
    wandb.init(
        project="Linear-Probe",
        name=f"LP-{args.latents_name}-{args.dim_reduction}-{args.target_dim}",
        config={
            "batch_size": args.batch_size,
            "n_iter": args.n_iter,
            "lr": args.lr,
            "dim_reduction": args.dim_reduction,
            "target_dim": args.target_dim,
        },
    )

    model, train_losses, val_losses, val_accuracies = train(
        train_dataset,
        val_dataset,
        input_dim=args.target_dim,
        num_classes=len(np.unique(labels)),
        batch_size=args.batch_size,
        n_iter=args.n_iter,
        lr=args.lr,
        num_workers=args.num_workers,
    )

    # save the model
    model_path = os.path.join(args.data_path, f"linear_probe_{args.latents_name}.pth")
    torch.save(model.state_dict(), model_path)
