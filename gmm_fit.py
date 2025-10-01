from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import wandb
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
from tqdm import trange
import argparse
import matplotlib.pyplot as plt
import wandb
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
from tqdm import trange
import argparse
import os
import numpy as np
from tqdm import tqdm
import h5py
import random
import torch.nn.functional as F
import os
import random
import numpy as np
import h5py
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from dataset import CustomH5Dataset



def process_image_avg(index, dataset):

    """
    Process a single image to extract and flatten latent data.
    """
    try:
        img, latent, label = dataset[index]

        mean, std = torch.chunk(latent, 2, dim=0)
        latent = std * torch.randn_like(std) + mean

        # average across spatial dimensions
        latent = latent.mean(dim=(-2, -1))  # Adjust based on latent shape

        return latent, label
    except Exception as e:
        print(f"Error processing index {index}: {e}")
        return None, None

def process_image(index, dataset):

    """
    Process a single image to extract and flatten latent data.
    """
    try:
        img, latent, label = dataset[index]

        mean, std = torch.chunk(latent, 2, dim=0)
        latent = std * torch.randn_like(std) + mean

        flattened_latent = latent.numpy().flatten()
        return flattened_latent, label
    except Exception as e:
        print(f"Error processing index {index}: {e}")
        return None, None


def load_and_flatten_latents(
    dataset,
    output_file,
    max_workers=8,
    batch_size=100,
    factor=1,
):
    """
    Load latent variables from a specified number of random label folders, flatten them, and save to an HDF5 file.
    Dynamically adjusts the dimensions of the latent variables based on the `factor`.

    Args:
        base_path (str): The root directory containing label folders with latent variable files.
        output_file (str): Path to the HDF5 file to save the data and labels.
        num_labels (int, optional): Number of label folders to randomly load. If None, loads all folders.
        samples_per_class (int, optional): Number of samples to randomly select from each class (label folder). If -1, loads all samples.
        max_workers (int): Number of threads to use for parallel processing.
        batch_size (int): Number of samples to write to HDF5 in a single batch.
        factor (float): Factor to determine the fraction of dimensions to retain. Positive values scale the full dimensions,
                        negative values retain only the corresponding absolute fraction (e.g., -0.5 keeps the first half).

    Returns:
        None
    """

    latent_dim = None  # Placeholder for determining latent dimensions dynamically

    buffer_data, buffer_labels = [], []

    # Prepare to write to HDF5
    with h5py.File(output_file, "w") as h5f:
        data_ds = None  # Will be initialized dynamically
        labels_ds = None

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_image, index, dataset): index
                for index in range(len(dataset))
            }

            for future in tqdm(
                as_completed(futures),
                desc="Processing Latents",
                leave=False,
            ):
                flattened_latent, label = future.result()

                if flattened_latent is not None:
                    if latent_dim is None:
                        # Determine original latent dimensions
                        original_dim = flattened_latent.shape[0]

                        # Adjust latent dimensions based on factor
                        if factor < 0:
                            latent_dim = int(original_dim * abs(factor))
                        else:
                            latent_dim = int(original_dim * factor)

                        # Initialize datasets
                        data_ds = h5f.create_dataset(
                            "data",
                            (0, latent_dim),
                            maxshape=(None, latent_dim),
                            dtype="float32",
                            chunks=True,
                        )
                        labels_ds = h5f.create_dataset(
                            "labels",
                            (0,),
                            maxshape=(None,),
                            dtype="S10",
                            chunks=True,
                        )

                    # Extract only the required portion of the latent variables
                    processed_latent = flattened_latent[
                        :latent_dim
                    ]  # Adjust for 1D array
                    buffer_data.append(processed_latent)
                    buffer_labels.append(label)

                    # If buffer reaches batch size, write to HDF5
                    if len(buffer_data) >= batch_size:
                        data_ds.resize(data_ds.shape[0] + len(buffer_data), axis=0)
                        labels_ds.resize(
                            labels_ds.shape[0] + len(buffer_labels), axis=0
                        )
                        data_ds[-len(buffer_data) :] = buffer_data
                        labels_ds[-len(buffer_labels) :] = buffer_labels

                        buffer_data, buffer_labels = [], []  # Clear the buffer

        # Write remaining data in buffer
        if buffer_data:
            data_ds.resize(data_ds.shape[0] + len(buffer_data), axis=0)
            labels_ds.resize(labels_ds.shape[0] + len(buffer_labels), axis=0)
            data_ds[-len(buffer_data) :] = buffer_data
            labels_ds[-len(buffer_labels) :] = buffer_labels


class GMM(nn.Module):
    def __init__(self, n_components, n_features):
        super(GMM, self).__init__()
        self.n_components = n_components
        self.n_features = n_features

        # Initialize the weights, means, and covariances
        self.weights = nn.Parameter(
            torch.ones(n_components) / n_components
        )  # Mixing coefficients
        self.means = nn.Parameter(
            torch.randn(n_components, n_features)
        )  # Means of Gaussians
        self.log_covariances = nn.Parameter(
            torch.zeros(n_components, n_features)
        )  # Log-diagonal covariances

    def forward(self, x):
        # Calculate the log-likelihood for each data point and component
        likelihoods = []
        for i in range(self.n_components):
            diag_cov = torch.exp(self.log_covariances[i])  # Ensure positive covariance
            dist = MultivariateNormal(self.means[i], torch.diag(diag_cov))
            likelihoods.append(dist.log_prob(x))  # Use log_prob directly
        likelihoods = torch.stack(
            likelihoods, dim=-1
        )  # Shape: [n_samples, n_components]

        # Weighted sum of log-likelihoods
        weighted_log_likelihoods = likelihoods + torch.log(
            self.weights + 1e-10
        )  # Avoid log(0)
        total_likelihood = torch.logsumexp(
            weighted_log_likelihoods, dim=-1
        )  # Log-sum-exp trick
        return total_likelihood.mean()  # Return average log-likelihood

    def bic(self, x):
        n_samples = x.size(0)
        n_params = self.n_components * (
            1 + self.n_features + self.n_features
        )  # Adjust for diagonal covariance
        log_likelihood = self.forward(x).item() * n_samples
        bic = n_params * np.log(n_samples) - 2 * log_likelihood
        return bic


def save_component_statistics(model, n_components, output_path):
    # Prepare to save the data in a CSV
    component_stats = []
    min_distance = float("inf")  # Initialize min_distance to infinity

    # Calculate min ||means[i] - means[j]|| for i != j
    for i in range(n_components):
        for j in range(
            i + 1, n_components
        ):  # Only compute for i < j to avoid redundant calculations
            distance = torch.norm(model.means[i] - model.means[j]).item()
            min_distance = min(min_distance, distance)

    # print(f"Min ||means[i] - means[j]||: {min_distance}")

    # Collect component statistics
    for i in range(n_components):
        mean_norm = torch.norm(model.means[i]).item()
        cov_norm = torch.norm(
            torch.exp(model.log_covariances[i])
        ).item()  # for diagonal covariance
        mu = model.means[i].detach().cpu().numpy()
        cov_matrix = torch.exp(model.log_covariances[i]).detach().cpu().numpy()

        # Calculate second moment E(x^2)
        E_x2 = (
            np.linalg.norm(mu) ** 2 + np.sum(cov_matrix)
            if cov_matrix.ndim == 1
            else np.linalg.norm(mu) ** 2 + np.trace(cov_matrix)
        )

        # Append the statistics, including min_distance for each component
        component_stats.append(
            [
                n_components,
                i + 1,  # Cluster index starting from 1
                mean_norm,
                cov_norm,
                E_x2,
                min_distance,  # Same min_distance for all rows of the same n_components
            ]
        )

    # Convert the list of statistics to a DataFrame
    df = pd.DataFrame(
        component_stats,
        columns=[
            "n_components",
            "cluster",
            "mean_norm",
            "cov_norm",
            "second_moment",
            "min_mean_distance",
        ],
    )

    # Check if the output file already exists
    if os.path.exists(output_path):
        # If the file exists, append to it (without writing the header)
        df.to_csv(output_path, mode="a", header=False, index=False)
        print(f"Statistics appended to {output_path}")
    else:
        # If the file doesn't exist, write a new file (with header)
        df.to_csv(output_path, index=False)
        print(f"Statistics saved to {output_path}")


def train_gmm(
    train_data,
    val_data,
    n_components,
    batch_size,
    n_iter,
    lr,
    device,
    args,
    num_workers=4,
):
    n_features = train_data.shape[1]
    model = GMM(n_components, n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dataset = TensorDataset(train_data)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_dataset = TensorDataset(val_data)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    train_losses, val_losses = [], []

    for local_epoch in trange(n_iter):
        model.train()
        batch_train_losses = []
        for batch_data in train_dataloader:
            batch = batch_data[0].to(device)
            optimizer.zero_grad()
            loss = -model(batch)  # Maximize log-likelihood
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss.item())

        epoch_train_loss = np.mean(batch_train_losses)
        train_losses.append(epoch_train_loss)

        model.eval()
        batch_val_losses = []
        with torch.no_grad():
            for batch_data in val_dataloader:
                batch = batch_data[0].to(device)
                loss = -model(batch)  # Maximize log-likelihood
                batch_val_losses.append(loss.item())

        epoch_val_loss = np.mean(batch_val_losses)
        val_losses.append(epoch_val_loss)

        wandb.log(
            {
                f"train_loss_n_components_{n_components}": epoch_train_loss,
                f"val_loss_n_components_{n_components}": epoch_val_loss,
                "epoch": local_epoch,
                "n_components": n_components,
            }
        )

    output_path = f"results/{args.latents_name}.csv"
    save_component_statistics(model, n_components, output_path)
    return model, train_losses, val_losses


def calculate_metric(train_data, val_data, args, device="cpu"):
    bics = []
    final_train_losses, final_val_losses = [], []
    components = []
    for n_components in args.components_list:
        wandb.log(
            {"current_n_components": n_components}
        )

        # Train GMM
        model, train_losses, val_losses = train_gmm(
            train_data,
            val_data,
            n_components,
            args.batch_size,
            args.n_iter,
            args.lr,
            device,
            args,
            num_workers=args.num_workers,
        )

        # Compute BIC
        bic = model.bic(train_data.to(device))
        bics.append(bic)

        # Log BIC to wandb
        wandb.log(
            {"BIC": bic, "n_components": n_components}
        )

        # Store the final loss (last epoch's loss) for this n_components
        final_train_losses.append(train_losses[-1])
        final_val_losses.append(val_losses[-1])
        components.append(n_components)

        log_final_loss_and_bic_plot(
            bics, final_train_losses, final_val_losses, components, args.latents_name
        )

    return bics, final_train_losses, final_val_losses


def log_final_loss_and_bic_plot(
    bics, final_train_losses, final_val_losses, components_list, name
):
    # Create a new figure
    plt.figure(figsize=(8, 6))

    # Plot Final Train Loss Curve
    plt.plot(
        components_list,
        final_train_losses,
        label="Final Train Loss Curve",
        marker="o",
        linestyle="-",
        color="blue",
    )

    # Plot Final Val Loss Curve
    plt.plot(
        components_list,
        final_val_losses,
        label="Final Val Loss Curve",
        marker="o",
        linestyle="-",
        color="green",
    )

    # Plot BIC Curve
    plt.plot(
        components_list,
        bics,
        label="BIC Curve",
        marker="o",
        linestyle="--",
        color="orange",
    )

    # Add labels, title, legend
    plt.xlabel("Number of Components (n_components)")
    plt.ylabel("Value")
    plt.title("Final Train/Val Loss and BIC Curves")
    plt.legend()
    plt.grid(True)

    # Save the figure to a file (optional, for debugging)
    plt.savefig(f"{name}-final_loss_bic_plot-{components_list[-1]}.png")

    # save the data as well
    np.savez(
        f"{name}-final_loss_bic_data-{components_list[-1]}.npz",
        components=components_list,
        final_train_losses=final_train_losses,
        final_val_losses=final_val_losses,
        bics=bics,
    )

    # Log the figure to wandb
    wandb.log({"Final_Train_Val_Loss_and_BIC_Plot": wandb.Image(plt)})

    # Close the plot to avoid memory issues
    plt.close()


def reduce_dimensionality(data, method, n_components, device="cpu"):
    """
    Reduce the dimensionality of data using PCA, UMAP, TSNE, or None.

    Args:
        data (numpy.ndarray or torch.Tensor): Input data of shape (n_samples, n_features).
        method (str): Dimensionality reduction method: 'None', 'PCA', 'UMAP', 'TSNE'.
        n_components (int): Number of dimensions to reduce to.
        device (str): Device to move the reduced data to ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Reduced dimensionality data as a PyTorch tensor.
    """
    if method == "PCA":
        print(f"Reducing dimensionality using PCA to {n_components} dimensions...")
        reducer = PCA(n_components=n_components)
    elif method == "UMAP":
        print(f"Reducing dimensionality using UMAP to {n_components} dimensions...")
        reducer = umap.UMAP(n_components=n_components)
    elif method == "None":
        print(f"No dimensionality reduction applied.")
        return torch.tensor(data, device=device)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")

    # Handle numpy and torch.Tensor input
    if isinstance(data, np.ndarray):
        reduced_data = torch.tensor(reducer.fit_transform(data), device=device)
    elif isinstance(data, torch.Tensor):
        reduced_data = torch.tensor(
            reducer.fit_transform(data.cpu().numpy()), device=device
        )
    else:
        raise ValueError("Data must be a numpy.ndarray or a torch.Tensor")

    print(f"Reduced data shape: {reduced_data.shape}")
    return reduced_data


# Add dimensionality reduction arguments to argparse
parser = argparse.ArgumentParser(description="GMM Training Parameters")
parser.add_argument(
    "--data-path", type=str, default="./imagenetdata/", help="Base path for the dataset file"
)
parser.add_argument(
    "--latents-name", type=str, help="name of the latents to be used"
)
parser.add_argument(
    "--batch-size", type=int, default=64, help="Batch size for training"
)
parser.add_argument("--use_gpu", type=int, default=0, help="GPU ID to use")
parser.add_argument(
    "--n-iter", type=int, default=2000, help="Number of training epochs"
)
parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")
parser.add_argument(
    "--n_label",
    type=int,
    default=0,
    help="Number of unique labels to select from the data; 0 means all",
)
parser.add_argument(
    "--samples-per-class",
    type=int,
    default=-1,
    help="Number of unique labels to select from the data; -1 mean all",
)
parser.add_argument(
    "--dim-reduction",
    type=str,
    default="PCA",
    choices=["None", "PCA", "UMAP"],
    help="Dimensionality reduction method",
)
parser.add_argument(
    "--target-dim", type=int, default=100, help="Target dimensionality after reduction"
)
parser.add_argument("--factor", type=float, default=1, help="Dimension adjust")
parser.add_argument(
    "--select_labels",
    type=str,
    nargs="+",
    help="Specific labels to use, e.g., --select_labels n01496331 n04273569",
)
parser.add_argument(
    "--components-list",
    type=int,
    nargs="+",
    default=[5, 10, 50, 100, 200, 300, 400, 500],
    help="List of components to evaluate, e.g., --components_list 5 10 50 100 200 300 400 500",
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=8,
    help="Number of workers for data loading",
)

args = parser.parse_args()

if __name__ == "__main__":
    # File path
    device = torch.device(
        f"cuda:{args.use_gpu}" if torch.cuda.is_available() else "cpu"
    )

    dataset = CustomH5Dataset(args.data_path, args.latents_name, samples_per_class=args.samples_per_class)
    
    output_file = os.path.join(args.data_path, args.latents_name + "_processed.h5")
    
    # prepares a h5 file with flattened latents and corresponding labels
    load_and_flatten_latents(
        dataset,
        output_file,
        max_workers=args.num_workers,
    )

    with h5py.File(output_file, "r") as h5f:
        data = h5f["data"][:]
        labels = h5f["labels"][:]

    # Apply dimensionality reduction
    data = reduce_dimensionality(
        data, method=args.dim_reduction, n_components=args.target_dim
    )
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data = (data - mean) / (std + 1e-8)
    # Convert data to PyTorch tensor
    data = torch.tensor(data, dtype=torch.float32)

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)  #

    # Initialize wandb
    wandb.init(
        project="GMM-Fitting",
        name=f"GMM-{args.latents_name}",
        config={
            "batch_size": args.batch_size,
            "n_iter": args.n_iter,
            "lr": args.lr,
            "n_label": args.n_label,
            "dim_reduction": args.dim_reduction,
            "target_dim": args.target_dim,
        },
    )

    # Train GMM and calculate BIC
    bics, final_train_losses, final_val_losses = calculate_metric(
        train_data, val_data, args, device
    )

    # # Log BIC and Final Loss curves as a single plot
    # log_final_loss_and_bic_plot(
    #     bics, final_train_losses, final_val_losses, args.components_list
    # )

    print("Training complete!")
