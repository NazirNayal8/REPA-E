import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
from sklearn.decomposition import PCA
from pysteps.utils import spectral
from sklearn.manifold import TSNE
from typing import (
    Sequence, 
    List, 
    Union, 
    Dict, 
    Iterable, 
    Tuple,
    Literal
)


Array = Union[np.ndarray, torch.Tensor]


def tsne_umap_seaborn(
        latents      : Dict[str, Array],
        labels       : Sequence      = None,        # optional vector of class-ids
        n_samples    : int           = 10_000,      # speed / memory cap
        pool_method  : Literal["mean", "flat"] = "mean",
        perplexity   : int           = 30,
        umap_nneigh  : int           = 15,
        random_state : int           = 0,
        fig_size     : Tuple[int,int] = (6, 5),
        palette      : str           = "tab10"
):
    """
    For each latent space:
    1) turns it into 2-D with t-SNE and UMAP
    2) draws the two projections side-by-side using Seaborn / Matplotlib.

    latents     – {name: tensor}  tensor shape (B,C,H,W) or (B,C)
    labels      – optional list/array of length B   (class, domain, …)
    pool_method – "mean": global average-pool per image
                  "flat": every spatial location becomes a sample (B*H*W)
    """

    def _prepare(z: Array) -> np.ndarray:
        z = torch.as_tensor(z).float()
        if z.ndim == 4:                       # (B,C,H,W)
            if pool_method == "mean":
                z = z.mean(dim=(2, 3))        # (B,C)
            else:                             # (B*H*W, C)
                z = z.permute(0, 2, 3, 1).reshape(-1, z.size(1))
        elif z.ndim != 2:
            raise ValueError(f"Unsupported shape {tuple(z.shape)}")
        return z.cpu().numpy()

    for name, z in latents.items():
        X = _prepare(z)

        # ── optional subsample ─────────────────────────────────────────
        if X.shape[0] > n_samples:
            idx = np.random.choice(X.shape[0], n_samples, replace=False)
            X   = X[idx]
            lab = None if labels is None else np.asarray(labels)[idx]
        else:
            lab = None if labels is None else np.asarray(labels)

        # ── t-SNE ──────────────────────────────────────────────────────
        tsne = TSNE(
            n_components = 2,
            perplexity   = min(perplexity, max(5, X.shape[0] // 3)),
            init         = "random",
            random_state = random_state,
        )
        X_tsne = tsne.fit_transform(X)

        # ── UMAP ───────────────────────────────────────────────────────
        reducer = umap.UMAP(
            n_components = 2,
            n_neighbors  = min(umap_nneigh, X.shape[0] - 1),
            random_state = random_state,
            init         = "random",
            metric       = "euclidean",
        )
        X_umap = reducer.fit_transform(X)

        # ── plotting ───────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(fig_size[0]*2, fig_size[1]),
                                 constrained_layout=True)
        titles = ("t-SNE", "UMAP")

        for emb, ax, title in zip((X_tsne, X_umap), axes, titles):
            if lab is None:
                ax.scatter(emb[:, 0], emb[:, 1], s=6, alpha=.6, color="blue")
            else:
                sns.scatterplot(x=emb[:, 0], y=emb[:, 1],
                                hue=lab,
                                palette=palette,
                                s=12, linewidth=0, alpha=.7, ax=ax)
                ax.legend(title="label", fontsize=8, loc="best")
            ax.set_title(f"{title} – {name}")
            ax.set_xticks([]); ax.set_yticks([])
        plt.show()


def cross_channel_corr(
    latents: Dict[str, Array],
    subsample: int = None,
    return_whiteness: bool = True,
    fig_size: Tuple[int, int] = (5, 4),
    cmap: str = "coolwarm",
):
    """
    For each model:  (1) compute the empirical cross-channel correlation
                     (2) plot it as a heat-map
                     (3) optionally return a 'whiteness index' =
                         mean|ρᵢⱼ|,  i≠j   (lower is better).

    Parameters
    ----------
    latents   : dict {name: tensor}         shape (B,C,H,W) or (N,C)
    subsample : int or None                 randomly pick max N samples
    return_whiteness : bool
    fig_size  : matplotlib figure size
    cmap      : heat-map colour-map

    Returns
    -------
    corrs     : dict {name: (C×C numpy array)}
    whiteness : dict {name: float}          only if return_whiteness=True
    """

    corrs, whiteness = {}, {}

    for name, z in latents.items():
        z = torch.as_tensor(z).float()

        # reshape to (C, N_samples)
        if z.dim() == 4:  # (B,C,H,W)
            z = z.permute(1, 0, 2, 3).reshape(z.size(1), -1)
        elif z.dim() == 2:  # (N,C)
            z = z.t()  # (C,N)
        else:
            raise ValueError(f"{name}: unsupported shape {tuple(z.shape)}")

        if subsample and z.size(1) > subsample:
            idx = torch.randperm(z.size(1))[:subsample]
            z = z[:, idx]

        # centre and normalise each channel
        z = z - z.mean(dim=1, keepdim=True)
        std = z.std(dim=1, keepdim=True) + 1e-8
        z = z / std

        # correlation = (1/N) Z Z^T   (since each channel now var=1)
        N = z.size(1)
        corr = (z @ z.t()) / N  # (C,C)
        corr = corr.cpu().numpy()

        corrs[name] = corr

        if return_whiteness:
            whiteness[name] = np.mean(np.abs(corr - np.eye(corr.shape[0])))

        # ---------- plot ---------------------------
        plt.figure(figsize=fig_size)
        sns.heatmap(
            corr, cmap=cmap, vmin=-1, vmax=1, square=True, cbar_kws=dict(label="ρ")
        )
        plt.title(f"Cross-channel correlation – {name}")
        plt.xlabel("Channel")
        plt.ylabel("Channel")
        plt.tight_layout()

    plt.show()
    return (corrs, whiteness) if return_whiteness else corrs


@torch.no_grad()
def eigenspectrum_dashboard(
    latents: Dict[str, Array],
    subsample: int = None,
    log_scale: bool = True,
    figsize: tuple = (8, 4),
):
    """
    Compute and plot the eigenvalue spectrum (sorted) of the empirical
    covariance matrix for one or more latent spaces.

    Parameters
    ----------
    latents   : dict {model_name: tensor/array}
                Shape (B, C, H, W) or (N, C).
    subsample : int or None.  If given, randomly choose at most this many
                samples to keep memory manageable.
    log_scale : put y-axis in log10 if True.
    figsize   : matplotlib figure size.

    Returns
    -------
    spectra   : dict {model_name: 1-D numpy array of eigenvalues}
    """
    spectra = {}
    for name, z in latents.items():
        z = torch.as_tensor(z).float()
        if z.dim() == 4:  # (B, C, H, W)
            B, C, H, W = z.shape
            z = z.permute(1, 0, 2, 3).reshape(C, -1)  # (C, N)
        elif z.dim() == 2:  # (N, C)
            z = z.t()  # (C, N)
            C, N = z.shape
        else:
            raise ValueError(f"{name}: expected 2-D or 4-D tensor, got {z.shape}")

        if subsample and z.size(1) > subsample:
            idx = torch.randperm(z.size(1))[:subsample]
            z = z[:, idx]

        # centre the data
        z = z - z.mean(dim=1, keepdim=True)

        # covariance = (1/N) * Z Zᵀ
        N = z.size(1)
        cov = (z @ z.t()) / N  # (C, C)

        # use symmetric eigendecomposition (faster, numerically stable)
        eigvals = torch.linalg.eigvalsh(cov).cpu().numpy()[::-1]  # descending
        spectra[name] = eigvals

    # --------------- plotting ---------------------
    plt.figure(figsize=figsize)
    for name, eigvals in spectra.items():
        y = np.log10(eigvals) if log_scale else eigvals
        plt.plot(range(1, len(eigvals) + 1), y, marker="o", label=name)

    plt.xlabel("Principal component index")
    plt.ylabel("log10 eigenvalue" if log_scale else "eigenvalue")
    plt.title("Eigenvalue spectrum of latent covariance")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return spectra


def latent_stats_dashboard(
    latents: Dict[str, Array],
    stats: Iterable[str] = ("mean", "var", "range"),
    cmap: str = "coolwarm",
    figsize: tuple = (14, 4),
) -> pd.DataFrame:
    """
    Compute and visualise per-channel statistics for several VAE latent spaces.

    Parameters
    ----------
    latents  : dict  {model_name: tensor/array}
               Each array shape must be (B, C, H, W) or (N, C).
    stats    : tuple of statistics to plot. Choose any of:
               {"mean", "var", "min", "max", "range"}.
    cmap     : colour-map passed to seaborn.heatmap.
    figsize  : base (W, H) in inches for each heatmap.

    Returns
    -------
    df       : tidy DataFrame with columns
               ['model', 'channel', 'mean', 'var', 'min', 'max', 'range'].
    """

    records = []

    for name, z in latents.items():
        # to torch tensor for uniform math  ────────────────────────────
        z = torch.as_tensor(z).float()
        # reshape so each row is one channel, all other dims flattened
        if z.dim() == 4:  # (B,C,H,W)
            z_flat = z.permute(1, 0, 2, 3).reshape(z.size(1), -1)
        elif z.dim() == 2:  # (N,C)  e.g. MLP VAE
            z_flat = z.t()
        else:
            raise ValueError(f"{name}: expected 2-D or 4-D tensor, got {z.shape}")

        # basic stats  ─────────────────────────────────────────────────
        means = z_flat.mean(1)
        vars_ = z_flat.var(1, unbiased=False)
        mins = z_flat.min(1).values
        maxs = z_flat.max(1).values
        ranges = maxs - mins

        for c in range(z_flat.size(0)):
            records.append(
                dict(
                    model=name,
                    channel=c,
                    mean=means[c].item(),
                    var=vars_[c].item(),
                    min=mins[c].item(),
                    max=maxs[c].item(),
                    range=ranges[c].item(),
                )
            )

    df = pd.DataFrame(records)

    # nice, consistent ordering
    df["channel"] = df["channel"].astype(int)
    df.sort_values(["model", "channel"], inplace=True)

    # ── Heat-maps ─────────────────────────────────────────────────────
    for stat in stats:
        pivot = df.pivot(index="model", columns="channel", values=stat)
        plt.figure(figsize=figsize)
        sns.heatmap(
            pivot,
            cmap=cmap,
            center=0.0 if stat in {"mean"} else None,
            cbar_kws=dict(label=stat),
        )
        plt.title(f"Per-channel {stat}")
        plt.xlabel("Channel")
        plt.ylabel("Model")
        plt.tight_layout()

    plt.show()
    return df


def pca_to_rgb(
    fmap: torch.Tensor,
    n_components: int = 3,
    per_channel_norm: bool = True,
    out_uint8: bool = False,
    show: bool = True,
    title: str | None = None,
    ax=None,
):
    """
    Project the first `n_components` principal components of a feature map
    to RGB and optionally plot the result.

    Parameters
    ----------
    fmap : torch.Tensor
        Shape (C, H, W) or (1, C, H, W). Must have C >= n_components.
    n_components : int
        Number of principal components -> RGB channels (default: 3).
    per_channel_norm : bool
        If True, min-max normalise each channel separately to [0,1].
        If False, use global min/max across all three channels.
    out_uint8 : bool
        Return uint8 image in [0,255] instead of float32 in [0,1].
    show : bool
        If True, display the image with matplotlib.
    title : str | None
        Optional title for the plot.
    ax : matplotlib axis or None
        Axis to draw on (created if None).

    Returns
    -------
    np.ndarray
        RGB image of shape (H, W, 3).
    """
    # 1. Sanity checks --------------------------------------------------------
    if fmap.dim() == 4 and fmap.size(0) == 1:
        fmap = fmap.squeeze(0)
    assert fmap.dim() == 3, "Expected tensor of shape (C,H,W)"
    C, H, W = fmap.shape
    assert C >= n_components, "Need at least n_components channels"

    # 2. Move to CPU & reshape to (N, C) where N = H*W ------------------------
    x = fmap.detach().cpu().float()  # (C,H,W)
    flat = x.permute(1, 2, 0).reshape(-1, C)  # (N,C)

    # 3. Fit PCA --------------------------------------------------------------
    pcs = PCA(n_components=n_components, svd_solver="auto").fit_transform(flat.numpy())
    pcs = pcs.reshape(H, W, n_components)  # (H,W,3)

    # 4. Normalise each channel to [0,1] --------------------------------------
    if per_channel_norm:
        mins = pcs.min(axis=(0, 1), keepdims=True)
        maxs = pcs.max(axis=(0, 1), keepdims=True)
    else:
        mins = pcs.min()
        maxs = pcs.max()
    rgb = (pcs - mins) / (maxs - mins + 1e-8)
    rgb = np.clip(rgb, 0, 1)

    # 5. Optional conversion to uint8 -----------------------------------------
    if out_uint8:
        rgb_out = (rgb * 255).round().astype(np.uint8)
    else:
        rgb_out = rgb.astype(np.float32)

    # 6. Visualise ------------------------------------------------------------
    if show:
        if ax is None:
            _, ax = plt.subplots()
        ax.imshow(rgb_out)
        ax.axis("off")
        if title is not None:
            ax.set_title(title)
        plt.show()

    return rgb_out


def get_rapsd(data):
    rapsd_img, frequencies_img = spectral.rapsd(
        data, fft_method=np.fft, return_freq=True
    )

    return rapsd_img, frequencies_img


def _fmap_pca_rgb(
    fmap: torch.Tensor, n_components: int = 3, per_channel_norm: bool = True
) -> np.ndarray:
    """
    Convert one feature-map (C,H,W) → RGB (H,W,3) via PCA.
    """
    if fmap.dim() == 4 and fmap.size(0) == 1:
        fmap = fmap.squeeze(0)
    assert fmap.dim() == 3, "expect (C,H,W)"
    C, H, W = fmap.shape
    assert C >= n_components, "need ≥ 3 channels"

    flat = fmap.detach().cpu().float().permute(1, 2, 0).reshape(-1, C)
    pcs = PCA(n_components=n_components).fit_transform(flat.numpy())
    pcs = pcs.reshape(H, W, n_components)

    if per_channel_norm:
        mins = pcs.min(axis=(0, 1), keepdims=True)
        maxs = pcs.max(axis=(0, 1), keepdims=True)
    else:
        mins, maxs = pcs.min(), pcs.max()

    rgb = (pcs - mins) / (maxs - mins + 1e-8)
    return np.clip(rgb, 0, 1).astype(np.float32)  # (H,W,3) ∈ [0,1]


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Full visualiser
# ──────────────────────────────────────────────────────────────────────────────
def visualise_with_input_pca_rgb(
    image: torch.Tensor,
    feature_maps: Sequence[torch.Tensor],
    fmap_titles: Sequence[str] | None = None,
    image_title: str = "Input",
    per_channel_norm: bool = True,
    figsize_per_panel: float = 4.0,
    tight: bool = True,
):
    """
    Display the original image plus a row of PCA-RGB projections.

    Parameters
    ----------
    image           : (3,H,W) or (H,W,3) or (1,3,H,W) tensor – **not** transformed.
    feature_maps    : list/tuple of (C,H,W) or (1,C,H,W) tensors.
    fmap_titles     : optional list of subplot titles for the feature-maps.
    image_title     : title for the leftmost (raw) image panel.
    per_channel_norm: normalise each PC channel separately (default: True).
    figsize_per_panel : horizontal inches per subplot.
    tight           : run `plt.tight_layout()` for compact spacing.
    """
    # ── prep input image ────────────────────────────────────────────────────
    if image.dim() == 4 and image.size(0) == 1:
        image = image.squeeze(0)
    if image.dim() == 3 and image.shape[0] == 3:  # (3,H,W) → (H,W,3)
        image_disp = image.detach().cpu().permute(1, 2, 0).float().clamp(0, 1).numpy()
    elif image.dim() == 3 and image.shape[2] == 3:  # already (H,W,3)
        image_disp = image.detach().cpu().float().clamp(0, 1).numpy()
    else:
        raise ValueError("image must be (3,H,W) or (H,W,3)")

    # ── convert all feature-maps ────────────────────────────────────────────
    if fmap_titles is None:
        fmap_titles = [""] * len(feature_maps)
    assert len(fmap_titles) == len(feature_maps)

    rgb_maps: List[np.ndarray] = [
        _fmap_pca_rgb(fm, per_channel_norm=per_channel_norm) for fm in feature_maps
    ]

    # ── build plots ─────────────────────────────────────────────────────────
    n_panels = 1 + len(rgb_maps)
    height = max(img.shape[0] for img in [image_disp] + rgb_maps)
    width = max(img.shape[1] for img in [image_disp] + rgb_maps)
    figsize = (figsize_per_panel * n_panels, figsize_per_panel * height / width)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False, frameon=False)
    axes = axes[0]  # unpack 1-D row

    # left: raw image
    axes[0].imshow(image_disp)
    axes[0].axis("off")
    axes[0].set_title(image_title)

    # right: each feature-map
    for ax, rgb, ttl in zip(axes[1:], rgb_maps, fmap_titles):
        ax.imshow(rgb)
        ax.axis("off")
        ax.set_title(ttl)

    if tight:
        plt.tight_layout()
    plt.show()
    return fig, axes
