import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pysteps.utils import spectral
from typing import Sequence, List

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
    x = fmap.detach().cpu().float()          # (C,H,W)
    flat = x.permute(1, 2, 0).reshape(-1, C) # (N,C)

    # 3. Fit PCA --------------------------------------------------------------
    pcs = PCA(n_components=n_components, svd_solver="auto").fit_transform(flat.numpy())
    pcs = pcs.reshape(H, W, n_components)    # (H,W,3)

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
    rapsd_img, frequencies_img = spectral.rapsd(data, fft_method=np.fft, return_freq=True)

    return rapsd_img, frequencies_img

def _fmap_pca_rgb(
        fmap: torch.Tensor,
        n_components: int = 3,
        per_channel_norm: bool = True
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
    pcs  = PCA(n_components=n_components).fit_transform(flat.numpy())
    pcs  = pcs.reshape(H, W, n_components)

    if per_channel_norm:
        mins = pcs.min(axis=(0, 1), keepdims=True)
        maxs = pcs.max(axis=(0, 1), keepdims=True)
    else:
        mins, maxs = pcs.min(), pcs.max()

    rgb = (pcs - mins) / (maxs - mins + 1e-8)
    return np.clip(rgb, 0, 1).astype(np.float32)       # (H,W,3) ∈ [0,1]


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
        tight: bool = True):
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
    if image.dim() == 3 and image.shape[0] == 3:            # (3,H,W) → (H,W,3)
        image_disp = image.detach().cpu().permute(1, 2, 0).float().clamp(0, 1).numpy()
    elif image.dim() == 3 and image.shape[2] == 3:          # already (H,W,3)
        image_disp = image.detach().cpu().float().clamp(0, 1).numpy()
    else:
        raise ValueError("image must be (3,H,W) or (H,W,3)")

    # ── convert all feature-maps ────────────────────────────────────────────
    if fmap_titles is None:
        fmap_titles = [""] * len(feature_maps)
    assert len(fmap_titles) == len(feature_maps)

    rgb_maps: List[np.ndarray] = [
        _fmap_pca_rgb(fm, per_channel_norm=per_channel_norm)
        for fm in feature_maps
    ]

    # ── build plots ─────────────────────────────────────────────────────────
    n_panels = 1 + len(rgb_maps)
    height   = max(img.shape[0] for img in [image_disp] + rgb_maps)
    width    = max(img.shape[1] for img in [image_disp] + rgb_maps)
    figsize  = (figsize_per_panel * n_panels, figsize_per_panel * height / width)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False, frameon=False)
    axes = axes[0]                       # unpack 1-D row

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