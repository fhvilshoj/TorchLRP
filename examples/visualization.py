import numpy as np
import torch
import matplotlib.pyplot as plt

def project(X, output_range=(0, 1)):
    absmax   = np.abs(X).max(axis=tuple(range(1, len(X.shape))), keepdims=True)
    X       /= absmax + (absmax == 0).astype(float)
    X        = (X+1) / 2. # range [0, 1]
    X        = output_range[0] + X * (output_range[1] - output_range[0]) # range [x, y]
    return X


def heatmap(X, cmap_name="seismic"):
    cmap = plt.cm.get_cmap(cmap_name)

    if X.shape[1] in [1, 3]: X = X.permute(0, 2, 3, 1).detach().cpu().numpy()
    if isinstance(X, torch.Tensor): X = X.detach().cpu().numpy()

    shape = X.shape
    tmp = X.sum(axis=-1) # Reduce channel axis

    tmp = project(tmp, output_range=(0, 255)).astype(int)
    tmp = cmap(tmp.flatten())[:, :3].T
    tmp = tmp.T

    shape = list(shape)
    shape[-1] = 3
    return tmp.reshape(shape).astype(np.float32)


def clip_quantile(X, quantile=1):
    """Clip the values of X into the given quantile."""
    if isinstance(X, torch.Tensor): X = X.detach().cpu().numpy()
    if not isinstance(quantile, (list, tuple)):
        quantile = (quantile, 100-quantile)

    low = np.percentile(X, quantile[0])
    high = np.percentile(X, quantile[1])
    X[X < low] = low
    X[X > high] = high

    return X

def grid(a, nrow=3, fill_value=1.):
    bs, h, w, c = a.shape

    # Reshape to grid
    rows = bs // nrow + int(bs % nrow != 0)
    missing = (nrow - bs % nrow) % nrow
    if missing > 0: # Fill empty spaces in the plot
        a = np.concatenate([a, np.ones((missing, h, w, c))*fill_value], axis=0)

    # Border around images
    a = np.pad(a, ((0, 0), (1, 1), (1, 1), (0, 0)), 'constant', constant_values=0.5)
    a = a.reshape(rows, nrow, h+2, w+2, c)
    a = np.transpose(a, (0, 2, 1, 3, 4))
    a = a.reshape( rows * (h+2), nrow * (w+2), c)
    return a

def heatmap_grid(a, nrow=3, fill_value=1., cmap_name="seismic", heatmap_fn=heatmap):
    # Compute colors
    a = heatmap_fn(a, cmap_name=cmap_name) 
    return grid(a, nrow, fill_value)

