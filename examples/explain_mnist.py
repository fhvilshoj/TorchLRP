import os
import sys
import torch
import pickle
import random
import pathlib
import argparse
import torchvision

import numpy as np
import matplotlib.pyplot as plt

from utils import get_mnist_model, prepare_mnist_model, get_mnist_data

# Append parent directory of this file to sys.path, 
# no matter where it is run from
base_path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.insert(0, base_path.as_posix())
from lrp.patterns import fit_patternnet, fit_patternnet_positive # PatternNet patterns

def store_patterns(file_name, patterns):
    with open(file_name, 'wb') as f:
        pickle.dump([p.detach().cpu().numpy() for p in patterns], f)


def load_patterns(file_name): 
    with open(file_name, 'rb') as f: p = pickle.load(f)
    return p


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
    tmp = X.sum(axis=-1)

    tmp = project(tmp, output_range=(0, 255)).astype(int)
    tmp = cmap(tmp.flatten())[:, :3].T
    tmp = tmp.T

    shape = list(shape)
    shape[-1] = 3
    return tmp.reshape(shape).astype(np.float32)


def prepare_batch_for_plotting(a, nrow=3, fill_value=1., cmap_name="seismic"):
    # Compute colors
    a = heatmap(a, cmap_name=cmap_name) 
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


def plot_attribution(a, ax_, preds, title, cmap='seismic'):
    ax_.imshow(a) 
    ax_.axis('off')

    cols = (a.shape[1] - 2) // 30
    rows = (a.shape[0] - 2) // 30
    for i in range(rows):
        for j in range(cols):
            ax_.text(28+j*30, 28+i*30, preds[i*cols+j].item(), horizontalalignment="right", verticalalignment="bottom", color="lime")
    ax_.set_title(title)

def main(args): 
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_samples_plot = min(args.batch_size, 9)

    model = get_mnist_model()
    prepare_mnist_model(model, epochs=args.epochs, train_new=args.train_new)
    model = model.to(args.device)
    train_loader, test_loader = get_mnist_data(transform=torchvision.transforms.ToTensor(), batch_size=args.batch_size)

    # Sample batch from test_loader
    for x, y in test_loader: break
    x = x[:num_samples_plot].to(args.device)
    y = y[:num_samples_plot].to(args.device)
    x.requires_grad_(True)

    with torch.no_grad(): 
        y_hat = model(x)
        pred = y_hat.max(1)[1]

    def run_and_plot_rule(rule, ax_, title=None, postprocess=None, pattern=None, cmap='seismic'): 

        # # # # For the interested reader:
        # This is where the LRP magic happens.
        # Reset gradient
        x.grad = None

        # Forward pass with rule argument to "prepare" the explanation
        y_hat = model.forward(x, explain=True, rule=rule, pattern=pattern)
        # Choose argmax
        y_hat = y_hat[torch.arange(x.shape[0]), y_hat.max(1)[1]]
        # y_hat *= 0.5 * y_hat # to use value of y_hat as starting point
        y_hat = y_hat.sum()

        # Backward pass (compute explanation)
        y_hat.backward()
        attr = x.grad

        if postprocess:  # Used to compute input * gradient
            with torch.no_grad(): 
                attr = postprocess(attr)

        attr = prepare_batch_for_plotting(attr, cmap_name=cmap)

        if title is None: title = rule
        plot_attribution(attr, ax_, pred, title, cmap=cmap)


    # # # # Patterns for PatternNet and PatternAttribution
    all_patterns_path = (base_path / 'examples' / 'pattern_all.pkl').as_posix()
    if not os.path.exists(all_patterns_path):  # Either load of compute them
        patterns_all = fit_patternnet(model, train_loader, device=args.device)
        store_patterns(all_patterns_path, patterns_all)
    else:
        patterns_all = [p.to(args.device) for p in load_patterns(all_patterns_path)]

    pos_patterns_path = (base_path / 'examples' / 'pattern_pos.pkl').as_posix()
    if not os.path.exists(pos_patterns_path):
        patterns_pos = fit_patternnet_positive(model, train_loader)#, max_iter=1)
        store_patterns(pos_patterns_path, patterns_pos)
    else:
        patterns_pos = [p.to(args.device) for p in load_patterns(pos_patterns_path)]


    # # # Plotting
    fig, ax = plt.subplots(2, 5, figsize=(10, 5))

    with torch.no_grad(): 
        x_plot = prepare_batch_for_plotting(x*2-1, cmap_name="gray")
        plot_attribution(x_plot, ax[0, 0], pred, "Input")

    # run_and_plot_rule("gradient", ax[1, 0], title="gradient")
    run_and_plot_rule("gradient", ax[1, 0], title="input $\\times$ gradient", postprocess = lambda attribution: attribution * x)

    run_and_plot_rule("epsilon", ax[0, 1])
    run_and_plot_rule("gamma+epsilon", ax[1, 1])
# 
    run_and_plot_rule("alpha1beta0", ax[0, 2])
    run_and_plot_rule("alpha2beta1", ax[1, 2])
# 
    run_and_plot_rule("patternnet", ax[0, 3], pattern=patterns_all, title="PatternNet $S(x)$", cmap='gray')
    run_and_plot_rule("patternnet", ax[1, 3], pattern=patterns_pos, title="PatternNet $S(x)_{+-}$", cmap='gray')

    run_and_plot_rule("patternattribution", ax[0, 4], pattern=patterns_all, title="PatternAttribution $S(x)$")
    run_and_plot_rule("patternattribution", ax[1, 4], pattern=patterns_pos, title="PatternAttribution $S(x)_{+-}$")

    fig.tight_layout()

    fig.savefig((base_path / 'examples'/ "Example_explanations.png").as_posix(), dpi=280)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("MNIST LRP Example")
    parser.add_argument('--batch_size', type=int, default=3000)
    parser.add_argument('--train_new', action='store_true', help='Train new predictive model')
    parser.add_argument('--epochs', '-e', type=int, default=5)
    parser.add_argument('--seed', '-d', type=int)

    args = parser.parse_args()

    if args.seed is None: 
        args.seed = int(random.random() * 1e9)
        print("Setting seed: %i" % args.seed)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    main(args)
