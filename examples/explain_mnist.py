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
        pickle.dump(patterns, f)

def load_patterns(file_name): 
    with open(file_name, 'rb') as f: p = pickle.load(f)
    return p

def heatmap(R,sx,sy, ax):
    R = R.reshape(4, 4, 28, 28)
    R = np.transpose(R, (0, 2, 1, 3))
    R = R.reshape(4*28, 4*28)

    b = 10*((np.abs(R)**3.0).mean()**(1.0/3))

    from matplotlib.colors import ListedColormap
    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:,0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)

    ax.axis('off')
    ax.imshow(R,cmap=my_cmap,vmin=-b,vmax=b,interpolation='nearest')

def prepare_batch_for_plotting(a, nrow=3, cmap='seismic'):
    # Normalize
    a /= torch.abs(a).view(a.size(0), -1).max(1)[0].view(-1, 1, 1, 1) + 1e-6

    # Make image grid
    grid = torchvision.utils.make_grid(a, nrow=nrow)
    grid = grid.permute(1, 2, 0)
    grid = grid.mean(-1)

    return grid

def plot_attribution(a, ax_, preds, title, cmap='seismic'):
    b = 1.
    # b = 10*((np.abs(a)**3.0).mean()**(1.0/3)) # L_3 norm? 
    ax_.imshow(a, cmap=cmap, vmin=-b, vmax=b)
    ax_.axis('off')
    cols = (a.shape[1] - 2) // 30
    rows = (a.shape[0] - 2) // 30
    for i in range(rows):
        for j in range(cols):
            ax_.text(28+j*30, 30+i*30, preds[i*cols+j].item(), horizontalalignment="right", verticalalignment="bottom", color="lime")
    ax_.set_title(title)

def main(args): 
    num_samples_plot = min(args.batch_size, 9)

    model = get_mnist_model()
    prepare_mnist_model(model, epochs=args.epochs, train_new=args.train_new)
    train_loader, test_loader = get_mnist_data(transform=torchvision.transforms.ToTensor(), batch_size=args.batch_size)

    # Sample batch
    x, y = next(iter(test_loader))
    x = x[:num_samples_plot]
    y = y[:num_samples_plot]
    x.requires_grad_(True)

    with torch.no_grad(): 
        y_hat = model(x)
        pred = y_hat.max(1)[1]

    def run_and_plot_rule(rule, ax_, title=None, postprocess=None, pattern=None, cmap='seismic'): 
        # Reset gradient
        x.grad = None

        # Forward pass and select argmax
        y_hat = model.forward(x, explain=True, rule=rule, pattern=pattern)
        y_hat = y_hat[torch.arange(x.shape[0]), y_hat.max(1)[1]]
        y_hat = y_hat.sum()

        # Backward pass
        y_hat.backward()
        attr = x.grad

        if postprocess:  # Used to compute input * gradient
            with torch.no_grad(): 
                attr = postprocess(attr)

        attr = prepare_batch_for_plotting(attr)

        if title is None: title = rule
        plot_attribution(attr, ax_, pred, title, cmap=cmap)

    # Patterns
    all_patterns_path = (base_path / 'examples' / 'pattern_all.pkl').as_posix()
    if not os.path.exists(all_patterns_path): 
        patterns_all = fit_patternnet(model, train_loader)
        store_patterns(all_patterns_path, patterns_all)
    else:
        patterns_all = load_patterns(all_patterns_path)

    pos_patterns_path = (base_path / 'examples' / 'pattern_pos.pkl').as_posix()
    if not os.path.exists(pos_patterns_path):
        patterns_pos = fit_patternnet_positive(model, train_loader)
        store_patterns(pos_patterns_path, patterns_pos)
    else:
        patterns_pos = load_patterns(pos_patterns_path)

    # Plotting
    fig, ax = plt.subplots(2, 5, figsize=(10, 5))

    with torch.no_grad(): 
        x_plot = prepare_batch_for_plotting(x*2.-1)
        plot_attribution(x_plot, ax[0, 0], pred, "Input", cmap='gray')

    # run_and_plot_rule("gradient", ax[0, 0])
    run_and_plot_rule("gradient", ax[1, 0], title="input $\\times$ gradient", postprocess = lambda attribution: attribution * x)

    run_and_plot_rule("epsilon", ax[0, 1])
    run_and_plot_rule("gamma+epsilon", ax[1, 1])

    run_and_plot_rule("alpha1beta0", ax[0, 2])
    run_and_plot_rule("alpha2beta1", ax[1, 2])

    run_and_plot_rule("patternattribution", ax[0, 3], pattern=list(patterns_all), title="PatternAttribution $S(x)$")
    run_and_plot_rule("patternattribution", ax[1, 3], pattern=list(patterns_pos), title="PatternAttribution $S(x)_{+-}$")

    run_and_plot_rule("patternnet", ax[0, 4], pattern=patterns_all, title="PatternNet $S(x)$", cmap='gray')
    run_and_plot_rule("patternnet", ax[1, 4], pattern=patterns_pos, title="PatternNet $S(x)_{+-}$", cmap='gray')

    fig.tight_layout()

    fig.savefig((base_path / 'examples'/ "Example_explanations.png").as_posix(), dpi=280)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("MNIST LRP Example")
    parser.add_argument('--batch_size', type=int, default=128)
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
