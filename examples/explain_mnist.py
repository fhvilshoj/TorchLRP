import os
import sys
import torch
import random
import pathlib
import argparse
import torchvision
import matplotlib.pyplot as plt

from utils import get_mnist_model, prepare_mnist_model, get_mnist_data
from utils import store_patterns, load_patterns

from visualization import heatmap_grid

# Append parent directory of this file to sys.path, 
# no matter where it is run from
base_path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.insert(0, base_path.as_posix())

import lrp
from lrp.patterns import fit_patternnet, fit_patternnet_positive # PatternNet patterns

def plot_attribution(a, ax_, preds, title, cmap='seismic', img_shape=28):
    ax_.imshow(a) 
    ax_.axis('off')

    cols = a.shape[1] // (img_shape+2)
    rows = a.shape[0] // (img_shape+2)
    for i in range(rows):
        for j in range(cols):
            ax_.text(28+j*30, 28+i*30, preds[i*cols+j].item(), horizontalalignment="right", verticalalignment="bottom", color="lime")
    ax_.set_title(title)

def main(args): 
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_samples_plot = min(args.batch_size, 9)

    model = get_mnist_model()
    # Either train new model or load pretrained weights
    prepare_mnist_model(args, model, epochs=args.epochs, train_new=args.train_new)
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

    def compute_and_plot_explanation(rule, ax_, title=None, postprocess=None, pattern=None, cmap='seismic'): 

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

        attr = heatmap_grid(attr, cmap_name=cmap)

        if title is None: title = rule
        plot_attribution(attr, ax_, pred, title, cmap=cmap)


    # # # # Patterns for PatternNet and PatternAttribution
    all_patterns_path = (base_path / 'examples' / 'patterns' / 'pattern_all.pkl').as_posix()
    if not os.path.exists(all_patterns_path):  # Either load of compute them
        patterns_all = fit_patternnet(model, train_loader, device=args.device)
        store_patterns(all_patterns_path, patterns_all)
    else:
        patterns_all = [torch.tensor(p, device=args.device, dtype=torch.float32) for p in load_patterns(all_patterns_path)]

    pos_patterns_path = (base_path / 'examples' / 'patterns' / 'pattern_pos.pkl').as_posix()
    if not os.path.exists(pos_patterns_path):
        patterns_pos = fit_patternnet_positive(model, train_loader, device=args.device)#, max_iter=1)
        store_patterns(pos_patterns_path, patterns_pos)
    else:
        patterns_pos = [torch.from_numpy(p).to(args.device) for p in load_patterns(pos_patterns_path)]


    # # # Plotting
    fig, ax = plt.subplots(2, 5, figsize=(10, 5))

    with torch.no_grad(): 
        x_plot = heatmap_grid(x*2-1, cmap_name="gray")
        plot_attribution(x_plot, ax[0, 0], pred, "Input")

    # compute_and_plot_explanation("gradient", ax[1, 0], title="gradient")
    compute_and_plot_explanation("gradient", ax[1, 0], title="input $\\times$ gradient", postprocess = lambda attribution: attribution * x)

    compute_and_plot_explanation("epsilon", ax[0, 1])
    compute_and_plot_explanation("gamma+epsilon", ax[1, 1])
# 
    compute_and_plot_explanation("alpha1beta0", ax[0, 2])
    compute_and_plot_explanation("alpha2beta1", ax[1, 2])
# 
    compute_and_plot_explanation("patternnet", ax[0, 3], pattern=patterns_all, title="PatternNet $S(x)$", cmap='gray')
    compute_and_plot_explanation("patternnet", ax[1, 3], pattern=patterns_pos, title="PatternNet $S(x)_{+-}$", cmap='gray')

    compute_and_plot_explanation("patternattribution", ax[0, 4], pattern=patterns_all, title="PatternAttribution $S(x)$")
    compute_and_plot_explanation("patternattribution", ax[1, 4], pattern=patterns_pos, title="PatternAttribution $S(x)_{+-}$")

    fig.tight_layout()

    fig.savefig((base_path / 'examples' / 'plots' / "mnist_explanations.png").as_posix(), dpi=280)
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
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    main(args)
