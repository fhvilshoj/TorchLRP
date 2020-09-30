import os
import sys
import torch
import pickle
from torch.nn import Sequential, Conv2d, Linear

import pathlib
import argparse
import torchvision
from torchvision import datasets, transforms as T
import configparser

import numpy as np
import matplotlib.pyplot as plt

# Append parent directory of this file to sys.path, 
# no matter where it is run from
base_path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.insert(0, base_path.as_posix())

import lrp
from lrp.patterns import fit_patternnet, fit_patternnet_positive # PatternNet patterns
from utils import store_patterns, load_patterns
from visualization import project, clip_quantile, heatmap_grid, grid

torch.manual_seed(1337)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# # # # # ImageNet Data
config = configparser.ConfigParser()
config.read((base_path / 'config.ini').as_posix())
sys.path.append(config['DEFAULT']['ImageNetDir'])
from torch_imagenet import ImageNetDataset

# Normalization as expected by pytorch vgg models
# https://pytorch.org/docs/stable/torchvision/models.html
_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view((1, 3, 1, 1))
_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view((1, 3, 1, 1))

def unnormalize(x):
    return x * _std + _mean

transform = T.Compose([
    T.Resize(256), 
    T.CenterCrop(224), 
    T.ToTensor(),
    T.Normalize( mean= _mean.flatten(),
                 std = _std.flatten()    ),
])

dataset = ImageNetDataset(transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=True)
# # # # # End ImageNet Data

# # # # # VGG model
vgg_num = int(sys.argv[1]) if len(sys.argv) > 1 else 16 # Default to vgg16

vgg = getattr(torchvision.models, "vgg%i"%vgg_num)(pretrained=True).to(device)
# vgg = torchvision.models.vgg16(pretrained=True).to(device)
vgg.eval()

print("Loaded vgg-%i" % vgg_num)

lrp_vgg = lrp.convert_vgg(vgg).to(device)
# # # # #

# Check that the vgg and lrp_vgg models does the same thing
for x, y in train_loader: break
x = x.to(device)
x.requires_grad_(True)

y_hat = vgg(x)
y_hat_lrp = lrp_vgg.forward(x)

assert torch.allclose(y_hat, y_hat_lrp, atol=1e-4, rtol=1e-4), "\n\n%s\n%s\n%s" % (str(y_hat.view(-1)[:10]), str(y_hat_lrp.view(-1)[:10]), str((torch.abs(y_hat - y_hat_lrp)).max()))
print("Done testing")
# # # # #

# # # # # Patterns for PatternNet and PatternAttribution
patterns_path = (base_path / 'examples' / 'patterns' / ('vgg%i_pattern_pos.pkl' % vgg_num)).as_posix()
if not os.path.exists(patterns_path):
    patterns = fit_patternnet_positive(lrp_vgg, train_loader, device=device)
    store_patterns(patterns_path, patterns)
else:
    patterns = [torch.tensor(p).to(device) for p in load_patterns(patterns_path)]

print("Loaded patterns")

# # # # # Plotting 
def compute_and_plot_explanation(rule, ax_, patterns=None, plt_fn=heatmap_grid): 
    # Forward pass
    y_hat_lrp = lrp_vgg.forward(x, explain=True, rule=rule, pattern=patterns)

    # Choose argmax
    y_hat_lrp = y_hat_lrp[torch.arange(x.shape[0]), y_hat_lrp.max(1)[1]]
    y_hat_lrp = y_hat_lrp.sum()

    # Backward pass (compute explanation)
    y_hat_lrp.backward()
    attr = x.grad

    # Plot
    attr = plt_fn(attr)
    ax_.imshow(attr)
    ax_.set_title(rule)
    ax_.axis('off')

# PatternNet is typically handled a bit different, when visualized.
def signal_fn(X):
    if X.shape[1] in [1, 3]: X = X.permute(0, 2, 3, 1).detach().cpu().numpy()
    X = clip_quantile(X)
    X = project(X)
    X = grid(X)
    return X

explanations = [
        # rule                  Pattern     plt_fn          Fig. pos
        ('alpha1beta0',         None,       heatmap_grid,   (1, 0)), 
        ('epsilon',             None,       heatmap_grid,   (0, 1)), 
        ('gamma+epsilon',       None,       heatmap_grid,   (1, 1)), 
        ('patternnet',          patterns,   signal_fn,      (0, 2)),
        ('patternattribution',  patterns,   heatmap_grid,   (1, 2)),
    ]

fig, ax = plt.subplots(2, 3, figsize=(12, 8))
print("Plotting")

# Plot inputs
input_to_plot = unnormalize(x).permute(0, 2, 3, 1).contiguous().detach().cpu().numpy()
input_to_plot = grid(input_to_plot, 3, 1.)
ax[0, 0].imshow(input_to_plot)
ax[0, 0].set_title("Input")
ax[0, 0].axis('off')

# Plot explanations
for i, (rule, pattern, fn, (p, q) ) in enumerate(explanations): 
    compute_and_plot_explanation(rule, ax[p, q], patterns=pattern, plt_fn=fn)

fig.tight_layout()
fig.savefig((base_path / 'examples' / 'plots' / ("vgg%i_explanations.png" % vgg_num)).as_posix(), dpi=280)
plt.show()



