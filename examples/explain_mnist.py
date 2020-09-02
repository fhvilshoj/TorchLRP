import torch
import matplotlib.pyplot as plt
from utils import get_mnist_model, prepare_mnist_model, get_mnist_data
import torchvision

def prepare_batch_for_plotting(a, nrow=4):
    # Normalize
    a /= torch.abs(a).view(batch_size, -1).max(1)[0].view(-1, 1, 1, 1) + 1e-6
    a = (a+1) / 2

    # Make image grid
    grid = torchvision.utils.make_grid(a, nrow=nrow)
    grid = grid.permute(1, 2, 0)
    grid = grid.mean(-1)

    # Make heatmap
    cmap = plt.get_cmap('seismic')
    colors = cmap(grid.reshape(-1))
    colors = colors.reshape(grid.size(0), grid.size(1), 4)
    return colors

def plot_attribution(a, ax_, preds, title):
    ax_.imshow(a)
    cols = (a.shape[1] - 2) // 30
    rows = (a.shape[0] - 2) // 30
    for i in range(rows):
        for j in range(cols):
            ax_.text(28+j*30, 30+i*30, preds[i*4+j].item(), horizontalalignment="right", verticalalignment="bottom", color="lime")
    ax_.set_title(title)

batch_size = 16
model = get_mnist_model()
prepare_mnist_model(model)
train_loader, test_loader = get_mnist_data(batch_size=batch_size)

# Sample batch
x, y = next(iter(test_loader))
x.requires_grad_(True)

with torch.no_grad(): 
    y_hat = model(x)
    pred = y_hat.max(1)[1]

fig, ax = plt.subplots(3, 2)
def run_and_plot_rule(rule, ax_, title=None, postprocess=None): 
    # Reset gradient
    x.grad = None

    # Forward pass and select argmax
    y_hat = model.forward(x, explain=True, rule=rule)
    y_hat = y_hat[torch.arange(batch_size), y_hat.max(1)[1]]
    y_hat = y_hat.sum()

    # Backward pass
    y_hat.backward()
    attr = x.grad

    if postprocess:  # Used to compute input * gradient
        with torch.no_grad(): 
            attr = postprocess(attr)

    attr = prepare_batch_for_plotting(attr)
    if title is None: title = rule
    plot_attribution(attr, ax_, pred, title)

run_and_plot_rule("gradient", ax[0, 0])
run_and_plot_rule("gradient", ax[0, 1], title="input $\\times$ gradient", postprocess = lambda attribution: attribution * x)
run_and_plot_rule("epsilon", ax[1, 0])
run_and_plot_rule("alpha1beta0", ax[2, 0])
run_and_plot_rule("alpha2beta1", ax[2, 1])

fig.savefig("Example_explanations.png")
plt.show()


