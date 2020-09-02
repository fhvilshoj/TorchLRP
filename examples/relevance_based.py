import torch
import matplotlib.pyplot as plt
from utils import get_mnist_model, prepare_mnist_model, get_mnist_data

batch_size = 16
model = get_mnist_model()
prepare_mnist_model(model)

train_loader, test_loader = get_mnist_data(batch_size=batch_size)

def normalize_and_merge_batch(a):
    a = a.permute(0, 2, 3, 1)
    # a -= a.view(batch_size, -1).min(1)[0].view(-1, 1, 1, 1)
    a /= torch.abs(a).view(batch_size, -1).max(1)[0].view(-1, 1, 1, 1) + 1e-6

    if a.min() < 0: a = (a+1) / 2
    else:           a = (a+1) / 2

    print(a.min(), a.max())

    a = a.view(4, 4, 28, 28, 1)
    a = a.permute(0, 2, 1, 3, 4)
    a = a.reshape(4*28, 4*28, 1)

    return a

x, y = next(iter(train_loader))
x.requires_grad_(True)

y_hat = model(x)
pred = y_hat.max(1)[1]
y_hat = y_hat[torch.arange(batch_size), pred]
y_hat = y_hat.sum()
y_hat.backward()
_y_hat = y_hat
a = normalize_and_merge_batch(x.grad)

def plot_attribution(a, ax_, title):

    cmap = plt.get_cmap('seismic')

    colors = cmap(a.view(-1))
    colors = colors.reshape(a.size(0), a.size(1), 4)
     
    ax_.imshow(colors)
    for i in range(4):
        for j in range(4):
            ax_.text(28+j*28, 28+i*28, pred[i*4+j].item(), horizontalalignment="right", verticalalignment="bottom", color="lime")
    ax_.set_title(title)

fig, ax = plt.subplots(3, 2)
plot_attribution(a, ax[0, 0], "Gradient (torch)")

def run_and_plot_rule(rule, ax_, title=None, postprocess=None): 
    x.grad = None
    y_hat = model.forward(x, explain=True, rule=rule)
    y_hat = y_hat[torch.arange(batch_size), y_hat.max(1)[1]]
    y_hat = y_hat.sum()
    y_hat.backward()
    assert torch.allclose(_y_hat, y_hat)

    attr = x.grad
    if postprocess: 
        with torch.no_grad(): 
            attr = postprocess(attr)

    attr = normalize_and_merge_batch(attr)
    if title is None: title = rule
    plot_attribution(attr, ax_, title)

run_and_plot_rule("epsilon", ax[0, 1])
run_and_plot_rule("gradient", ax[1, 0])
run_and_plot_rule("gradient", ax[1, 1], title="input $\\times$ gradient", postprocess = lambda attribution: attribution * x)
run_and_plot_rule("alpha1beta0", ax[2, 0])
run_and_plot_rule("alpha2beta1", ax[2, 1])
# 
fig.savefig("model trained 1 epoch")
plt.show()


