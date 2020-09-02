import torch
import matplotlib.pyplot as plt
from utils import get_mnist_model, prepare_mnist_model, get_mnist_data

batch_size = 16
model = get_mnist_model()
prepare_mnist_model(model)

train_loader, test_loader = get_mnist_data(batch_size=batch_size)

def normalize_and_merge_batch(a):
    a = a.permute(0, 2, 3, 1)
    a -= a.view(batch_size, -1).min(1)[0].view(-1, 1, 1, 1)
    a /= torch.abs(a).view(batch_size, -1).max(1)[0].view(-1, 1, 1, 1) + 1e-6

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

x.grad = None
y_hat = model.forward(x, explain=True)
y_hat = y_hat[torch.arange(batch_size), y_hat.max(1)[1]]
y_hat = y_hat.sum()
y_hat.backward()

assert torch.allclose(_y_hat, y_hat)

b = normalize_and_merge_batch(x.grad)

fig, ax = plt.subplots(2, 1)

ax[0].imshow(a, cmap='seismic')
for i in range(4):
    for j in range(4):
        ax[0].text(28+j*28, 28+i*28, pred[i*4+j].item(), horizontalalignment="right", verticalalignment="bottom", color="lime")


ax[1].imshow(b, cmap='seismic')
for i in range(4):
    for j in range(4):
        ax[1].text(28+j*28, 28+i*28, pred[i*4+j].item(), horizontalalignment="right", verticalalignment="bottom", color="lime")

ax[0].set_title("Gradient")
ax[1].set_title("LRP-$\\varepsilon$")

fig.savefig("model trained 1 epoch")
plt.show()


