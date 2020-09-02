import os
import torch
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn
import torchvision

import matplotlib.pyplot as plt

class LinearEpsilon(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        Z = F.linear(input, weight, bias)
        ctx.save_for_backward(input, weight, Z)
        return Z

    @staticmethod
    def backward(ctx, relevance_output):
        input, weight, Z = ctx.saved_tensors
        Z               += ((Z > 0).float()*2.-1) * 1e-6
        relevance_output = relevance_output / Z
        relevance_input  = F.linear(relevance_output, weight.t(), bias=None)
        relevance_input  = relevance_input * input
        return relevance_input, None, None

class Conv2DEpsilon(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        Z = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        ctx.save_for_backward(input, weight, Z)
        return Z
    
    @staticmethod
    def backward(ctx, relevance_output):
        input, weight, Z = ctx.saved_tensors
        Z               += ((Z > 0).float()*2-1) * 1e-6
        relevance_output = relevance_output / Z
        relevance_input  = F.conv_transpose2d(relevance_output, weight, None, padding=1)
        relevance_input  = relevance_input * input
        return relevance_input, *[None]*6

linear = LinearEpsilon.apply
conv2d = Conv2DEpsilon.apply

class MyLinear(torch.nn.Linear):
    def forward(self, input, explain=False):
        if not explain: return super(MyLinear, self).forward(input)
        return linear(input, self.weight, self.bias)

class Conv2D(torch.nn.Conv2d): 
    def _conv_forward_explain(self, input, weight):
        if self.padding_mode != 'zeros':
            return conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input, explain=False):
        if not explain: return super(Conv2D, self).forward(input)
        return self._conv_forward_explain(input, self.weight)

class Sequential(torch.nn.Sequential):
    def forward(self, input, explain=False):
        if not explain: return super(Sequential, self).forward(input)
        for module in self:
            if isinstance(module, Conv2D) or isinstance(module, MyLinear): 
                input = module.forward(input, explain=True)
            else: 
                input = module(input)
        return input

model = Sequential(
    Conv2D(1, 32, 3, 1, 1),
    nn.ReLU(),
    Conv2D(32, 32, 3, 1, 1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    MyLinear(14*14*32, 10)
)

transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

epochs = 1
batch_size = 16
model_path = "model.pth"

dataset1 = torchvision.datasets.MNIST('../data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()

if os.path.exists(model_path): 
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
else: 
    for e in range(epochs):
        for i, (x, y) in enumerate( train_loader):
            y_hat = model(x)
            loss  = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            acc = (y == y_hat.max(1)[1]).float().sum() / x.size(0)
            if i%10 == 0: 
                print("\r[%i/%i, %i/%i] loss: %.4f acc: %.4f" % (e, epochs, i, len(train_loader), loss.item(), acc.item()), end="", flush=True)
    torch.save(model.state_dict(), model_path)

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
        ax[0].text(28+j*28, 28+i*28, pred[i*4+j].item(), horizontalalignment="right", verticalalignment="bottom", color="magenta")


ax[1].imshow(b, cmap='seismic')
for i in range(4):
    for j in range(4):
        ax[1].text(28+j*28, 28+i*28, pred[i*4+j].item(), horizontalalignment="right", verticalalignment="bottom", color="magenta")

ax[0].set_title("Gradient")
ax[1].set_title("LRP-$\\varepsilon$")

fig.savefig("model trained 1 epoch")
plt.show()


