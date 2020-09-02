# Implementation of LRP for pytorch
A simple PyTorch implementation of the most basic Layer-Wise Relevance
Propagation rules for linear layers and convolutional layers.

The modules simply decorates `torch.nn.Sequential`, `torch.nn.Linear`, 
and `torch.nn.Conv2d` to be able to use `autograd` backprop algorithm
to compute explanations.

The code can be used as follows:

```python 
import torch
import lrp

model = Sequential(
    lrp.Conv2d(1, 32, 3, 1, 1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 2),
    torch.nn.Flatten(),
    lrp.Linear(14*14*32, 10)
)

y_hat = model.forward(x, explain=True, rule="alpha2beta1")
y_hat = y_hat[torch.arange(batch_size), y_hat.max(1)[1]] # Choose maximizing output neuron
y_hat = y_hat.sum()

# Backward pass (do explanation)
y_hat.backward()
explanation = x.grad
```

**Implemented rules:**
|Rule |Key | Note |
|:----|:---|:-----|
|$\epsilon$-rule| "epsilon" | Rule implemented, but $\epsilon$ fixed to $10^{-6}$ |
|$\alpha=1\quad \beta=0$ | "alpha1beta0" | |
|$\alpha=2\quad \beta=1$ | "alpha2beta1" | |

_Note:_ Biases are currently ignored in the $\alpha\beta$-rule implementation.

For a complete running example, which generates this plot: 
![Example Explanation](examples/Example_explanations.png)

Please see [examples/explain_mnist.py](examples/explain_mnistpy).




