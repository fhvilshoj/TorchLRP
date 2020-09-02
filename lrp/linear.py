import torch
from .functional import linear

class Linear(torch.nn.Linear):
    def forward(self, input, explain=False):
        if not explain: return super(Linear, self).forward(input)
        return linear(input, self.weight, self.bias)
