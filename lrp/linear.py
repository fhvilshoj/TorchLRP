import torch
from .functional import linear

class Linear(torch.nn.Linear):
    def forward(self, input, explain=False, rule="epsilon"):
        if not explain: return super(Linear, self).forward(input)
        return linear[rule](input, self.weight, self.bias)
