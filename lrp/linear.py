import torch
from .functional import linear

class Linear(torch.nn.Linear):
    def forward(self, input, explain=False, rule="epsilon", **kwargs):
        if not explain: return super(Linear, self).forward(input)

        p = kwargs.get('pattern')
        if p is not None: return linear[rule](input, self.weight, self.bias, p)
        else: return linear[rule](input, self.weight, self.bias)
