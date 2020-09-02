import torch
from . import Linear, Conv2d

class Sequential(torch.nn.Sequential):
    def forward(self, input, explain=False, rule="epsilon"):
        if not explain: return super(Sequential, self).forward(input)
        for module in self:
            if isinstance(module, Conv2d) or isinstance(module, Linear): 
                input = module.forward(input, explain=True, rule=rule)
            else: 
                input = module(input)
        return input
