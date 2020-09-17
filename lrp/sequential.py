import torch
from . import Linear, Conv2d
from .maxpool import MaxPool2d

class Sequential(torch.nn.Sequential):
    def forward(self, input, explain=False, rule="epsilon", pattern=None):
        if not explain: return super(Sequential, self).forward(input)
        for module in self:
            if isinstance(module, Conv2d) or isinstance(module, Linear): 
                if pattern is not None: 
                    input = module.forward(input, explain=True, rule=rule, pattern=pattern.pop(0))
                else:                   
                    input = module.forward(input, explain=True, rule=rule)

            elif  isinstance(module, MaxPool2d):
                input = module.forward(input, explain=True, rule=rule)

            else: 
                input = module(input)
        return input
