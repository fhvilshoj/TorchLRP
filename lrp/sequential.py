import torch

from . import Linear, Conv2d
from .maxpool import MaxPool2d
from .functional.utils import normalize

def grad_decorator_fn(module):
    """
        Currently not used but can be used for debugging purposes.
    """
    def fn(x): 
        return normalize(x)
    return fn

avoid_normalization_on = ['relu', 'maxp']
def do_normalization(rule, module):
    if "pattern" not in rule.lower(): return False
    return not str(module)[:4].lower() in avoid_normalization_on

def is_kernel_layer(module):
    return isinstance(module, Conv2d) or isinstance(module, Linear)

def is_rule_specific_layer(module):
    return isinstance(module, MaxPool2d)

class Sequential(torch.nn.Sequential):
    def forward(self, input, explain=False, rule="epsilon", pattern=None):
        if not explain: return super(Sequential, self).forward(input)

        first = True

        # copy references for user to be able to reuse patterns
        if pattern is not None: pattern = list(pattern) 

        for module in self:
            if do_normalization(rule, module):
                input.register_hook(grad_decorator_fn(module))

            if is_kernel_layer(module): 
                P = None
                if pattern is not None: 
                    P = pattern.pop(0)
                input = module.forward(input, explain=True, rule=rule, pattern=P)

            elif is_rule_specific_layer(module):
                input = module.forward(input, explain=True, rule=rule)

            else: # Use gradient as default for remaining layer types
                input = module(input)
            first = False

        if do_normalization(rule, module): 
            input.register_hook(grad_decorator_fn(module))

        return input

