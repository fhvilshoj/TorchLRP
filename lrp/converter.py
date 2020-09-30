import torch
from .conv       import Conv2d 
from .linear     import Linear
from .sequential import Sequential

conversion_table = { 
        'Linear': Linear,
        'Conv2d': Conv2d
    }

# # # # # Convert torch.models.vggxx to lrp model
def convert_vgg(module, modules=None):
    # First time
    if modules is None: 
        modules = []
        for m in module.children():
            convert_vgg(m, modules=modules)

            # Vgg model has a flatten, which is not represented as a module
            # so this loop doesn't pick it up.
            # This is a hack to make things work.
            if isinstance(m, torch.nn.AdaptiveAvgPool2d): 
                modules.append(torch.nn.Flatten())

        sequential = Sequential(*modules)
        return sequential

    # Recursion
    if isinstance(module, torch.nn.Sequential): 
        for m in module.children():
            convert_vgg(m, modules=modules)

    elif isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
        class_name = module.__class__.__name__
        lrp_module = conversion_table[class_name].from_torch(module)
        modules.append(lrp_module)
    # maxpool is handled with gradient for the moment

    elif isinstance(module, torch.nn.ReLU): 
        # avoid inplace operations. They might ruin PatternNet pattern
        # computations
        modules.append(torch.nn.ReLU())
    else:
        modules.append(module)

