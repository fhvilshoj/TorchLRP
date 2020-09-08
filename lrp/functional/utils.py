import torch

# rhos
identity_fn    = lambda w, b: (w, b)

def gamma_fn(gamma): 
    def _gamma_fn(w, b):
        w = w + w * torch.max(torch.tensor(0.), w) * gamma
        if b is not None: b = b + b * torch.max(torch.tensor(0.), b) * gamma
        return w, b
    return _gamma_fn


# incrs
add_epsilon_fn = lambda e: lambda x:   x + ((x > 0).float()*2-1) * e

