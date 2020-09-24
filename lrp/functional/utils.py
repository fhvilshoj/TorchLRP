import torch

# # # rhos
identity_fn    = lambda w, b: (w, b)


def gamma_fn(gamma): 
    def _gamma_fn(w, b):
        w = w + w * torch.max(torch.tensor(0., device=w.device), w) * gamma
        if b is not None: b = b + b * torch.max(torch.tensor(0., device=b.device), b) * gamma
        return w, b
    return _gamma_fn


# # # incrs
add_epsilon_fn = lambda e: lambda x:   x + ((x > 0).float()*2-1) * e


# # # Other stuff
def safe_divide(a, b):
    return a / (b + (b == 0).float())

def normalize(x):
    n_dim = len(x.shape)

    # This is what they do in `innvestigate`. Have no idea why?
    # https://github.com/albermax/innvestigate/blob/1ed38a377262236981090bb0989d2e1a6892a0b1/innvestigate/layers.py#L321
    if n_dim == 2: return x
    
    abs = torch.abs(x.view(x.shape[0], -1))
    absmax = torch.max(abs, axis=1)[0].view(x.shape[0], 1)
    for i in range(2, n_dim): absmax = absmax.unsqueeze(-1)

    x = safe_divide(x, absmax)
    x = x.clamp(-1, 1)

    return x
