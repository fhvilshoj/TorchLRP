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



# Other stuff
def safe_divide(a, b):
    return a / (b + (b == 0).float())

def normalize(x, range=(-1, 1)):
    absmax = torch.abs(x).max(axis=1, keepdims=True)[0]

    # print(x.view(-1)[:100])
    # print(safe_divide(x, absmax).view(-1)[:100])
    # assert False
    return safe_divide(x, absmax)
