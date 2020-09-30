import torch
import torch.nn.functional as F
from .functional.utils import safe_divide

from tqdm import tqdm

__all__  = [
    'fit_patternnet',
    'fit_patternnet_positive',
]

"""
    
    This implementation is based on the implementation from
    https://github.com/albermax/innvestigate/blob/master/innvestigate/analyzer/pattern_based.py

"""
class RunningMean:
    def __init__(self, shape, device):
        self.value = torch.zeros(shape, device=device)
        self.count = 0

    def update(self, mean, cnt):
        total       = self.count + cnt
        new_factor  = safe_divide(cnt, total)
        old_factor  = 1 - new_factor
        self.value  = self.value * old_factor + mean * (new_factor)
        self.count += cnt


def _prod(module, x, y, mask):
    y_masked = y * mask

    x_copy = x * 1. #  [bs , h]

    if isinstance(module, torch.nn.Linear):
        W       = module.weight     # only for linear layers
        W_fn    = lambda w: w.t()   # only for linear layers

    elif isinstance(module, torch.nn.Conv2d): 
        p1, p2 = module.padding
        s1, s2 = module.stride
        k1, k2 = module.kernel_size

        x = F.pad(x, (p1, p1, p2, p2)).unfold(2, k1, s1).unfold(3, k2, s2)
        bs, c, h, w, *_ = x.shape # [bs, c, h, w, kh, kw]

        x = x.permute(0, 2, 3, 1, 4, 5).contiguous() 
        # [ bs, h ]
        x = x.view( -1, c*k1*k2, ) # [ bs*h*w, c*kh*kw ]

        def reshape_output(o):
            o = o.permute(0, 2, 3, 1).contiguous()
            return o.view(-1, module.out_channels) 

        y_masked    = reshape_output(y_masked)  # [ bs, h, w, c ] -> [ bs*h*w, out_c ]
        y           = reshape_output(y)         # [ bs, h, w, c ] -> [ bs*h*w, out_c ]
        mask        = reshape_output(mask)      # [ bs, h, w, c ] -> [ bs*h*w, out_c ]

        W       = module.weight.view(module.out_channels, -1) 
        def W_fn(w):
            w = w.view(W.t().shape)
            w = w.t().contiguous()
            return w.view(module.weight.shape)
    else:
        raise NotImplmentedError()

    cnt     = mask.sum(axis=0, keepdims=True)
    cnt_all = torch.ones_like(mask).sum(axis=0, keepdims=True)

    x_mean  = safe_divide(x.t() @ mask, cnt)
    xy_mean = safe_divide(x.t() @ y_masked, cnt)
    y_mean  = safe_divide(y.sum(0), cnt_all)

    return cnt, cnt_all, x_mean, y_mean, xy_mean, W, W_fn

def _fit_pattern(model, train_loader, max_iter, device, mask_fn = lambda y: torch.ones_like(y)):
    stats_x     = [] 
    stats_y     = []
    stats_xy    = []
    weights     = []
    cnt         = []
    cnt_all     = []

    first = True
    for b, (x, _) in enumerate(tqdm(train_loader)): 
        x = x.to(device)

        i = 0
        for m in model:
            y = m(x) # Note, this includes bias.
            if not (isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d)): 
                x = y.clone()
                continue
            
            mask = mask_fn(y).float().to(device)
            if isinstance(m, torch.nn.Conv2d): y_wo_bias = y - m.bias.view(-1, 1, 1) 
            else:                              y_wo_bias = y - m.bias

            cnt_, cnt_all_, x_, y_, xy_, w, w_fn = _prod(m, x, y_wo_bias, mask)

            if first:
                stats_x.append(RunningMean(x_.shape, device))
                stats_y.append(RunningMean(y_.shape, device)) # Use all y
                stats_xy.append(RunningMean(xy_.shape, device))
                weights.append((w, w_fn))

            stats_x[i].update(x_, cnt_)
            stats_y[i].update(y_.sum(0), cnt_all_)
            stats_xy[i].update(xy_, cnt_)

            x = y.clone()
            i += 1

            
        first = False

        if max_iter is not None and b+1 == max_iter: break

    def pattern(x_mean, y_mean, xy_mean, W2d):
        x_  = x_mean.value
        y_  = y_mean.value
        xy_ = xy_mean.value

        W, w_fn = W2d
        ExEy = x_ * y_
        cov_xy = xy_ - ExEy # [in, out]

        w_cov_xy = torch.diag(W @ cov_xy) # [out,]

        A = safe_divide(cov_xy, w_cov_xy[None, :])
        A = w_fn(A) # Reshape to original kernel size

        return A

    patterns = [pattern(*vars) for vars in zip(stats_x, stats_y, stats_xy, weights)]
    return patterns


@torch.no_grad()
def fit_patternnet(model, train_loader, max_iter=None, device='cpu'):
    return _fit_pattern(model, train_loader, max_iter, device)

@torch.no_grad()
def fit_patternnet_positive(model, train_loader, max_iter=None, device='cpu'):
    return _fit_pattern(model, train_loader, max_iter, device, lambda y: y >= 0)


