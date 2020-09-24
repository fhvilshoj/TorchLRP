import torch
import torch.nn.functional as F
from torch.autograd import Function

from .utils import identity_fn, gamma_fn, add_epsilon_fn, normalize

def _forward_rho(rho, incr, ctx, input, weight, bias, stride, padding, dilation, groups):
        ctx.save_for_backward(input, weight, bias)
        ctx.rho = rho
        ctx.incr = incr
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        Z = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        return Z

def _backward_rho(ctx, relevance_output):
    input, weight, bias    = ctx.saved_tensors

    weight, bias     = ctx.rho(weight, bias)
    Z                = ctx.incr(F.conv2d(input, weight, bias, ctx.stride, ctx.padding, ctx.dilation, ctx.groups))

    relevance_output = relevance_output / Z
    relevance_input  = F.conv_transpose2d(relevance_output, weight, None, padding=1)
    relevance_input  = relevance_input * input

    return relevance_input, None, None, None, None, None, None, 



class Conv2DEpsilon(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, **kwargs):
        return _forward_rho(identity_fn, add_epsilon_fn(1e-1), ctx, input, weight, bias, stride, padding, dilation, groups)
    
    @staticmethod
    def backward(ctx, relevance_output):
        return _backward_rho(ctx, relevance_output)

class Conv2DGamma(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, **kwargs):
        return _forward_rho(gamma_fn(0.1), add_epsilon_fn(1e-10), ctx, input, weight, bias, stride, padding, dilation, groups)
    
    @staticmethod
    def backward(ctx, relevance_output):
        return _backward_rho(ctx, relevance_output)

class Conv2DGammaEpsilon(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, **kwargs):
        return _forward_rho(gamma_fn(0.1), add_epsilon_fn(1e-1), ctx, input, weight, bias, stride, padding, dilation, groups)
    
    @staticmethod
    def backward(ctx, relevance_output):
        return _backward_rho(ctx, relevance_output)

def _conv_alpha_beta_forward(ctx, input, weight, bias, stride, padding, dilation, groups, **kwargs): 
    Z = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    ctx.save_for_backward(input, weight, Z,  bias)
    return Z

def _conv_alpha_beta_backward(alpha, beta, ctx, relevance_output):
        input, weights, Z, bias = ctx.saved_tensors
        sel = weights > 0
        zeros = torch.zeros_like(weights)

        weights_pos       = torch.where(sel,  weights, zeros)
        weights_neg       = torch.where(~sel, weights, zeros)

        input_pos         = torch.where(input >  0, input, torch.zeros_like(input))
        input_neg         = torch.where(input <= 0, input, torch.zeros_like(input))

        def f(X1, X2, W1, W2): 

            Z1  = F.conv2d(X1, W1, bias=None, stride=1, padding=1) 
            Z2  = F.conv2d(X2, W2, bias=None, stride=1, padding=1)
            Z   = Z1 + Z2

            rel_out = relevance_output / (Z + (Z==0).float()* 1e-6)

            t1 = F.conv_transpose2d(rel_out, W1, bias=None, padding=1) 
            t2 = F.conv_transpose2d(rel_out, W2, bias=None, padding=1)

            r1  = t1 * X1
            r2  = t2 * X2

            return r1 + r2
        pos_rel = f(input_pos, input_neg, weights_pos, weights_neg)
        neg_rel = f(input_neg, input_pos, weights_pos, weights_neg)

        return pos_rel * alpha - neg_rel * beta, None, None, None, None, None, None


class Conv2DAlpha1Beta0(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, **kwargs):
        return _conv_alpha_beta_forward(ctx, input, weight, bias, stride, padding, dilation, groups, **kwargs)
    
    @staticmethod
    def backward(ctx, relevance_output):
        return _conv_alpha_beta_backward(1., 0., ctx, relevance_output)


class Conv2DAlpha2Beta1(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, **kwargs):
        return _conv_alpha_beta_forward(ctx, input, weight, bias, stride, padding, dilation, groups, **kwargs)
    
    @staticmethod
    def backward(ctx, relevance_output):
        return _conv_alpha_beta_backward(2., 1., ctx, relevance_output)


def _pattern_forward(attribution, ctx, input, weight, bias, stride, padding, dilation, groups, pattern):
    Z = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    ctx.save_for_backward(input, weight, pattern)

    ctx.stride = stride
    ctx.padding = padding
    ctx.attribution = attribution
    return Z

def _pattern_backward(ctx, relevance_output):
    input, weight, P = ctx.saved_tensors

    if ctx.attribution: P = P * weight # PatternAttribution
    relevance_input  = F.conv_transpose2d(relevance_output, P, padding=ctx.padding, stride=ctx.stride)

    return relevance_input, None, None, None, None, None, None, None

class Conv2DPatternAttribution(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, pattern=None):
        return _pattern_forward(True, ctx, input, weight, bias, stride, padding, dilation, groups, pattern)

    @staticmethod
    def backward(ctx, relevance_output):
        return _pattern_backward(ctx, relevance_output)

class Conv2DPatternNet(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, pattern=None):
        return _pattern_forward(False, ctx, input, weight, bias, stride, padding, dilation, groups, pattern)

    @staticmethod
    def backward(ctx, relevance_output):
        return _pattern_backward(ctx, relevance_output)

conv2d = {
        "gradient":             F.conv2d,
        "epsilon":              Conv2DEpsilon.apply,
        "gamma":                Conv2DGamma.apply,
        "gamma+epsilon":        Conv2DGammaEpsilon.apply,
        "alpha1beta0":          Conv2DAlpha1Beta0.apply,
        "alpha2beta1":          Conv2DAlpha2Beta1.apply,
        "patternattribution":   Conv2DPatternAttribution.apply,
        "patternnet":           Conv2DPatternNet.apply,
}

