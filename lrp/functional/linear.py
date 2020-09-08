import torch
import torch.nn.functional as F
from torch.autograd import Function

class LinearEpsilon(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        Z = F.linear(input, weight, bias)
        ctx.save_for_backward(input, weight, Z)
        return Z

    @staticmethod
    def backward(ctx, relevance_output):
        input, weight, Z = ctx.saved_tensors
        Z               += ((Z > 0).float()*2.-1) * 1e-6
        relevance_output = relevance_output / Z
        relevance_input  = F.linear(relevance_output, weight.t(), bias=None)
        relevance_input  = relevance_input * input
        return relevance_input, None, None

def _forward_alpha_beta(ctx, input, weight, bias):
    Z = F.linear(input, weight, bias)
    ctx.save_for_backward(input, weight, Z, bias)
    return Z

def _backward_alpha_beta(alpha, beta, ctx, relevance_output):
    """
        Inspired by https://github.com/albermax/innvestigate/blob/1ed38a377262236981090bb0989d2e1a6892a0b1/innvestigate/analyzer/relevance_based/relevance_rule.py#L270
    """
    input, weights, Z, bias = ctx.saved_tensors
    sel = weights > 0
    zeros = torch.zeros_like(weights)

    weights_pos       = torch.where(sel,  weights, zeros)
    weights_neg       = torch.where(~sel, weights, zeros)

    input_pos         = torch.where(input >  0, input, torch.zeros_like(input))
    input_neg         = torch.where(input <= 0, input, torch.zeros_like(input))

    def f(X1, X2, W1, W2): 

        Z1  = F.linear(X1, W1, bias=None) 
        Z2  = F.linear(X2, W2, bias=None)
        Z   = Z1 + Z2

        rel_out = relevance_output / (Z + (Z==0).float()* 1e-6)

        t1 = F.linear(rel_out, W1.t(), bias=None) 
        t2 = F.linear(rel_out, W2.t(), bias=None)

        r1  = t1 * X1
        r2  = t2 * X2

        return r1 + r2

    pos_rel = f(input_pos, input_neg, weights_pos, weights_neg)
    neg_rel = f(input_neg, input_pos, weights_pos, weights_neg)

    return pos_rel * alpha - neg_rel * beta, *[None]*2        

class LinearAlpha1Beta0(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        return _forward_alpha_beta(ctx, input, weight, bias)

    @staticmethod
    def backward(ctx, relevance_output):
        return _backward_alpha_beta(1., 0., ctx, relevance_output)

class LinearAlpha2Beta1(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        return _forward_alpha_beta(ctx, input, weight, bias)

    @staticmethod
    def backward(ctx, relevance_output):
        return _backward_alpha_beta(2., 1., ctx, relevance_output)


class LinearPatternAttribution(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, pattern=None):
        ctx.save_for_backward(input, weight, pattern, bias)
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, relevance_output):
        input, weight, P, bias = ctx.saved_tensors
        P = weight * P
        Z                = F.linear(input, P, bias)
        Z               += ((Z > 0).float()*2.-1) * 1e-6 # Safety tiny normalization to avoid dividing with zero
        relevance_output = relevance_output / Z

        relevance_input  = F.linear(relevance_output, P.t(), bias=None)
        relevance_input  = relevance_input * input
        return relevance_input, *[None] * 3


linear = {
        "gradient":             F.linear,
        "epsilon":              LinearEpsilon.apply,
        "alpha1beta0":          LinearAlpha1Beta0.apply,
        "alpha2beta1":          LinearAlpha2Beta1.apply,
        "patternattribution":   LinearPatternAttribution.apply,
}
