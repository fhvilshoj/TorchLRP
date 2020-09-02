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

linear = LinearEpsilon.apply
