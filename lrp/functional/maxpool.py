import torch
import torch.nn.functional as F
from torch.autograd import Function

class MaxPooling2d(Function):
    @staticmethod
    def forward(ctx, input, kernel_size=2, stride=None, padding=0):
        ctx.kernel_size = kernel_size
        ctx.stride      = stride
        ctx.padding     = padding

        ctx.save_for_backward(input)

        return F.max_pool2d(input, kernel_size=kernel_size, stride=stride, padding=padding)

    @staticmethod
    def backward(ctx, relevance_output):
        input, *_ = ctx.saved_tensors
        Z = F.avg_pool2d(input, kernel_size=ctx.kernel_size, stride=ctx.stride, padding=ctx.padding) + 1e-10
        relevance_output = relevance_output / Z
        relevance_input  = torch.autograd.grad(Z, input, relevance_output)
        relevance_input = relevance_input * input

        return relevance_input, None, None, None 

maxpool2d = {
        "gradient":             F.max_pool2d,
        "epsilon":              F.max_pool2d,# MaxPooling2d.apply,
        "gamma":                F.max_pool2d,# MaxPooling2d.apply,
        "gamma+epsilon":        F.max_pool2d,# MaxPooling2d.apply,
        "alpha1beta0":          F.max_pool2d,# MaxPooling2d.apply,
        "alpha2beta1":          F.max_pool2d,# MaxPooling2d.apply,
        "patternattribution":   F.max_pool2d,# MaxPooling2d.apply,
        "patternnet":           F.max_pool2d,# MaxPooling2d.apply,
}

