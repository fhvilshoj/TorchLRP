import torch.nn.functional as F
from torch.autograd import Function

class Conv2DEpsilon(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        Z = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        ctx.save_for_backward(input, weight, Z)
        return Z
    
    @staticmethod
    def backward(ctx, relevance_output):
        input, weight, Z = ctx.saved_tensors
        Z               += ((Z > 0).float()*2-1) * 1e-6
        relevance_output = relevance_output / Z
        relevance_input  = F.conv_transpose2d(relevance_output, weight, None, padding=1)
        relevance_input  = relevance_input * input
        return relevance_input, *[None]*6

conv2d = Conv2DEpsilon.apply

