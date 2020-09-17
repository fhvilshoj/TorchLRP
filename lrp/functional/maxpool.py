import torch
import torch.nn.functional as F
from torch.autograd import Function

class MaxPooling2d(Function):
    @staticmethod
    # kernel_size: Union[T, Tuple[T, ...]], stride: Optional[Union[T, Tuple[T, ...]]] = None, padding: Union[T, Tuple[T, ...]] = 0, dilation: Union[T, Tuple[T, ...]] = 1
    def forward(ctx, input, kernel_size=2, stride=None, padding=0):
        ctx.kernel_size = kernel_size
        ctx.stride      = stride
        ctx.padding     = padding

        ctx.save_for_backward(input)

        return F.max_pool2d(input, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

    @staticmethod
    def backward(ctx, relevance_output):
        input = ctx.saved_tensors
        Z = F.avg_pool2d(input, kernel_size=ctx.kernel_size, stride=ctx.stride, padding=ctx.padding) + 1e-10
        relevance_output = relevance_output / Z
        relevance_input  = torch.autograd.grad(Z, input, relevance_input)
        relevance_input = relevance_input * input

        return relevance_input, None, None, None 


maxpool2d = {
        "gradient":             F.max_pool2d,
        "epsilon":              MaxPooling2d.apply,
        "gamma":                MaxPooling2d.apply,
        "gamma+epsilon":        MaxPooling2d.apply,
        "alpha1beta0":          MaxPooling2d.apply,
        "alpha2beta1":          MaxPooling2d.apply,
        "patternattribution":   MaxPooling2d.apply,
        "patternnet":           MaxPooling2d.apply,
}

