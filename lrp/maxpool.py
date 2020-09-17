import torch
from .functional import maxpool2d

class MaxPool2d(torch.nn.MaxPool2d):
    def forward(self, input, explain=False, rule="epsilon", **kwargs):
        if not explain: return super(MaxPool2d, self).forward(input)
        return maxpool2d[rule](input, self.kernel_size, self.stride, self.padding)
