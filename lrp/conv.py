import torch
import torch.nn.functional as F
from .functional import conv2d

class Conv2d(torch.nn.Conv2d): 
    def _conv_forward_explain(self, input, weight, conv2d_fn):
        if self.padding_mode != 'zeros':
            return conv2d_fn(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return conv2d_fn(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input, explain=False, rule="epsilon"):
        if not explain: return super(Conv2d, self).forward(input)
        return self._conv_forward_explain(input, self.weight, conv2d[rule])
