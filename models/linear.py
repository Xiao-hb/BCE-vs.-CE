""" Linear layer (alternate definition)
"""
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import math
from torch.nn import init
from copy import copy


class Linear(nn.Linear):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Wraps torch.nn.Linear to support AMP + torchscript usage by manually casting
    weight & bias to input.dtype to work around an issue w/ torch.addmm in this use case.
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if torch.jit.is_scripting():
            bias = self.bias.to(dtype=input.dtype) if self.bias is not None else None
            return F.linear(input, self.weight.to(dtype=input.dtype), bias=bias)
        else:
            return F.linear(input, self.weight, self.bias)


class inner_product(nn.Module):
    r"""
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    def __init__(self, in_features = 512, out_features = 1000, bias_mode = 'diverse',  bias_init_mode = 0, bias_init_mean = 0.):
    # def __init__(self, in_features = 512, out_features = 1000, bias_mode = False,  bias_init_mode = 0, bias_init_mean = 0.):
        super(inner_product, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mode = bias_mode
        self.bias_init_mode = bias_init_mode
        self.bias_init_mean = float(bias_init_mean)
        if self.bias_mode == False:
            self.bias = None
        elif self.bias_mode == 'unified':
            self.bias = Parameter(torch.tensor(bias_init_mode))
        elif self.bias_mode == 'diverse':
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            raise AttributeError('bias should be one of False, unified, or diverse')
        if self.bias is not None:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_mode == 'diverse':
            if self.bias_init_mode == 0:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)
            elif self.bias_init_mode == 1:
                value = range(self.out_features)
                self.bias = Parameter(torch.Tensor(value))
            elif self.bias_init_mode == 2:
                value = [x * 64. / self.out_features for x in range(self.out_features)]
                self.bias = Parameter(torch.Tensor(value))
            elif self.bias_init_mode == 3:
                value = [x * 64. / self.out_features for x in range(self.out_features)]
                self.bias = Parameter(torch.Tensor(value[::-1]))
            elif self.bias_init_mode == 4:
                value = [math.log(96 * x + 1) for x in range(self.out_features)]
                self.bias = Parameter(torch.Tensor(value))
            elif self.bias_init_mode == 5:
                value = [math.log(96 * x + 1) for x in range(self.out_features)]
                self.bias = Parameter(torch.Tensor(value[::-1]))
            elif self.bias_init_mode == 6:
                value = torch.ones(self.out_features)
                value[0:self.out_features//4] *= math.log(96 * self.out_features)
                value[self.out_features//4:self.out_features//2] *= 0.
                value[self.out_features//2:(3*self.out_features//4)] *= math.log(96 * self.out_features)
                value[(3*self.out_features//4):] *= 0.
                self.bias = Parameter(torch.Tensor(value))
            elif self.bias_init_mode == 7:
                value = torch.ones(self.out_features)
                value[0:self.out_features//4] *= 0.
                value[self.out_features//4:self.out_features//2] *= math.log(96 * self.out_features)
                value[self.out_features//2:(3*self.out_features//4)] *= 0.
                value[(3*self.out_features//4):] *= math.log(96 * self.out_features)
                self.bias = Parameter(torch.Tensor(value))
            else:
                raise NotImplementedError
        self.bias.data = self.bias.data + self.bias_init_mean

    def forward(self, input: torch.Tensor):
        output = F.linear(input, self.weight)
        output = output - self.bias if self.bias is not None else output
        return output
