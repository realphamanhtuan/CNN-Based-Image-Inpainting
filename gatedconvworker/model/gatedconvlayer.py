from functools import partial
import torch
from torch import nn
import torch.nn.functional as F

def padding_function(image, kernel_size, stride, dilation_rate):
    h = image.shape[2]
    w = image.shape[3]
    nH = ((h + stride - 1) // stride - 1) * stride + (kernel_size - 1) * dilation_rate + 1
    nW = ((w + stride - 1) // stride - 1) * stride + (kernel_size - 1) * dilation_rate + 1
    top_bottom = nH - h
    left_right = nW - w

    if (top_bottom > 0):
        bottom = top_bottom // 2
        top = top_bottom - bottom
    else: top = bottom = 0

    if (left_right > 0):
        right = left_right // 2
        left = left_right - right
    else: left = right = 0
    ret = torch.nn.ZeroPad2d((left, right, top, bottom))(image)
    return ret

class Gen_Conv(nn.Module):
    def __init__(self, input_cnum, output_cnum, kernel_size = 3, stride = 1, dilation_rate = 1, activation=nn.ELU()):
        super().__init__()

        self.output_cnum = output_cnum
        self.activation = activation
        self.conv = nn.Conv2d(input_cnum, output_cnum, kernel_size=kernel_size, stride=stride, padding="valid", dilation=dilation_rate) #same
        torch.nn.init.zeros_(self.conv.bias)
        torch.nn.init.kaiming_uniform_(self.conv.weight, nonlinearity="relu", mode="fan_out")

        self.padding_function = partial(padding_function, kernel_size=kernel_size, stride=stride, dilation_rate=dilation_rate)

    def forward(self, x):
        x = self.conv(self.padding_function(x))

        if self.output_cnum == 3 or self.activation is None:
            return x
        #print(x.shape)
        x, y = torch.split(x, self.output_cnum // 2, 1)
        x = self.activation(x)
        y = torch.sigmoid(y)
        x = x * y
        return x

class Gen_DeConv(nn.Module):
    def __init__(self, input_cnum, output_cnum):
        super().__init__()
        self.conv = Gen_Conv(input_cnum, output_cnum, 3, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest', recompute_scale_factor=False) 
        x = self.conv(x)
        return x

class Conv2D_Spectral_Norm(nn.Conv2d): 
    def __init__(self, cnum_in, cnum_out, kernel_size, stride, padding=0, n_iter=1, eps=1e-12, bias=True):
        super().__init__(cnum_in, cnum_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.register_buffer("weight_u", torch.empty(self.weight.size(0), 1))
        nn.init.trunc_normal_(self.weight_u)
        self.eps = eps

    def forward(self, x):

        size = self.weight.size()
        weight_orig = self.weight.view(size[0], -1).detach()

        v = F.normalize(weight_orig.t() @ self.weight_u, 2, 0, self.eps)
        self.weight_u = F.normalize((weight_orig @ v), 2, 0, self.eps)

        sigma = self.weight_u.t() @ weight_orig @ v
        self.weight.data.div_(sigma)

        x = super().forward(x)
        return x


class Dis_Conv(nn.Module):
    def __init__(self, input_cnum, output_cnum, kernel_size, stride):
        super().__init__()
        #self.conv_sn = torch.nn.utils.spectral_norm(nn.Conv2d(input_cnum, output_cnum, kernel_size, stride, 0))
        self.conv_sn = Conv2D_Spectral_Norm(input_cnum, output_cnum, kernel_size, stride)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = padding_function(x, self.kernel_size, self.stride, 1)
        x = self.conv_sn(x)
        x = self.activation(x)
        return x
