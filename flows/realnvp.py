import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import _resnet
from collections import OrderedDict
from functools import reduce
import operator as op


class CouplingLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, mask):
        super(CouplingLayer, self).__init__()
        self.in_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels
        self.hidden_channels = hidden_channels
        self.mask = mask
        factor = self._infer_mask_type()
        self.conv = _resnet(
            factor * in_channels + factor - 1,
            factor * hidden_channels,
            factor * in_channels,
        )
        self.tanh_scale = nn.Parameter(torch.ones(1))  # scale parameter for the tanh output
        self.bn1 = nn.BatchNorm2d(factor * in_channels // 2, affine=False)  # pre-coupling
        self.bn2 = nn.BatchNorm2d(factor * in_channels // 2, affine=False)  # post-coupling

    def _infer_mask_type(self):
        if self.mask.ndim == 1 or self.mask.shape[2:] == torch.Size([1, 1]):
            self.mask_type = "channel-wise"
            return 1  # half the channels if channel-wise
        else:
            self.mask_type = "spatial"
            return 2

    def forward(self, x):
        with_logdet = False
        if isinstance(x, tuple):
            x, logdet = x
            with_logdet = logdet is not None
        if self.mask_type == "channel-wise":
            xx = self.extend(x)
            s, t = self.conv(xx).chunk(2, dim=1)
            s = torch.tanh(s)*self.tanh_scale
            x[:, ~self.mask, :, :] = self.bn2(torch.exp(s) * x[:, ~self.mask, :, :] + t)
            factor = 1
        else:
            self.mask = self.mask.to(x.device)
            xx = self.extend(x)
            s, t = self.conv(xx).chunk(2, dim=1)
            s = torch.tanh(s)*self.tanh_scale
            s = (1 - self.mask) * s
            x = self.mask * x + (1-self.mask) * self.bn2((torch.exp(s) * x + t))
            factor = 2
        if with_logdet:
            hxw = x.shape[2]*x.shape[3]
            logdet += torch.sum(s, dim=(1, 2, 3))
            logdet += -(hxw//factor) * torch.sum(torch.log(self.bn2.running_var + self.bn2.eps) / 2)
        return x, logdet

    def backward(self, x):
        if self.mask_type == "channel-wise":
            out = torch.zeros_like(x)
            xx = self.extend(x)
            s, t = self.conv(xx).chunk(2, dim=1)
            s = torch.tanh(s) * self.tanh_scale
            out[:, self.mask, :, :] = x[:, self.mask, :, :]
            out[:, ~self.mask, :, :] = (self.debatchnorm(x[:, ~self.mask, :, :]) - t) * torch.exp(-s)
        else:
            xx = self.extend(x)
            s, t = self.conv(xx).chunk(2, dim=1)
            s = torch.tanh(s) * self.tanh_scale
            out = self.debatchnorm((1-self.mask) * x)
            out = self.mask * x + (1 - self.mask) * (out - t) * torch.exp(-s)
        return out

    def debatchnorm(self, x):
        if self.bn2.affine:
            x = (x - self.bn2.bias) / self.bn2.weight
        x = x * torch.sqrt(
            self.bn2.running_var + self.bn2.eps
        )[None, :, None, None] + self.bn2.running_mean[None, :, None, None]
        return x

    def extend(self, x):
        if self.mask_type == "channel-wise":
            xx = self.bn1(x[:, self.mask, :, :])
            xx = F.relu(torch.cat([xx, -xx], dim=1))  # split positive and negative pixels into separate channels
        else:
            n_batch = x.shape[0]
            xx = 2 * self.bn1(self.mask * x)  # 2 is correction coefficient
            xx = F.relu(torch.cat([
                xx,
                -xx,
                self.mask.repeat(n_batch, 1, 1, 1)], dim=1)  # allow for spatial masking specific adjustment
            )  # additional degree of freedom; non-linearity
        return xx


class CombinedLayer(nn.Module):

    def __init__(self, input_shape, hidden_dim, level=1):
        super(CombinedLayer, self).__init__()
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.level = level
        self._make_masks()
        in_channels = input_shape[0]
        self.spatial = nn.Sequential(OrderedDict([
            ("layer_0", CouplingLayer(in_channels, hidden_dim, self.mask_type1)),
            ("layer_1", CouplingLayer(in_channels, hidden_dim, 1 - self.mask_type1)),
            ("layer_2", CouplingLayer(in_channels, hidden_dim, self.mask_type1)),
        ]))
        in_channels = 4 * in_channels
        self.channelwise = nn.Sequential(OrderedDict([
            ("layer_0", CouplingLayer(in_channels, hidden_dim, self.mask_type2[0])),
            ("layer_1", CouplingLayer(in_channels, hidden_dim, self.mask_type2[1])),
            ("layer_2", CouplingLayer(in_channels, hidden_dim, ~self.mask_type2[0])),
            ("layer_3", CouplingLayer(in_channels, hidden_dim, ~self.mask_type2[1])),
        ]))

    def _make_masks(self):
        n_channels, height, width = self.input_shape
        row = torch.arange(height) + 1
        col = torch.arange(width)
        self.mask_type1 = torch.remainder(row[:, None] + col[None, :], 2).reshape(1, 1, height, width)
        patch_size = 4**self.level
        self.mask_type2 = torch.tensor(
            [[1, 1, 0, 0], [1, 0, 1, 0]]
        ).repeat_interleave(patch_size//4, dim=1).repeat((1, 4*n_channels//patch_size)).bool()

    @staticmethod
    def squeeze(x):
        batch_size, n_channels, height, width = x.shape
        out = x.reshape(batch_size, n_channels, height // 2, 2, width // 2, 2)
        out = out.transpose(3, 4).reshape(batch_size, n_channels, height // 2, width // 2, 4)
        out = out.permute(0, 1, 4, 2, 3).reshape(batch_size, -1, height // 2, width // 2)
        return out

    @staticmethod
    def unsqueeze(x):
        batch_size, n_channels, height, width = x.shape
        out = x.reshape(batch_size, n_channels//4, 4, height, width).permute(0, 1, 3, 4, 2)
        out = out.reshape(batch_size, n_channels//4, height, width, 2, 2).transpose(3, 4)
        out = out.reshape(batch_size, n_channels//4, height*2, width*2)
        return out

    def forward(self, x):
        if isinstance(x, tuple):
            x, logdet = x
        else:
            logdet = None
        x, logdet = self.spatial((x, logdet))
        x = self.squeeze(x)
        x, logdet = self.channelwise((x, logdet))
        return x, logdet

    def backward(self, x):
        for i in range(3, -1, -1):
            x = getattr(self.channelwise, f"layer_{i}").backward(x)
        x = self.unsqueeze(x)
        for i in range(2, -1, -1):
            x = getattr(self.spatial, f"layer_{i}").backward(x)
        return x


class RealNVP(nn.Module):
    def __init__(self, input_shape, hidden_dim=None, num_flows=None):
        super(RealNVP, self).__init__()
        in_channels, height, width = input_shape
        if num_flows is None:
            num_flows = int(math.log(min(height, width), 2)) - 2
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        for i in range(num_flows):
            setattr(self, f"flow_{i}", CombinedLayer(input_shape, hidden_dim, level=i+1))
            input_shape = self.update_shape(input_shape)
        self.num_flows = num_flows

    @staticmethod
    def update_shape(shape):
        n_channels, height, width = shape
        return n_channels * 4, height // 2, width // 2

    def forward(self, x):
        logdet = 0
        for i in range(self.num_flows):
            x, logdet = getattr(self, f"flow_{i}")((x, logdet))
        logdet += -torch.sum(x ** 2, dim=(1, 2, 3)) / 2
        cxhxw = reduce(op.mul, x.shape[1:])
        logdet += -cxhxw * math.log(2 * math.pi) / 2
        neg_logp = -logdet.mean()
        return x, neg_logp

    def backward(self, x):
        for i in range(self.num_flows - 1, -1, -1):
            x = getattr(self, f"flow_{i}").backward(x)
        return x
