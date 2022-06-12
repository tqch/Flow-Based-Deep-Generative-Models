import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import ResNet, DecoupledBatchNorm2d
from collections import OrderedDict
from functools import reduce
import operator as op


class CouplingLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_residual_blocks, mask):
        super(CouplingLayer, self).__init__()
        self.in_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_residual_blocks = num_residual_blocks
        self.mask = mask
        factor = self._infer_mask_type()
        self.conv = ResNet(
            factor * in_channels + factor - 1,
            hidden_channels,
            factor * in_channels,
            num_blocks=num_residual_blocks
        )
        self.tanh_scale = nn.Parameter(torch.ones(factor * in_channels // 2))  # scale parameter for the tanh output
        self.bn1 = DecoupledBatchNorm2d(factor * in_channels // 2, scale=False)  # pre-coupling
        self.bn2 = DecoupledBatchNorm2d(factor * in_channels // 2, scale=False)  # post-coupling

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
            s = torch.tanh(s) * self.tanh_scale[None, :, None, None]
            x_ = torch.exp(s) * x[:, ~self.mask, :, :] + t
            x[:, ~self.mask, :, :] = self.bn2(x_)
            factor = 1
        else:
            self.mask = self.mask.to(x.device)
            xx = self.extend(x)
            s, t = self.conv(xx).chunk(2, dim=1)
            s = torch.tanh(s) * self.tanh_scale[None, :, None, None]
            s = (1 - self.mask) * s
            x_ = (torch.exp(s) * x + t) * (1 - self.mask)
            x = self.mask * x + (1 - self.mask) * self.bn2(x_)
            factor = 2
        if with_logdet:
            hxw = x.shape[2] * x.shape[3]
            logdet += torch.sum(s, dim=(1, 2, 3))
            if self.training:
                curr_var = torch.var(x_, dim=(0, 2, 3))
                logdet += -(hxw // factor) * torch.sum(
                    torch.log(curr_var + self.bn2.eps) / 2)
            else:
                logdet += -(hxw // factor) * torch.sum(
                    torch.log(self.bn2.running_var + self.bn2.eps) / 2)
        return x, logdet

    def backward(self, x):
        if self.mask_type == "channel-wise":
            out = torch.zeros_like(x)
            xx = self.extend(x)
            s, t = self.conv(xx).chunk(2, dim=1)
            s = torch.tanh(s) * self.tanh_scale[None, :, None, None]
            out[:, self.mask, :, :] = x[:, self.mask, :, :]
            out[:, ~self.mask, :, :] = (self.debatchnorm(x[:, ~self.mask, :, :]) - t) * torch.exp(-s)
        else:
            self.mask = self.mask.to(x.device)
            xx = self.extend(x)
            s, t = self.conv(xx).chunk(2, dim=1)
            s = torch.tanh(s) * self.tanh_scale[None, :, None, None]
            out = self.debatchnorm((1 - self.mask) * x)
            out = self.mask * x + (1 - self.mask) * (out - t) * torch.exp(-s)
        return out

    def debatchnorm(self, x):
        x = (x - self.bn2.bias[None, :, None, None]) / self.bn2.weight[None, :, None, None]
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

    def __init__(self, input_shape, hidden_dim, num_residual_blocks, last_level=False, multiplier=2):
        super(CombinedLayer, self).__init__()
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.num_residual_blocks = num_residual_blocks
        self.last_level = last_level
        self._make_masks()
        in_channels = input_shape[0]
        self.spatial = [
            ("layer_0", CouplingLayer(in_channels, hidden_dim, num_residual_blocks, self.mask_type1)),
            ("layer_1", CouplingLayer(in_channels, hidden_dim, num_residual_blocks, 1 - self.mask_type1)),
            ("layer_2", CouplingLayer(in_channels, hidden_dim, num_residual_blocks, self.mask_type1)),
        ]
        if last_level:
            self.spatial.append(
                ("layer_3", CouplingLayer(in_channels, hidden_dim, num_residual_blocks, 1 - self.mask_type1)))
        self.spatial = nn.Sequential(OrderedDict(self.spatial))
        self.multiplier = multiplier  # multiplier of hidden_dim in channel-wise layer

        if not last_level:
            in_channels = 4 * in_channels
            self.channelwise = nn.Sequential(OrderedDict([
                ("layer_0", CouplingLayer(in_channels, multiplier * hidden_dim, num_residual_blocks, self.mask_type2)),
                ("layer_1", CouplingLayer(in_channels, multiplier * hidden_dim, num_residual_blocks, ~self.mask_type2)),
                ("layer_2", CouplingLayer(in_channels, multiplier * hidden_dim, num_residual_blocks, self.mask_type2))
            ]))

    def _make_masks(self):
        n_channels, height, width = self.input_shape
        row = torch.arange(height) + 1
        col = torch.arange(width)
        self.mask_type1 = torch.remainder(row[:, None] + col[None, :], 2).reshape(1, 1, height, width)
        if not self.last_level:
            self.mask_type2 = torch.tensor([1, 0]).repeat_interleave(n_channels * 2).bool()

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
        out = x.reshape(batch_size, n_channels // 4, 4, height, width).permute(0, 1, 3, 4, 2)
        out = out.reshape(batch_size, n_channels // 4, height, width, 2, 2).transpose(3, 4)
        out = out.reshape(batch_size, n_channels // 4, height * 2, width * 2)
        return out

    def forward(self, x):
        if isinstance(x, tuple):
            x, logdet = x
        else:
            logdet = None
        x, logdet = self.spatial((x, logdet))
        if not self.last_level:
            x = self.squeeze(x)
            x, logdet = self.channelwise((x, logdet))
        return x, logdet

    def backward(self, x):
        if not self.last_level:
            for i in range(2, -1, -1):
                x = getattr(self.channelwise, f"layer_{i}").backward(x)
            x = self.unsqueeze(x)
        for i in range(2 + self.last_level, -1, -1):
            x = getattr(self.spatial, f"layer_{i}").backward(x)
        return x


class RealNVP(nn.Module):
    def __init__(
            self,
            input_shape,
            hidden_dim=32,
            num_residual_blocks=8,
            num_levels=2,
            factor_out=None
    ):
        super(RealNVP, self).__init__()
        in_channels, height, width = input_shape
        if num_levels is None:
            num_levels = int(math.log(min(height, width), 2)) - 2
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.num_residual_blocks = num_residual_blocks
        self.num_levels = num_levels
        self.factor_out = factor_out
        if num_levels < 3:
            self.level_0 = CombinedLayer(input_shape, hidden_dim, num_residual_blocks, multiplier=1)
            input_shape = self.update_shape(input_shape)
            self.level_1 = CombinedLayer(input_shape, hidden_dim, num_residual_blocks, last_level=True, multiplier=1)
        else:
            for i in range(num_levels - 1):
                setattr(self, f"level_{i}", CombinedLayer(input_shape, hidden_dim * 2 ** i, num_residual_blocks))
                input_shape = self.update_shape(input_shape)
            # last level
            setattr(self, f"level_{num_levels - 1}",
                    CombinedLayer(input_shape, hidden_dim * 2 ** (num_levels - 1), num_residual_blocks, last_level=True))
        self.squeeze = CombinedLayer.squeeze
        self.unsqueeze = CombinedLayer.unsqueeze

    def split(self, x):
        # split channels according to the factor out ratio
        n_channels = x.shape[1]
        order = [torch.arange(i, n_channels, 4) for i in [0, 3, 1, 2]]
        order = torch.cat(order)
        split = int(n_channels * (1 - self.factor_out))
        x_ = x[:, order[split:], :, :]
        x = x[:, order[:split], :, :]
        return x, x_

    @staticmethod
    def merge(x, x_):
        # merge channels according to the factor out ratio
        x = torch.cat([x, x_], dim=1)
        batch_size, n_channels, height, width = x.shape
        chunks = torch.chunk(x, 4, dim=1)  # split along channel axis
        chunks = [chunks[i] for i in [0, 2, 3, 1]]
        x = torch.stack(chunks, dim=2).reshape(batch_size, -1, height, width)
        return x

    def update_shape(self, shape):
        n_channels, height, width = shape
        if self.factor_out is None:
            return n_channels * 4, height // 2, width // 2
        else:
            return int(n_channels * 4 * (1 - self.factor_out)), height // 2, width // 2

    def forward(self, x):
        logdet = 0
        if self.factor_out is not None:
            fout = []
        for i in range(self.num_levels):
            x, logdet = getattr(self, f"level_{i}")((x, logdet))
            if self.factor_out is not None:
                if i != self.num_levels - 1:
                    x, x_ = self.split(x)
                    fout.append(x_)
                else:
                    for x_ in fout[-1::-1]:
                        x = self.merge(x, x_)
                        x = self.unsqueeze(x)
        logdet += -torch.sum(x ** 2, dim=(1, 2, 3)) / 2
        cxhxw = reduce(op.mul, x.shape[1:])
        logdet += -cxhxw * math.log(2 * math.pi) / 2
        neg_logp = -logdet.mean()
        return x, neg_logp

    def backward(self, x):
        if self.factor_out is not None:
            fout = []
            for i in range(self.num_levels - 1):
                x = self.squeeze(x)
                x, x_ = self.split(x)
                fout.append(x_)
        for i in range(self.num_levels - 1, -1, -1):
            x = getattr(self, f"level_{i}").backward(x)
            if self.factor_out is not None:
                if i != 0:
                    x = self.merge(x, fout[i - 1])
        return x

    def l2_reg(self):
        # calculate the l2 penalty w.r.t. tanh_scale in CouplingLayer and weight_norm in WNConv2d
        l2_reg = 0
        for name, param in self.named_parameters():
            if name.split(".")[-1] in ("tanh_scale", "weight_norm"):
                if param.requires_grad:
                    l2_reg += param.pow(2).sum()
        return l2_reg
