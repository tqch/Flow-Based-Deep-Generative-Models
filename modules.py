import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoupledBatchNorm2d(nn.BatchNorm2d):
    # modified 2d batch normalization to decouple rescaling from affine transformation
    def __init__(
            self,
            num_features,
            eps=1e-05,
            momentum=0.1,
            offset=True,
            scale=True,
            track_running_stats=True,
            device=None,
            dtype=None
    ):
        affine = True
        super(DecoupledBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, device, dtype)
        self.offset = offset
        self.scale = scale
        if not offset:
            delattr(self, "bias")
            self.register_buffer("bias", torch.zeros(num_features))
        if not scale:
            delattr(self, "weight")
            self.register_buffer("weight", torch.ones([num_features]))

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, offset={offset}, scale={scale}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )


class StdConv2d(nn.Conv2d):
    # standard 2d convolution layer
    # allow for the scale argument
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros',
            device=None,
            dtype=None,
            scale=None
    ):
        super(StdConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                        groups, bias, padding_mode, device, dtype)


class WNConv2d(nn.Module):
    # 2d convolution layer with weight normalization
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            groups=1,
            bias=True,
            scale=True
    ):
        super(WNConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.weight_dir = nn.Parameter(torch.empty(
            out_channels, in_channels // self.groups, kernel_size, kernel_size))
        self.scale = scale
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer("bias", None)
        if scale:
            self.weight_norm = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer("weight_norm", torch.tensor([1.]))  # constant

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_dir)
        nn.init.ones_(self.weight_norm)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        weight_dir = self.weight_dir / torch.norm(self.weight_dir, p=2, dim=(1, 2, 3), keepdim=True)
        weight = self.weight_norm[:, None, None, None] * weight_dir
        out = F.conv2d(x, weight, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
        return out

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != 0:
            s += ', padding={padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if not self.scale:
            s += ', scale=False'
        return s.format(**self.__dict__)


class BasicBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride=1,
            groups=1,
            use_batch_norm=True,
            use_weight_norm=True
    ):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.use_batch_norm = use_batch_norm
        self.use_weight_norm = use_weight_norm

        self.conv_layer = WNConv2d if use_weight_norm else StdConv2d

        self.bn1 = self.norm_layer(in_channels, scale=False)
        self.bn2 = self.norm_layer(out_channels, scale=False)

        self.conv1 = self.conv_layer(in_channels, out_channels,
                                     3, stride, 1, groups=groups, bias=not use_batch_norm, scale=False)
        self.conv2 = self.conv_layer(out_channels, out_channels,
                                     3, 1, 1, groups=groups, bias=True, scale=True)
        if stride != 1 or in_channels != out_channels:
            self.downsample = self.conv_layer(in_channels, out_channels,
                                              1, stride=stride, groups=groups, bias=True, scale=True)

    def norm_layer(self, in_channels, scale):
        if self.use_batch_norm:
            return DecoupledBatchNorm2d(in_channels, scale=scale)
        else:
            return nn.Identity()

    def forward(self, x):
        skip = x
        x = self.conv1(F.relu(self.bn1(x)))
        x = self.conv2(F.relu(self.bn2(x)))
        if hasattr(self, "downsample"):
            skip = self.downsample(skip)
        return x + skip


class BottleneckBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride=1,
            groups=1,
            use_batch_norm=True,
            use_weight_norm=True
    ):
        super(BottleneckBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.use_batch_norm = use_batch_norm
        self.use_weight_norm = use_weight_norm

        self.bn1 = self.norm_layer(in_channels, scale=False)
        self.bn2 = self.norm_layer(in_channels, scale=False)
        self.bn3 = self.norm_layer(in_channels, scale=False)

        self.conv_layer = WNConv2d if use_weight_norm else StdConv2d

        self.conv1 = self.conv_layer(in_channels, in_channels,
                                     1, 1, 0, groups=groups, bias=not use_batch_norm, scale=False)
        self.conv2 = self.conv_layer(in_channels, in_channels,
                                     3, stride, 1, groups=groups, bias=not use_batch_norm, scale=False)
        self.conv3 = self.conv_layer(in_channels, out_channels, 1, 1, 0, groups=groups, bias=True, scale=True)
        if stride != 1 or in_channels != out_channels:
            self.downsample = WNConv2d(in_channels, out_channels, 1, groups=groups, bias=True, scale=True)

    def norm_layer(self, in_channels, scale):
        if self.use_batch_norm:
            return DecoupledBatchNorm2d(in_channels, scale=scale)
        else:
            return nn.Identity()

    def forward(self, x):
        skip = x
        x = self.conv1(F.relu(self.bn1(x)))
        x = self.conv2(F.relu(self.bn2(x)))
        x = self.conv3(F.relu(self.bn3(x)))
        if hasattr(self, "downsample"):
            skip = self.downsample(skip)
        return x + skip


class ResNet(nn.Module):
    # resnet for RealNVP
    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels,
            num_blocks=2,
            groups=1,
            skip=True,
            block=BasicBlock,
            use_batch_norm=True,
            use_weight_norm=True
    ):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.skip = skip
        self.groups = groups
        self.block = block
        self.use_batch_norm = use_batch_norm
        self.use_weight_norm = use_weight_norm
        self.conv_layer = WNConv2d if use_weight_norm else StdConv2d

        self.conv1 = self.conv_layer(in_channels, hidden_channels,
                                     3, 1, 1, groups=groups, bias=True, scale=False)
        self.skip1 = self.conv_layer(hidden_channels, hidden_channels,
                                     1, 1, 0, groups=groups, bias=True, scale=True)
        for i in range(num_blocks):
            setattr(
                self,
                f"residual_block_{i}",
                block(hidden_channels, hidden_channels, stride=1, groups=groups)
            )
            if skip:
                setattr(
                    self,
                    f"skip_connection_{i}",
                    self.conv_layer(hidden_channels, hidden_channels,
                                    1, 1, 0, groups=groups, bias=True, scale=True)
                )

        self.bn = self.norm_layer(hidden_channels, scale=False)
        self.conv2 = self.conv_layer(hidden_channels, out_channels,
                                     1, 1, 0, groups=groups, bias=True, scale=True)

    def norm_layer(self, in_channels, scale):
        if self.use_batch_norm:
            return DecoupledBatchNorm2d(in_channels, scale=scale)
        else:
            return nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        if self.skip:
            out = self.skip1(x)
        for i in range(self.num_blocks):
            x = getattr(self, f"residual_block_{i}")(x)
            if self.skip:
                out += getattr(self, f"skip_connection_{i}")(x)
        if self.skip:
            x = out
        x = F.relu(self.bn(x))
        x = self.conv2(x)
        return x

def _resnet(*args, **kwargs):
    # blocks = [BasicBlock(in_channels, 2*in_channels, groups=groups)]
    # if num_blocks > 1:
    #     blocks.extend([
    #         BasicBlock(2*in_channels, 2*in_channels, groups=groups)
    #         for _ in range(num_blocks-1)
    # ])
    # return nn.Sequential(*blocks)
    return ResNet(*args, **kwargs)