import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from solver import bisect


class PlanarFlow(nn.Module):
    """
    z_{k+1} = z_k + utanh(w^Tz_k+b)
    w: projection
    b: bias
    u: expanding/contracting direction
    """

    def __init__(self, input_dim, random_init=False):
        super(PlanarFlow, self).__init__()
        self.input_dim = input_dim
        if random_init:
            torch.manual_seed(1234)
            self.direction = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(1, self.input_dim)))
            self.projection = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(1, self.input_dim)))
            self.modify()
        else:
            self.direction = nn.Parameter(torch.zeros(1, self.input_dim))
            self.projection = nn.Parameter(torch.zeros(1, self.input_dim))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        wxb = torch.sum(x * self.projection, dim=-1, keepdim=True) + self.bias
        hx = torch.tanh(wxb)
        Dhx = 1 / (torch.cosh(wxb)) ** 2
        coef = (self.projection * self.direction).sum(dim=-1, keepdim=True)
        out = x + self.direction * hx
        logdet = torch.log((1 + Dhx * coef).abs())  # log determinant of Jacobian
        return out, logdet

    def modify(self):
        coef = torch.sum(self.projection * self.direction)  # w^Tu
        # in order to guarantee invertibility
        # it suffices to have w^Tu > -1
        # we follow the appendix to modify u
        if coef <= -1:
            m_coef = -1 + torch.log(1 + torch.exp(coef))
            unit_proj = self.projection.data / torch.sum(self.projection.data ** 2).sqrt()
            self.direction.data = self.direction.data + (m_coef - coef) * unit_proj

    def inverse(self, z):
        coef = (self.projection * self.direction).sum().item()
        bias = self.bias.item()

        def mapping(x):
            fx = x + coef * np.tanh(x + bias)
            return fx

        fx = np.sum(self.projection.cpu().numpy() * z, axis=-1)
        parallel = bisect(mapping, fx, x_min=fx - coef, x_max=fx + coef)
        hx = np.tanh(parallel + bias)
        return z - self.direction.cpu().numpy() * hx[:, None]


class RadialFlow(nn.Module):
    def __init__(self, input_dim, random_init=False):
        super(RadialFlow, self).__init__()
        self.input_dim = input_dim
        if random_init:
            torch.manual_seed(1234)
            self.reference = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(1, self.input_dim)))
            self.alpha = nn.Parameter(torch.randn(1).abs())
            self.beta = nn.Parameter(torch.randn(1))
            self.modify()
        else:
            self.reference = nn.Parameter(torch.zeros(1, self.input_dim))
            self.alpha = nn.Parameter(torch.zeros(1))
            self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        r = torch.sum((x - self.reference) ** 2, dim=-1, keepdim=True).sqrt()
        h = self.beta / (self.alpha + r)
        Dh = -self.beta / (self.alpha + r) ** 2
        out = x + h * (x - self.reference)
        logdet = (self.input_dim - 1)*torch.log((1 + h).abs()) + torch.log(1 + h + Dh * r)
        return out, logdet

    def modify(self):
        self.alpha.data = torch.max(torch.zeros(1).to(self.alpha), self.alpha.data)
        if self.beta.item() <= -self.alpha.item():
            self.beta.data = -self.alpha.data + torch.log(1 + torch.exp(self.beta))

    def inverse(self, z):
        alpha = self.alpha.item()
        beta = self.beta.item()
        ref = self.reference.data.cpu().numpy()

        def mapping(x):
            fx = x + x * beta / (x + alpha)
            return fx

        fx = np.sqrt(np.sum((z - ref) ** 2, axis=1))
        r = bisect(mapping, fx, x_min=np.zeros(len(fx)), x_max=fx + np.abs(beta))

        return ref + r[:, None] * (z - ref) / fx[:, None]


class NormalizingFlow(nn.Module):
    def __init__(
            self,
            input_dim,
            n_maps,
            transformation="planar"
    ):
        super(NormalizingFlow, self).__init__()
        self.input_dim = input_dim
        self.n_maps = n_maps
        self.transformation = transformation
        self.flows = nn.Sequential(OrderedDict(
            (f"flow_{i}", self.get_flow())
            for i in range(self.n_maps)
        ))

    def __iter__(self):
        self.map_id = 0
        return self

    def __next__(self):
        if self.map_id < self.n_maps:
            flow = getattr(self.flows, f"flow_{self.map_id}")
            self.map_id += 1
            return flow
        else:
            raise StopIteration

    def get_flow(self):
        if self.transformation == "planar":
            return PlanarFlow(self.input_dim)
        elif self.transformation == "radial":
            return RadialFlow(self.input_dim)

    def forward(self, x):
        logp = 0
        for flow in self:
            x, logdet = flow(x)
            logp += logdet.mean()
        logp += -(x.size(1) / 2) * np.log(2 * np.pi)
        logp += -(x ** 2).sum(dim=1).mean() / 2
        return -logp

    def modify(self):
        for flow in self:
            flow.modify()

    def inverse(self, z):
        with torch.no_grad():
            x = np.array(z.clone())
            for flow in self:
                x = flow.inverse(x)
        return x
