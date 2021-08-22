import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import mdn


# %%
class ShapeGuesser(nn.Module):
    def __init__(self, out_dim):
        super(ShapeGuesser, self).__init__()
        self.res = models.resnet18()
        # self.res = models.resnet101()
        self.fc = nn.Linear(1000, out_dim)

    def forward(self, x):
        x = self.res(x)
        x = self.fc(x)
        return x


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x


class ReshapeLayer(nn.Module):
    def __init__(self, shape):
        super(ReshapeLayer, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1, self.shape)


class KinematicMDNv3(nn.Module):
    def __init__(self, n_gaussians, out_features, p=0.5):
        super(KinematicMDNv3, self).__init__()
        in_features = 1000
        n_hidden = 256
        num_gaussians = n_gaussians
        self.model = nn.Sequential(
            models.resnet18(), nn.Dropout(p=p),
            mdn.MDN(in_features, n_hidden, out_features, num_gaussians))

    def forward(self, x):
        pi, sigma, mu = self.model(x)
        return pi, sigma, mu


# %%
class KinematicMDNv2(nn.Module):
    def __init__(self, n_gaussians, out_features):
        super(KinematicMDNv2, self).__init__()
        in_features = 1000
        n_hidden = 256
        num_gaussians = n_gaussians
        self.model = nn.Sequential(
            models.resnet18(),
            mdn.MDN(in_features, n_hidden, out_features, num_gaussians))

    def forward(self, x):
        pi, sigma, mu = self.model(x)
        return pi, sigma, mu


# %%
class KinematicMDN(nn.Module):
    def __init__(self, n_gaussians):
        super(KinematicMDN, self).__init__()
        in_features = 4600
        n_hidden = 4600
        out_features = 22
        num_gaussians = n_gaussians

        self.model = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding=0),
            # nn.ReLU(),
            nn.Tanh(),
            # PrintLayer(),
            nn.Conv2d(8, 8, 3, stride=2, padding=0),
            # nn.ReLU(),
            nn.Tanh(),
            # PrintLayer(),
            nn.Conv2d(8, 4, 3, stride=2, padding=0),
            # nn.ReLU(),
            nn.Tanh(),
            # PrintLayer(),
            ReshapeLayer(4600),
            # PrintLayer(),
            mdn.MDN(in_features, n_hidden, out_features, num_gaussians))

    def forward(self, x):
        pi, sigma, mu = self.model(x)
        return pi, sigma, mu


# TODO: so, it's time to design a network structure
# TODO: how how how how how how how how how how how
# TODO: multi-dimensional...n_gaussian * d ? for mu and sigma?
# read this shit https://arxiv.org/pdf/1605.03170.pdf
# %%
# kmdn = KinematicMDN(n_gaussians=16)
# x = torch.rand(25,1,108,192)
# pi,sigma,mu=kmdn(x)
# print(pi.shape)
# print(sigma.shape)
# print(mu.shape)
