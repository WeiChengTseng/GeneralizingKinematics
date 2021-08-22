"""A module for a mixture density network layer

For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.

This implementation was borrowed from https://github.com/sagelywizard/pytorch-mdn.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import math

ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)


class MDN(nn.Module):
    """A mixture density network layer

    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.

    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions

    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.

    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """
    def __init__(self, in_features, n_hidden, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.n_hidden = n_hidden
        self.z_h = nn.Sequential(nn.Linear(in_features, n_hidden), nn.Tanh())
        self.z_pi = nn.Linear(n_hidden, num_gaussians)
        self.z_sigma = nn.Sequential(
            nn.Linear(n_hidden, num_gaussians * out_features),
            # nn.Tanh(),
            nn.ELU())
        self.z_mu = nn.Linear(n_hidden, num_gaussians * out_features)

    def forward(self, x):
        z_h = self.z_h(x)
        pi = nn.functional.softmax(self.z_pi(z_h), -1)
        sigma = self.z_sigma(z_h) + 1 + 1e-8
        mu = self.z_mu(z_h)
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu


def gaussian_probability(sigma, mu, data):
    """Returns the probability of `data` given MoG parameters `sigma` and `mu`.

    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        data (BxI): A batch of data. B is the batch size and I is the number of
            input dimensions.

    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    # print('data shape in gaussian_probability', data.shape)
    data = data.unsqueeze(1).expand_as(sigma)
    ret = (ONEOVERSQRT2PI / sigma) * torch.exp(-0.5 * ((data - mu) / sigma)**2)
    return torch.prod(ret, 2)


def mdn_loss(pi, sigma, mu, target):
    """Calculates the error, given the MoG parameters and the target

    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    gp = gaussian_probability(sigma, mu, target)
    prob = pi * gp
    nll = -torch.log(torch.sum(prob, dim=1))
    return torch.mean(nll)


def torch_gumbel(x, axis=1):
    loc = torch.tensor([0.0]).expand_as(x)
    scale = torch.tensor([1.0]).expand_as(x)
    gumby = torch.distributions.gumbel.Gumbel(loc, scale)
    z = gumby.sample().to(x.device)
    indices = (torch.log(x) + z).argmax(dim=axis)
    return indices


def sample(pi, sigma, mu):
    """
    MoG samples.
    doing the categorical sampling by my damn self because Categorical sucks
    """
    n_samples = len(pi)
    k = torch_gumbel(pi)
    indices = (torch.arange(n_samples), k)
    sliced_sigma = sigma[indices]
    sliced_mu = mu[indices]
    rn = torch.randn_like(sliced_sigma)
    sampled = rn * sliced_sigma + sliced_mu
    return sampled


def og_broken_sample(pi, sigma, mu):
    """Draw samples from a MoG.
    """
    pi = torch.clamp(pi, 1e-8, 1.0)
    categorical = Categorical(pi)
    pis = list(categorical.sample().data)
    samples = Variable(sigma.data.new(
        sigma.size(0), sigma.size(2)).normal_()).to(sigma.device)
    for i, idx in enumerate(pis):
        samples[i] = samples[i] * sigma[i, idx] + mu[i, idx]
    return samples


# print('pi', pi.shape)
# print('sigma', sigma.shape)
# print('mu', mu.shape)
# print('rn', rn.shape)
# print('sigma[indices]', sigma[indices].shape)
# print('mu[indices]', mu[indices].shape)
