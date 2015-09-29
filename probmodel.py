# -*- coding: utf-8 -*-

import numpy as np

class ProbabilityModel(object):
    """Defines a probability model, which defines a probability function over a set of variables.
    The definition of a probability model is intentionally loose to accommodate a number of cases.
    Simple cases like traditional probability distributions, such as the Normal distribution,
    the Poisson distribution, can be defined as a probability model. A more complex probability
    model can be defined as a set of probability models connected together. For instance, the
    Hidden Markov Model can be defined as probability model with parameters:
    N - the number of data points (also the number of hidden states)
    GaussianEmissionModel - a normally distributed probability model describing the emission probability
    CategoricalTransitionModel - a categorical distribution describing transitions
    """
    def __init__(self, name = 'Generic Probability Model'):
        self.name = name

    def prob(self, at, log = False):
        raise NotImplementedError()
        
    def sample(self):
        raise NotImplementedError()

    def mean(self):
        raise NotImplementedError('Mean is not defined for {0}.'format(self.name))


class NormalModel(ProbabilityModel):

    def __init__(self, dim = 1, name = 'Normal probability model'):
        """Initialize the normal probability model.
        """
        ProbabilityModel.__init__(self, name = name)
        # initialize the mean vector
        self.mu = np.zeros(dim)
        self.sigma = np.eye(dim)

    def prob(self, at, log = False):
        """Compute the probability of an observation.
        """
        
