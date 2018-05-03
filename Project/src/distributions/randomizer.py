import numpy as np
from scipy.stats import bernoulli


class randomvals(object):
    """
    Abstract base class for randombool, randomtext, randominterval
    """
    def rvs(self):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + '()'


class randombool(randomvals):
    """
    Distribution object for sampling bools uniformly
    Args:
        size    : number of random booleans to sample (event shape)
    """
    def __init__(self, size):
        self.dist = bernoulli(p=0.5)
        self.size = size

    def rvs(self, random_state=None):
        variates = self.dist.rvs(size=self.size)
        return list(np.array([True if v == 1 else False for v in variates]))

    def __repr__(self):
        return self.__class__.__name__ + '(size={})'.format(self.size)


class randomtext(randomvals):
    """
    Distribution object for sampling random phrases from a corpus uniformly
    Args:
        phrases : list of phrases
        size    : number of random phrases to sample (event shape)
    """
    def __init__(self, phrases, size):
        assert len(set(phrases)) == len(phrases), "List of phrases have redundant phrases"
        self.phrases = np.array(sorted(phrases))
        self.size = size

    def rvs(self, random_state=None):
        rv_ = list(np.random.choice(self.phrases, self.size))
        if self.size == 1:
            return rv_[0]
        return rv_

    def __repr__(self):
        return self.__class__.__name__ + '(nphrases={}, size={})'.format(len(self.phrases), self.size)


class randominterval(randomvals):
    """
    Distribution object for sampling evenly-spaced random reals between low and high (both-inclusive)
    Args:
        low     : lower bound
        high    : higher bound
        size    : number of random variates (event shape)
        diff    : spacing between 2 consecutive values
    """
    def __init__(self, low, high, diff, size):
        self.low = low
        self.high = high
        self.size = size
        self.diff = diff
        self.vals = np.arange(low, high + diff, diff).clip(min=low, max=high)

    def rvs(self, random_state=None):
        rv_ = list(np.random.choice(self.vals, self.size))
        if self.size == 1:
            return rv_[0]
        return rv_

    def __repr__(self):
        return self.__class__.__name__ +\
            '(low={}, high={}, diff={}, size={})'.format(self.low, self.high, self.diff, self.size)
