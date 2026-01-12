import numpy as np


class OnlineCovariance:
    def __init__(self, dim):
        self.n = 0 
        self.mean = np.zeros(dim)
        self.M2 = np.zeros((dim, dim))

    def update(self, x):
        """
        Docstring for update
        
        :param self: Description
        :param x: Description
        """

        x =  np.asarray(x)
        self.n += 1

        delta = x - self.mean
        self.mean += delta / self.n 
        delta2 = x  - self.mean

        self.M2 += np.outer(delta, delta2)

    def covar(self, unbiased=True):
        if self.n < 2: 
            return np.zeros_like(self.M2)
        denominator = self.n - 1  if unbiased else self.n
        return self.M2 / denominator 
