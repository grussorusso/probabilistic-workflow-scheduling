import numpy as np
import math
from scipy.stats import gamma, uniform, halfnorm

class Distribution():

    def __init__ (self, mean):
        self.mean = mean
    
    def get_mean (self):
        return self.mean

    def sample (self, rng: np.random.Generator, samples: int):
        assert(samples >= 1)

        s = self._sample(rng, samples)
        if samples == 1:
            return float(s)
        else:
            return s

    def _sample (self, rng: np.random.Generator, samples: int):
        raise RuntimeError("abstract distribution cannot be used")

    def get_percentile (self, percentile):
        raise RuntimeError("abstract distribution cannot be used")

    def rescaled (self, new_mean):
        raise RuntimeError("abstract distribution cannot be used")



class Gamma(Distribution):

    def __init__ (self, mean, scv):
        super().__init__(mean)
        assert(mean > 0.0)
        self.shape = 1.0/scv
        self.scale = mean/self.shape

    def _sample (self, rng: np.random.Generator, samples: int):
        return rng.gamma(self.shape, self.scale, samples)

    def get_percentile (self, percentile):
        return gamma.ppf(percentile, self.shape, scale=self.scale) 

    def rescaled (self, new_mean) -> Distribution:
        g = Gamma(new_mean, 1.0/self.shape)
        return g

    def __repr__ (self):
        return f"Gamma(SCV={1.0/self.shape})"

class Deterministic(Distribution):

    def __init__ (self, mean):
        super().__init__(mean)
        assert(mean > 0.0)

    def _sample (self, rng: np.random.Generator, samples: int):
        if samples == 1:
            return self.mean
        else:
            return self.mean*np.ones((samples,))

    def get_percentile (self, percentile):
        return self.mean

    def rescaled (self, new_mean) -> Distribution:
        return Deterministic(new_mean)

    def __repr__ (self):
        return f"Deterministic"

class Uniform(Distribution):

    def __init__ (self, mean):
        assert(mean > 0.0)
        super().__init__(mean)
        self.scv = 1.0/3
        self.a = 0
        self.b = 2*mean

    def _sample (self, rng: np.random.Generator, samples: int):
        return rng.uniform(self.a, self.b, size=samples)

    def get_percentile (self, percentile):
        return uniform.ppf(percentile, loc=self.a, scale=(self.b-self.a)) 

    def rescaled (self, new_mean) -> Distribution:
        g = Uniform(new_mean)
        return g

    def __repr__ (self):
        return f"Uniform(SCV={self.scv})"

class HalfNormal(Distribution):

    def __init__ (self, mean):
        assert(mean > 0.0)
        super().__init__(mean)
        self.sigma = mean * math.sqrt(math.pi/2.0)

    def _sample (self, rng: np.random.Generator, samples: int):
        return np.fabs(rng.normal(0, self.sigma, size=samples))

    def get_percentile (self, percentile):
        return halfnorm.ppf(percentile, scale=self.sigma) 

    def rescaled (self, new_mean) -> Distribution:
        g = HalfNormal(new_mean)
        return g

    def __repr__ (self):
        return "HalfNormal"

if __name__ == "__main__":
    g = Gamma(5.0, 2.0)
    print(g.get_mean())
    print(g.sample(np.random.default_rng(), 10))
    print(g.get_percentile(0.5))
    print(g.get_percentile(0.8))
    samples = g.sample(np.random.default_rng(), 10000)
    print(samples.var())

    g2 = g.rescaled(50.0)
    print(g2.get_mean())
    print(g2.get_percentile(0.5))
    print(g2.get_percentile(0.8))
    samples = g2.sample(np.random.default_rng(), 10000)
    print(samples.var())

    u = Uniform(0.44)
    print(u.get_mean())
    print(u.sample(np.random.default_rng(), 10))
    print(u.get_percentile(0.5))
    print(u.get_percentile(0.8))


    hn = HalfNormal(10.0)
    print(hn.get_mean())
    print(hn.get_percentile(0.5))
    print(hn.get_percentile(0.99))
    samples = hn.sample(np.random.default_rng(), 100000)
    print(samples.mean())
    print(samples.var())
