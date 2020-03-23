import numpy as np

class Signals:
    def __init__(self,n_samples, statistics):
        self.n_samples = n_samples
        self.sigma_m, self.sigma_n = statistics

    def make_signal(self):
        # Reference the name
        return np.random.normal(scale=self.sigma_m, size=self.n_samples)

    def make_noise(self):
        # Reference the name
        return np.random.normal(scale=self.sigma_n, size=self.n_samples)

    def signal(self):
        return self.make_signal() + self.make_noise()

    def mass2luminosity(masses, power=3./5, mass_0=1, normalize=True):
        N = np.mean(np.abs(masses.flatten())**2)

        return (masses / mass_0)**power / N
