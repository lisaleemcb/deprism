import numpy as np

import utils

class Models:
    def __init__(self, k, params):
        self.k = k
        self.params = params
        self.biases = False

    def pspec(self, matter_pspec):
        return self.biases * matter_pspec

    def set_params(self, params):
        self.params = params

    def get_pnames(self):
        pass

    def get_pvals(self):
        pass


class ScalarBias(Models):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    def pspec(self, k_indices, params=None, N=None):
        if params is None:
            b_i = self.params['b_i']
            b_j = self.params['b_j']
            b_k = self.params['b_k']
            P_m = self.params['P_m'][k_indices]

        else:
            b_i = params['b_i']
            b_j = params['b_j']
            b_k = params['b_k']
            P_m = params['P_m'][k_indices]

        #P_ii = b_i**2 * P_m
        P_jj = b_j**2 * P_m
        P_ij = b_i * b_j * P_m
        P_jk = b_j * b_k * P_m
        P_ki = b_k * b_i * P_m

        pspec = np.asarray([*P_jj, *P_ij, *P_jk, *P_ki]).flatten() # np.zeros((int(comb(runs, 2) + runs), n_bins)
        #pspec = np.asarray([*P_ii, *P_ij, *P_jk, *P_ki]).flatten() # np.zeros((int(comb(runs, 2) + runs), n_bins)

        if N is not None:
            return pspec + utils.generate_noise(N)

        return pspec

class ScalarBias_crossonly(Models):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    def pspec(self, k_indices, params=None, N=None):
        if params is None:
            b_i = self.params['b_i']
            b_j = self.params['b_j']
            b_k = self.params['b_k']
            P_m = self.params['P_m'][k_indices]

        else:
            b_i = params['b_i']
            b_j = params['b_j']
            b_k = params['b_k']
            P_m = params['P_m'][k_indices]

        P_ij = b_i * b_j * P_m
        P_jk = b_j * b_k * P_m
        P_ki = b_k * b_i * P_m

        pspec = np.asarray([*P_ij, *P_jk, *P_ki]).flatten() # np.zeros((int(comb(runs, 2) + runs), n_bins)
        #pspec = np.asarray([*P_ii, *P_ij, *P_jk, *P_ki]).flatten() # np.zeros((int(comb(runs, 2) + runs), n_bins)

        if N is not None:
            return pspec + utils.generate_noise(N)

        return pspec

class ScalarBias_2auto_3cross(Models):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    def pspec(self, k_indices, params=None, N=None):
        if params is None:
            b_i = self.params['b_i']
            b_j = self.params['b_j']
            b_k = self.params['b_k']
            P_m = self.params['P_m'][k_indices]

        else:
            b_i = params['b_i']
            b_j = params['b_j']
            b_k = params['b_k']
            P_m = params['P_m'][k_indices]

        P_ii = b_i**2 * P_m
        P_jj = b_j**2 * P_m
        P_ij = b_i * b_j * P_m
        P_jk = b_j * b_k * P_m
        P_ki = b_k * b_i * P_m

        pspec = np.asarray([*P_ii, *P_jj, *P_ij, *P_jk, *P_ki]).flatten() # np.zeros((int(comb(runs, 2) + runs), n_bins)

        if N is not None:
            return pspec + utils.generate_noise(N)

        return pspec

class ScalarBias_plusBias(Models):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    def pspec(self, k_indices, params=None, N=None):
        if params is None:
            b_i = self.params['b_i']
            b_j = self.params['b_j']
            b_k = self.params['b_k']
            P_m = self.params['P_m'][k_indices]

        else:
            b_i = params['b_i']
            b_j = params['b_j']
            b_k = params['b_k']
            P_m = params['P_m'][k_indices]

        P_ii = b_i**2 * P_m
        P_ij = b_i * b_j * P_m
        P_jk = b_j * b_k * P_m
        P_ki = b_k * b_i * P_m

        pspec = np.asarray([*P_ii, *P_ij, *P_jk, *P_ki, b_i]).flatten() # np.zeros((int(comb(runs, 2) + runs), n_bins)

        if N is not None:
            return pspec + utils.generate_noise(N)

        return pspec

class LoggedScalarBias(Models):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    def pspec(self, k_indices, params=None):
        if params is None:
            b_i = self.params['b_i']
            b_j = self.params['b_j']
            b_k = self.params['b_k']
            P_m = self.params['P_m'][k_indices]

        else:
            b_i = self.params['b_i']
            b_j = params['b_j']
            b_k = params['b_k']
            P_m = params['P_m'][k_indices]

        P_ii = b_i**2 * P_m
        P_ij = b_i * b_j * P_m
        P_jk = b_j * b_k * P_m
        P_ki = b_k * b_i * P_m

        pspec = np.asarray([np.log(P_ii), np.log(P_ij), np.log(P_jk), np.log(P_ki)]).flatten() # np.zeros((int(comb(runs, 2) + runs), n_bins)

        return pspec

class DegreeOneBias(Models):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    def pspec(self, k_indices, params=None):
        if params is None:
            a_0 = self.params['a_i'] #params[0]
            c_0 = self.params['c_i'] #params[0]

            a_j = self.params['a_j']
            c_j = self.params['c_j']

            a_k = self.params['a_k']
            c_k = self.params['c_k']

            P_m = self.params['P_m'][k_indices]

        else:
            a_0 = self.params['a_i'] #params[0]
            c_0 = self.params['c_i'] #params[0]

            a_j = params['a_j']
            c_j = params['c_j']

            a_k = params['a_k']
            c_k = params['c_k']

            P_m = params['P_m'][k_indices]

        k = self.k[k_indices]

        P_ii = (a_0 * k + c_0)**2 * P_m
        P_ij = (a_0 * k + c_0) * (a_j * k + c_j) * P_m
        P_jk = (a_j * k + c_j) * (a_k * k + c_k) * P_m
        P_ki = (a_0 * k + c_0) * (a_k * k + c_k) * P_m
        #P_11 = (a_j * k + c_j)**2 * P_m
        #P_22 = (a_k * k + c_k)**2 *  P_m

        pspec = np.asarray([P_ii, P_ij, P_jk, P_ki]).flatten()

        return pspec

class COemission(Models):
    def __init__(self, masses, power=3/5., normalize=True, **kwargs):
        self.masses = masses
        self.power = power
        super().__init__(**kwargs)

    def emission(self):
        N = 3.2e4
        SFR = utils.mass2SFR(self.masses)

        luminosity = N * SFR**self.power
        brightnesstemp = utils.luminosity2brightnesstemp(luminosity, 1, 1)

        return brightnesstemp

class CIIemission(Models):
    def __init__(self, masses, z=6, normalize=True, **kwargs):
        self.masses = masses
        super().__init__(**kwargs)

    def emission(self):
        SFR = utils.mass2SFR(self.masses)
        # using model Dumitru et al. (2019)

        # logL = (1.4 - 0.07 * z) * np.log(SFR) + 7.1 - 0.07 * z
        #  return np.exp(logL)

        # using model De Looze et al. (2014) (see Section 5.5.5 eq. 17)
        log_L_CII = (np.log(SFR) + 8.52) / 1.18
        luminosity = np.exp(log_L_CII)
        brightnesstemp = utils.luminosity2brightnesstemp(luminosity, 1, 1)

        return brightnesstemp
