import numpy as np

class Models:

    def __init__(self, k, params):
        self.k = k
        self.params = params
        self.biases = False

    def pspec(self, matter_pspec):
        return self.biases * matter_pspec

    def set_params(self, params):
        self.params = params

class ScalarBias(Models):

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

        pspec = np.asarray([P_ii, P_ij, P_jk, P_ki]).flatten() # np.zeros((int(comb(runs, 2) + runs), n_bins)

        return pspec

class DegreeOneBias(Models):

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
