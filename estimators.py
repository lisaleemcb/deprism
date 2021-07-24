import numpy as np
import utils

class Estimators:
    # Initializer / Instance Attributes
    def __init__(self, k, data, noise, guesses=None):
        self.k = k
        self.data = np.asarray(data)
        self.noise = noise
        self.guesses = guesses

    @staticmethod
    def LSE_generic(data, noise, A):
        e_inv = np.dot(A.T, np.dot(np.linalg.inv(noise), A))
        errors = np.linalg.inv(e_inv)
        estimator = np.dot(errors, np.dot(A.T, np.dot(np.linalg.inv(noise), data)))

        return estimator, errors, e_inv

    def LSE_3cross_1auto_scalarbias(self):
        if self.data.shape is None:
            raise Exception('data shape must be 4 observations')
        A = np.array([[2, 0, 0, 1],
                    [1, 1, 0, 1],
                    [0, 1, 1, 1],
                    [1, 0, 1, 1]])

        num_P_m = (self.data.size - 1) % 4
        print(num_P_m)

        #A = np.zeros()

        return LSE_generic(self.data, self.noise, A)

    def LSE_3CC_1PS_bias(self, noise=None, inject_noise=False):
        if self.data.shape is None:
            raise Exception('data shape must be 5 observations')

        if noise is not None:
            self.noise = utils.log_noise(noise, self.data)
            print(self.noise)

        else:
            self.noise = utils.log_noise(self.noise, self.data)

        A_base = np.array([[2, 0, 0, 1],
                    [1, 1, 0, 1],
                    [0, 1, 1, 1],
                    [1, 0, 1, 1],
                    [1, 0, 0, 0]])

        num_P_m = int((self.data.size - 1) / 4)
        num_params = 3 + num_P_m
        num_obvs = self.data.size

        A = np.zeros((num_obvs, num_params))

        for i in range(num_obvs - 1):
            #line = A_base[]
            A[i][:3] = A_base[i % 4 ][:3]
            A[i][3 + i // 4] = 1

        A[-1][0] = 1

        if inject_noise is True:
            noise_r = np.asarray([np.random.normal(scale=np.sqrt(x)) for x in np.diag(self.noise)])
            noisey_data = self.data + noise_r
            self.data = noisey_data

        #print('noise matrix is: ', self.noise)
        estimates, errors, e_inv = Estimators.LSE_generic(np.log(self.data), self.noise, A)
        return estimates, errors

    def LSE_logged_test(self, noise=None, inject_noise=False):
        if self.data.shape is None:
            raise Exception('data shape must be 5 observations')

        if noise is not None:
            self.noise = noise

        A_base = np.array([[2, 0, 0, 1],
                    [1, 1, 0, 1],
                    [0, 1, 1, 1],
                    [1, 0, 1, 1],
                    [1, 0, 0, 0]])

        num_P_m = int((self.data.size - 1) / 4)
        num_params = 3 + num_P_m
        num_obvs = self.data.size

        A = np.zeros((num_obvs, num_params))

        for i in range(num_obvs - 1):
            #line = A_base[]
            A[i][:3] = A_base[i % 4 ][:3]
            A[i][3 + i // 4] = 1

        A[-1][0] = 1

        if inject_noise is True:
            noise_r = np.asarray([np.random.normal(scale=np.sqrt(x)) for x in np.diag(self.noise)])
            noisey_data = self.data + noise_r
            self.data = noisey_data

        print('no error matrix: ', np.dot(A.T, np.dot(np.linalg.inv(self.noise), self.data)))

        #print('noise matrix is: ', self.noise)
        estimates, errors, e_inv = Estimators.LSE_generic(self.data, self.noise, A)
        return estimates, errors, e_inv

    def pspec_from_3lines(self):
        auto, cross_ij, cross_jk, cross_ki = self.data
        return ((auto**3 * cross_ij * cross_ki)/(b_0**8 * cross_jk))**.25

    def Beane_et_al(self, N, P_jj=1, P_kk=1, R_ijk=1):
        P_ii, P_ij, P_jk, P_ki = self.data

        N_i = 0
        N_j = 0
        N_k = 0

        P_ii_tot = P_ii + N_i
        P_jj_tot = P_jj + N_j
        P_kk_tot = P_kk + N_j

        var = (P_ij / P_ki)**2 * (P_ki**2 + P_ii_tot * P_kk_tot) \
                + (P_ki / P_ij)**2 * (P_ij**2 + P_ii_tot * P_jj_tot) \
                + ((P_ij * P_ki) / (P_jk**2))**2 * (P_jk**2 + P_jj_tot * P_kk_tot) \
                + ((P_ij * P_ki) / (P_jk**2)) * (P_ii_tot * P_jk + P_ij * P_ki) \
                - (((P_ij)**2 * P_ki) / (P_jk**3)) * (P_kk_tot * P_ij + P_ki * P_jk) \
                - ((P_ij * (P_ki**2))/ (P_jk**3)) * (P_jj_tot * P_ki + P_ij * P_jk)


        return P_ii, (R_ijk * P_ij * P_ki) / P_jk, var

    def pspec_from_3lines_plus_autonoise(self):
        auto, cross_ij, cross_jk, cross_ki = self.data
        return (cross_ij * cross_ki) / (b_0**2 * cross_jk)

    def LSE_lin_perturb(self, k_indices, order1_perturb=None):
        if k_indices.size < 2:
            raise Exception('this estimator requires at least 2 k-modes')

        k_1 = k_indices[0]
        k_2 = k_indices[1]

        P_m1 = self.guesses['P_m1']
        P_m2 = self.guesses['P_m2']
        a_0 = self.guesses['a_0']
        c_0 = self.guesses['c_0']
        a_j = self.guesses['a_j']
        c_j = self.guesses['c_j']
        a_k = self.guesses['a_k']
        c_k = self.guesses['c_k']

        A = np.array([[(a_0 * k_1 + c_0)**2, 0, 0, 0, 0, 0],
              [(a_0 * k_1 + c_0) * (a_j * k_1 + c_j), 0, (a_0 * k_1 + c_0) * k_1 * P_m1,
                                                        (a_0 * k_1 + c_0) * P_m1, 0, 0],
              [(a_j * k_1 + c_j) * (a_k * k_1 + c_k), 0, (a_k * k_1 + c_k) * k_1 * P_m1,
                                                        (a_k * k_1 + c_k) * P_m1,
                                                        (a_j * k_1 + c_j) * k_1 * P_m1,
                                                        (a_j * k_1 + c_j) * P_m1],
              [(a_k * k_1 + c_k) * (a_0 * k_1 + c_0),0,0,0,
                           (a_0 * k_1 + c_0) * k_1 * P_m1, (a_0 * k_1 + c_0) * P_m1],
                [0, (a_0 * k_2 + c_0)**2, 0, 0, 0, 0],
              [0, (a_0 * k_2 + c_0) * (a_j * k_2 + c_j), (a_0 * k_2 + c_0) * k_2 * P_m2,
                                                        (a_0 * k_2 + c_0) * P_m2, 0, 0],
              [0, (a_j * k_2 + c_j) * (a_k * k_2 + c_k), (a_k * k_2 + c_k) * k_2 * P_m2,
                                                        (a_k * k_2 + c_k) * P_m2,
                                                        (a_j * k_2 + c_j) * k_2 * P_m2,
                                                        (a_j * k_2 + c_j) * P_m2],
              [0, (a_k * k_2 + c_k) * (a_0 * k_2 + c_0),0,0,
                           (a_0 * k_2 + c_0) * k_2 * P_m2, (a_0 * k_2 + c_0) * P_m2]])

        if order1_perturb is None:
            order1_perturb = np.zeros_like(self)

        return self.LSE_generic(np.zeros_like(self.data) + order1_perturb, self.noise, A)

    def LSE_loglin_perturb(self, k_indices):
        if k_indices.size < 2:
            raise Exception('this estimator requires at least 2 k-modes')

        k_1 = k_indices[0]
        k_2 = k_indices[1]

        #k = self.k[k_indices]
        a_0 = self.guesses['a_0']
        c_0 = self.guesses['c_0']
        a_j = self.guesses['a_j']
        c_j = self.guesses['c_j']
        a_k = self.guesses['a_k']
        c_k = self.guesses['c_k']


        # fix to be more modular
        A_k = None

        data_1ogk = np.array([np.log(self.data[0]) - 2 * np.log(a_0 * k_1 + c_0),
                        np.log(self.data[1]) - np.log(a_0 * k_1 + c_0) - np.log(a_j * k_1 + c_j),
                        np.log(self.data[2]) - np.log(a_j * k_1 + c_j) - np.log(a_k * k_1 + c_k),
                        np.log(self.data[3]) - np.log(a_k * k_1 + c_k) - np.log(a_0 * k_1 + c_0),
                        np.log(self.data[4]) - 2 * np.log(a_0 * k_2 + c_0),
                        np.log(self.data[5]) - np.log(a_0 * k_2 + c_0) - np.log(a_j * k_2 + c_j),
                        np.log(self.data[6]) - np.log(a_j * k_2 + c_j) - np.log(a_k * k_2 + c_k),
                        np.log(self.data[7]) - np.log(a_k * k_2 + c_k) - np.log(a_0 * k_2 + c_0)]).T

        A = np.array([[1, 0, 0, 0, 0, 0],
                    [1, 0, k_1 / (a_j * k_1 + c_j), 1 / (a_j * k_1 + c_j),0,0],
                    [1, 0, k_1 / (a_j * k_1 + c_j), 1 / (a_j * k_1 + c_j),
                                            k_1 / (a_k * k_1 + c_k), 1 / (a_k * k_1 + c_k)],
                    [1, 0, 0, 0, k_1 / (a_k * k_1 + c_k), 1 / (a_k * k_1 + c_k)],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, k_2 / (a_j * k_2 + c_j), 1 / (a_j * k_2 + c_j),0,0],
                    [0, 1, k_2 / (a_j * k_2 + c_j), 1 / (a_j * k_2 + c_j),
                                            k_2 / (a_k * k_2 + c_k), 1 / (a_k * k_2 + c_k)],
                    [0, 1, 0, 0, k_2 / (a_k * k_2 + c_k), 1 / (a_k * k_2 + c_k)]])

        return self.LSE_generic(data_1ogk, self.noise, A)
