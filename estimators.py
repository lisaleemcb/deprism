import numpy as np

class Estimators:
    # Initializer / Instance Attributes
    def __init__(self, data, noise, guesses=None):
        self.data = data
        self.noise = noise
        self.guesses = guesses

    @staticmethod
    def LSE_generic(data, noise, A):
        errors = np.linalg.inv(np.dot(A.T, np.dot(np.linalg.inv(noise), A)))
        estimator = np.dot(errors, np.dot(A.T, np.dot(np.linalg.inv(noise), data)))

        return estimator, errors

    def LSE_3cross_1auto_scalarbias_constraint(self):
        if data.shape is None:
            raise Exception('data shape must be 5 observations')
        A = np.array([[2, 0, 0, 1],
                    [1, 1, 0, 1],
                    [0, 1, 1, 1],
                    [1, 0, 1, 1],
                    [1, 0, 0, 0]])

        return LSE_generic(self.data, self.noise, A)

    def pspec_from_3lines(auto, cross_ij, cross_ki, cross_jk, b_0=1.45103492e-17):
        return ((auto**3 * cross_ij * cross_ki)/(b_0**8 * cross_jk))**.25

    def pspec_from_3lines_plus_autonoise(cross_ij, cross_ki, cross_jk, b_0=1.45103492e-17):
        return (cross_ij * cross_ki) / (b_0**2 * cross_jk)

    def LSE_lin_perturb(self, k_1, k_2, order1_perturb=None):
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

    def LSE_loglin_perturb(self, k_1, k_2):
        a_0 = self.guesses['a_0']
        c_0 = self.guesses['c_0']
        a_j = self.guesses['a_j']
        c_j = self.guesses['c_j']
        a_k = self.guesses['a_k']
        c_k = self.guesses['c_k']

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
