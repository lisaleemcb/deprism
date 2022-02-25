import numpy as np
import copy as cp
import scipy
import matplotlib.pyplot as plt

from astropy.cosmology import Planck15
from astropy import units as u
from astropy import constants as const
from scipy.integrate import simpson

c = 2.99792e8 # m/s (kilometers per second)
k_B = 1.381e-23 # J/K (joules per kelvin)

L_solar=3.828e26

H_0 = 70e3 # m/s/Mpc (meters per second per Megaparsec)
Omega_b = 0.046
Omega_c = 0.2589
Omega_m = 0.27
Omega_r = 10e-4
Omega_Lambda = 0.73

nu_CII = 1.900539e12 # Hz
nu_CO = 115.271203e9 # Hz
nu_O_III = 1e9 # Hz

lambda_CII = 158 # nanometers

def gaussian(x, mean, sigma, normed=False):
    N = 1

    if normed is True:
        N = 1 / np.sqrt(2 * np.pi * sigma**2)

    return N * np.exp(- (x - mean)**2 / (2 * sigma**2))

def SI2Jansky(I):
    return I / 1e-26

def Jansky2SI(I):
    return I * 1e-26

def m2Mpc(m):
    return m / 3.086e22

def Mpc2m(Mpc):
    return Mpc * 3.086e22

def a(z):
    return 1 / (1 + z)

def H(z, Omega_b=Omega_b, Omega_c=Omega_c, Omega_r=Omega_r, Omega_Lambda=Omega_Lambda,
                H_0=H_0):

    Omega_m = Omega_c + Omega_b
    Omega_0 = Omega_r + Omega_m + Omega_Lambda

    return H_0 * np.sqrt(Omega_r / a(z)**4 + Omega_m / a(z)**3 + Omega_Lambda)

def mass2luminosity(masses, power=3./5, mass_0=1, normalize=True):
        N = 1 # np.mean(np.abs(masses.flatten())**2)

        return (masses / mass_0)**power / N

def SFR(z):
    return 134.39 * z**(-1.18)

def phi(z):
    return .00358 * z**(-0.85)

def emissivity(z, phi=phi, SFR=SFR, L_0=6e6, alpha=-1.96):
    return  phi(z) * L_0 * L_solar * SFR(z) * scipy.special.gamma(2 + alpha)

def specific_intensity(z, e=None, nu=nu_CII, L=None, ):
    if L is None:
        e = emissivity(z)

    if L is not None:
        e = phi(z) * L * L_solar

    I = (e * c) / (4.0 * np.pi * nu * H(z))

    # this factor here is to convert from Mpc^2 to m^2
    return I / (3.0857e22)**2

def mass2SFR(masses, power=5./3, mass_0=1, normalize=True):
    # using model Pullen et al. (2013)
    N = 1.0
    SFR = N * masses**power

    return SFR

def extract_bias(k_indices, lumens, P_m):
    autospectra = np.zeros(len(k_indices) * 3)
    matterspectra = np.zeros(len(k_indices) * 3)

    for i, index in enumerate(k_indices):
        j = i * 3
        autospectra[j] = lumens[0][index]
        autospectra[j+1] = lumens[3][index]
        autospectra[j+2] = lumens[5][index]

        matterspectra[j] =  P_m[index]
        matterspectra[j+1] =  P_m[index]
        matterspectra[j+2] =  P_m[index]

    biases = np.sqrt(autospectra / matterspectra)

    return biases

def fetch_data(k_bins, k_indices, spectra, b_0=0):
    data = pop_data(k_bins, k_indices, [spectra[0], spectra[1], spectra[4],
                             spectra[2]], b_0)

    return data

def fetch_data2auto_3cross(k_bins, k_indices, lumens, b_0=0):
    lines = np.asarray([lumens[0], lumens[3], lumens[1], lumens[4], lumens[2]])
    data = np.zeros((len(k_bins[k_indices]) * 5 + 1))
    n_k = len(k_indices)

    for i in range(len(lines)):
        data[i*n_k:i*n_k+n_k] = lines[i][k_indices]

    data[-1] = b_0

    return data

def pop_data(k_bins, k_indices, spectra, b_0):
    P_00, P_01, P_12, P_02 = spectra
    data = np.zeros((len(k_bins[k_indices]) * 4 + 1))
    n_k = len(k_indices)

    for i in range(len(spectra)):
        data[i*n_k:i*n_k+n_k] = spectra[i][k_indices]

    data[-1] = b_0

    return data

def construct_noise(data):
    noise = np.identity(data.size)

    for i, data in enumerate(data):
        sigma = np.abs(.001 * np.log(data))
        noise[i,i] = np.random.normal(scale=sigma)**2

    return noise

def noise_matrix(data, sigmas):
    matrix = np.identity(data.size)
    sigma2 = np.ones(data.size)

    for i in range(data.size - 1):
        sigma2[i] = sigmas[i % 4]**2

    sigma2[-1] = sigmas[4]**2

    return sigma2 * matrix

def log_noise(noise, data):
    modified_noise = np.diag(noise) / data

    return modified_noise * np.identity(data.size)

def lines_indices(runs=3):
    lines_indices = np.zeros((int(scipy.special.comb(runs, 2) + runs), 2))

    counter = 0
    for i in range(runs):
        for j in range(i, runs):
            lines_indices[counter,0] = i
            lines_indices[counter,1] = j

            counter += 1

    return lines_indices

def plot_spectra(k, density, lumens):
    fig, ax = plt.subplots()
    pc = 0
    sc = 0

    indices = lines_indices()
    for i, val in enumerate((indices)):
        idx1 = int(val[0])
        idx2 = int(val[1])

        if val[0] == val[1]:
            # biases[pc] = np.sqrt(lumens[i,k_index] / density_pspec_dim[k_index])

            ax.loglog(k, lumens[i],
            #label="autocorrelation between line {} and line {}".format(str(line_names[int(val[0])]),
                    #           str(line_names[int(val[1])])), lw=1.5, alpha=.9)
            label="autocorrelation between line {} and line {}".format(int(indices[i,0]),
                                    int(indices[i,1])), lw=1.5, alpha=.9)


            pc += 1

    for i, val in enumerate((indices)):

        idx1 = int(val[0])
        idx2 = int(val[1])

        if val[0] != val[1]:
            ax.loglog(k, lumens[i],
            label="crosscorrelation between line {} and line {}".format(int(indices[i,0]),
                                                int(indices[i,1])), lw=1.5, alpha=.9, ls='--')
            # label="crosscorrelation between line {} and line {}".format(str(line_names[int(val[0])]),
            #                        str(line_names[int(val[1])])), lw=1.5, alpha=.9, ls='--')

            sc += 1

    # ax.loglog(k, density, label='matter power density', lw=1.5, alpha=.9, color='darkgray')

    legends = fig.legend()

    return fig
            # ax[1].loglog(k, biases[line_a] * biases[line_b] * density_pspec_dim, color=colors[0], ls='--')

def generate_noise(N):
    sigma = np.sqrt(np.diag(N))
    noise = np.zeros_like(sigma)

    for i, s in enumerate(sigma):
        noise[i] = np.random.normal(scale=s)

    return noise

def choose_params(params, k_indices):
    pvals = np.asarray(list(params.values()), dtype=object)

    ndim = len(pvals) - 1 + len(pvals[-1][k_indices])

    guesses = np.zeros(ndim)

    n_biases = len(pvals) - 1
    for i in range(n_biases):
        guesses[i] = pvals[i]

    n_bins = len(pvals[-1][k_indices])
    for i in range(n_bins):
        guesses[-n_bins + i] =  pvals[-1][k_indices[i]]

    return guesses

def errors_on_logged_autospectra(eta_i, log_P, error_eta_i, error_log_P):
    var_P_ii = (2 * np.exp(2 * eta_i) * np.exp(log_P))**2 * error_eta_i**2 + (np.exp(2 * eta_i) * np.exp(log_P))**2 * error_log_P**2

    return np.sqrt(var_P_ii)

def errors_on_autospectra(b_i, P_m, error_b_i, error_P_m):
    var_P_ii = 4 * b_i**2 * P_m**2 * error_b_i**2 + b_i**4 * error_P_m**2
    return np.sqrt(var_P_ii)

def central_diff(function, params, key, k_indices):
    # only works with one k index
    step_scale=1e-3

    front_p = cp.deepcopy(params)
    back_p = cp.deepcopy(params)

    front_func = 0
    back_func = 0

    if key != 'P_m':
        h = step_scale * params[key]

        front_p[key] = front_p[key] + h
        front_func = function(k_indices, params=front_p)

        back_p[key] = back_p[key] - h
        back_func = function(k_indices, params=back_p)

    if key == 'P_m':
        h = step_scale * params[key][k_indices]

        front_p[key][k_indices] = front_p[key][k_indices] + h
        front_func = function(k_indices, params=front_p)

        back_p[key][k_indices] = back_p[key][k_indices] - h
        back_func = function(k_indices, params=back_p)

    return (front_func - back_func) / (2.0 * h)

def Fisher(function, params, noise, k_indices):
    sigma_squared = np.diag(noise)
    n_params = len(params.keys())


    Fisher_matrix = np.zeros((n_params, n_params))
    for i, key_i in enumerate(list(params.keys())):
        for j, key_j in enumerate(list(params.keys())):

            arg = (central_diff(function, params, key_i, k_indices)[:]
                 * central_diff(function, params, key_j, k_indices)[:] / sigma_squared[:])

            Fisher_matrix[i][j] = arg.sum()


    return Fisher_matrix

def Fisher_analytic(params, noise, k_indices):
    b_i = params['b_i']
    b_j = params['b_j']
    b_k = params['b_k']
    P_m = params['P_m'][k_indices]

    s2_ii = noise[0,0]
    s2_ij = noise[1,1]
    s2_jk = noise[2,2]
    s2_ki = noise[3,3]

    Fisher = np.array([[(2*b_i*P_m)**2/s2_ii + (b_j*P_m)**2/s2_ij + (b_k*P_m)**2/s2_ki, (b_i*b_j*P_m**2)/s2_ij, (b_k*b_i*P_m**2)/s2_ki, (2*b_i**3*P_m)/s2_ii + (b_j**2*b_i*P_m)/s2_ij + (b_k**2*b_i*P_m)/s2_ki],
    [(b_i*b_j*P_m**2)/s2_ij, (b_i*P_m)**2/s2_ij + (b_k*P_m)**2/s2_jk, (b_j*b_k*P_m**2)/s2_jk, (b_i**2*b_j*P_m)/s2_ij + (b_k**2*b_j*P_m)/s2_jk],
    [(b_i*b_k*P_m**2)/s2_ki, (b_j*b_k*P_m**2)/s2_jk, (b_j*P_m)**2/s2_jk + (b_i*P_m)**2/s2_ki, (b_k*b_j**2*P_m)/s2_jk + (b_k*b_i**2*P_m)/s2_ki],
    [(2*b_i**3*P_m)/s2_ii + (b_i*b_j**2*P_m)/s2_ij + (b_k**2*b_i*P_m)/s2_ki, (b_i**2*b_j*P_m)/s2_ij + (b_j*b_k**2*P_m)/s2_jk, (b_j**2*b_k*P_m)/s2_jk + (b_i**2*b_k*P_m)/s2_ki, b_i**4/s2_ii + (b_i*b_j)**2/s2_ij + (b_j*b_k)**2/s2_jk + (b_i*b_k)**2/s2_ki]], dtype=float)

    return Fisher

def d_Fisher(function, params, sigma, *args):
    #Fisher_matrix = Fisher(function, params, sigma, *args)

    step_scale=1e-4
    d_Fisher = np.zeros((len(params), len(params)))

    def diff_step(function, params, i, j, step_1, step_2):
        stepped_params = cp.copy(params)
        stepped_params[i] = stepped_params[i] + step_1
        stepped_params[j] = stepped_params[j] + step_2

        f = Fisher(function, stepped_params, sigma, *args)[i,j]

        return f

    for i in range(len(params)):
        for j in range(i, len(params)):
            print('Now on: row ', i, ', column ', j)

            h_1 = step_scale * params[i]
            h_2 = step_scale * params[j]

            f11 = diff_step(function, params, i, j, h_1, h_2)
            f1_1 = diff_step(function, params, i, j, h_1, -h_2)
            f_11 = diff_step(function, params, i, j, -h_1, h_2)
            f_1_1 = diff_step(function, params, i, j, -h_1, -h_2)

            deriv = (f11 - f1_1 - f_11 + f_1_1) / (4 * h_1 * h_2)

            d_Fisher[i,j] = deriv
            d_Fisher[j,i] = deriv

    return d_Fisher

def overdensity(density):
    mean = np.mean(density)
    delta = (density - mean) / mean

    return delta

def FWHM_to_sigma(FWHM):
    sigma = FWHM / np.sqrt(8 * np.log(2))

    return sigma.to(u.radian)

def calc_sigma_perp(z, sigma_beam):
    sigma_perp = Planck15.comoving_distance(z) * sigma_beam

    return sigma_perp.to(u.Mpc, equivalencies=u.dimensionless_angles())

def calc_sigma_para(z, nu_obs, delta_nu):
    sigma_para = (const.c / Planck15.H(z)) * delta_nu * (1 + z) / nu_obs

    return sigma_para.decompose().decompose(bases=[u.Mpc])

def set_sigma_para(sigma_para, z, nu_rest):
    # finds the appropriate delta_nu to get specified sigma_parallel
    delta_nu = (sigma_para * Planck15.H(z) * nu_rest) / (const.c * (1 + z))

    return delta_nu

def calc_sm_func(mu, k, sigma_perp, sigma_para):
    sm_func = np.exp(-k**2 * sigma_perp**2) * np.exp(-mu**2 * k**2 * (sigma_para**2 - sigma_perp**2))

    return sm_func

def calc_W_k(k, sigma_perp, sigma_para):
    W_k = np.zeros(len(k))
    mu = np.linspace(0,1,int(1e5))

    for i in range(len(k)):
        W_k[i] = simpson(calc_sm_func(mu, k[i], sigma_perp, sigma_para), mu)

    return W_k

def nu_to_wavelength(nu):
    return (const.c / nu)

def wavelength_to_nu(wavelength):
    return (const.c / wavelength )

def calc_nu_obs(nu_rest, z):
    nu_obs = nu_rest / (1 + z)

    return nu_obs

def calc_z_of_nu_obs(nu_obs, nu_rest):
    nu_obs / nu_rest - 1

def angular_res(wavelength, D):
 # in meters
    angular_res = wavelength / D

    return angular_res

def calc_N_modes(k, delta_k, V_surv):
    N_modes = k**2 * delta_k * V_surv / (4 * np.pi**2)

    return N_modes

def calc_V_surv_ij(z, lambda_i=lambda_CII, Omega_surv_j=1.7, B_nu_j=200):
    # units in nanometers, GHz
    A = 3.7e7 # just a random prefactor (c Mpc / h)^3
    V_surv_ij = A * (lambda_i / 158) * np.sqrt((1 + z) / 8) * (Omega_surv_j / 16) * (B_nu_j / 20)

    return V_surv_ij
