import numpy as np
import copy as cp
import scipy
import matplotlib.pyplot as plt

import analysis
import models

from astropy.cosmology import Planck15
from astropy import units as u
from astropy import constants as const
from scipy.integrate import simps #simpson

c = 2.99792e8 # m/s (kilometers per second)
k_B = 1.381e-23 # J/K (joules per kelvin)

L_solar=3.828e26

H_0 = 70e3 # m/s/Mpc (meters per second per Megaparsec)
Omega_b = 0.046
Omega_c = 0.2589
Omega_m = 0.27
Omega_r = 10e-4
Omega_Lambda = 0.73

nu_CII = 1.900539e12 * u.Hz # Hz
nu_CO = 115.271203e9 * u.Hz # Hz
nu_O_III = 1e9 * u.Hz # Hz

lambda_CII = 158 * u.nm # nanometers

def error_bars(P_x, P_line1, P_line2, P_N, W_k, N_modes):
    sigma = (1 / np.sqrt(N_modes)) * np.sqrt(P_x**2
                                            + P_line1 * (P_line2 + P_N / W_k**2))
    return sigma

def var_x(P_i, W_i, P_j, W_j, P_Ni, P_Nj, P_x, N_modes):
    W_x = np.sqrt(W_i * W_j)

    return ((P_i * W_i + P_Ni) * (P_j * W_j + P_Nj) + P_x**2 * W_x**2) / (2 * N_modes)

def var_auto(P_i, W_i, P_N, N_modes):
    return (P_i * W_i + P_N)**2 / N_modes

def calc_N_modes(k, V_surv, align='center'):

    k_vals = np.zeros(len(k) - 1)
    delta_k = k[1:] - k[:-1]

    if align is 'left':
        k_vals = k[:-1]

    if align is 'center':
        k_vals = (k[1:] + k[:-1]) / 2

    if align is 'right':
        k_vals = k[1:]

    N_modes = k_vals**2 * delta_k * V_surv.to_value() / (4 * np.pi**2)

    return N_modes

def calc_P_N(sigma_pix, V_vox, t_pix):
    P_N = sigma_pix**2 * V_vox / t_pix

    return P_N

def calc_t_pix(N_det, t_obs, Omega_surv, sigma_beam):
    Omega_pix = 2 * np.pi * sigma_beam**2

    t_pix = N_det * t_obs / (Omega_surv / Omega_pix)

    return t_pix.to(u.s, equivalencies=u.dimensionless_angles())

def calc_L_perp(redshift, Omega):
    # This assumes a square survey
    return np.sqrt(Planck15.comoving_distance(redshift)**2 \
            * Omega).to(u.Mpc, equivalencies=u.dimensionless_angles()) * Planck15.h

def calc_L_para(redshift, nu_rest, delta_nu):
    return ((const.c / Planck15.H(redshift)) * delta_nu * (1 + redshift)**2 / \
            nu_rest).to(u.Mpc) * Planck15.h

def calc_V(redshift, width_perp, width_para, nu_rest):
    L_perp = calc_L_perp(redshift, width_perp)
    L_para = calc_L_para(redshift, nu_rest, width_para)

    return L_perp**2 * L_para

def calc_V_vox(redshift, sigma_beam, delta_nu, nu_rest):
    Omega_beam = sigma_beam**2
    V_vox = calc_V(redshift, Omega_beam, delta_nu, nu_rest)

    return V_vox

def calc_V_survey(redshift, Omega_surv, B_nu, nu_rest):
    V_survey = calc_V(redshift, Omega_surv, B_nu, nu_rest)

    return V_survey

def calc_V_surv_generic(redshift,lambda_rest, Omega_surv, B_nu):
    A_s = Omega_surv
    B_nu = B_nu
    r = Planck15.comoving_distance(redshift)
    y = (lambda_rest * (1 + redshift)**2) / Planck15.H(redshift)

    V = r**2 * y * A_s * B_nu * Planck15.h**3

    return V.to(u.Mpc**3, equivalencies=u.dimensionless_angles())

def calc_V_surv_ij(redshift, lambda_i, Omega_surv_j, B_nu_j):
    # units in nanometers, GHz
    A = 3.3e7 * u.Mpc**3 # just a random prefactor (Mpc / h)^3
    V_surv_ij = A * (lambda_i / (158 * u.micron)) * np.sqrt((1 + redshift) / 8) \
                    * (Omega_surv_j / (16 * u.degree**2)) * (B_nu_j / (20 * u.GHz))

    return V_surv_ij.decompose().to(u.Mpc**3)

def calc_L_perp(redshift, Omega):
    # This assumes a square survey
    R_z = Planck15.comoving_distance(redshift)
    return np.sqrt(R_z**2 * Omega).to(u.Mpc, equivalencies=u.dimensionless_angles()) * Planck15.h

def calc_L_para(redshift, nu_rest, delta_nu):
    return ((const.c / Planck15.H(redshift)) * delta_nu * (1 + redshift)**2 / \
            nu_rest).to(u.Mpc) * Planck15.h

def set_L_perp(k_perp_min, redshift):
    R_z = Planck15.comoving_distance(redshift) * Planck15.h
    Omega_surv = (2 * (2 * np.pi)**2 / (k_perp_min * R_z**2)) * u.radian**2

    return Omega_surv.to(u.degree**2)

def set_L_para(k_para_min, redshift, nu_rest):
    B_nu = (2 * np.pi * Planck15.H(redshift) * nu_rest) / \
            (const.c * k_para_min * (1 + redshift)**2) * Planck15.h

    return B_nu

def calc_sigma_beam(z, lambda_i, D_j):
    return (lambda_i * (1 + z) / D_j).decompose()

def FWHM_to_sigma(FWHM):
    sigma = FWHM / np.sqrt(8 * np.log(2))

    return sigma.to(u.radian)

def calc_sigma_perp(z, sigma_beam):
    sigma_perp = Planck15.comoving_distance(z) * sigma_beam

    return sigma_perp.to(u.Mpc, equivalencies=u.dimensionless_angles())

def calc_sigma_para(z, nu_obs, delta_nu):
    sigma_para = (const.c / Planck15.H(z)) * delta_nu * (1 + z) / nu_obs

    return sigma_para.decompose().decompose(bases=[u.Mpc])

def set_sigma_para(sigma_para, z, nu_obs):
    # finds the appropriate delta_nu to get specified sigma_parallel
    delta_nu = (sigma_para * Planck15.H(z) * nu_obs) / (const.c * (1 + z))

    return delta_nu

def calc_W_k(mu, k, sigma_perp, sigma_para):
    W_k = np.exp(-k**2 * sigma_perp**2) * np.exp(-mu**2 * k**2 * (sigma_para**2 - sigma_perp**2))

    return W_k

def calc_W_beam(k, sigma_perp, sigma_para):
    W_beam = np.zeros(len(k))
    mu = np.linspace(0,1,int(1e5))

    for i in range(len(k)):
        W_beam[i] = simps(calc_W_k(mu, k[i], sigma_perp, sigma_para), mu)

    return W_beam

def W_volume(k, k_perp_min, k_para_min):
    W_volume = 1 - (np.sqrt(pi) * k_para_min) / (2 * k) * np.erf(k / k_para_min)

    pass

def angular_res(wavelength, D):
 # in meters
    angular_res = wavelength / D

    return angular_res

def calc_V_surv_ij(z, lambda_i=lambda_CII, Omega_surv_j=1.7, B_nu_j=200):
    # units in nanometers, GHz
    A = 3.7e7 # just a random prefactor (c Mpc / h)^3
    V_surv_ij = A * (lambda_i / 158) * np.sqrt((1 + z) / 8) * (Omega_surv_j / 16) * (B_nu_j / 20)

    return V_surv_ij
