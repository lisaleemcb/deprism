import numpy as np
import copy as cp
import scipy
import matplotlib.pyplot as plt

import analysis
import models
import utils

from astropy.cosmology import Planck18
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

"""
functions required for noise power calculations
(expressions from Padmanabhan et al. 2022)
"""

def calc_P_N(survey_specs, redshift=7.0, rest_wavelength=None):
    if rest_wavelength is None:
        rest_wavelength = survey_specs['rest_wavelength']

    V_pix = calc_V_pix(survey_specs, redshift, rest_wavelength)
    t_pix = calc_t_pix(survey_specs, redshift, rest_wavelength)
    sigma_N = survey_specs['sigma_N']

    P_N = V_pix * sigma_N**2 / t_pix

    return P_N

def calc_V_pix(survey_specs, redshift, rest_wavelength):
    theta_beam = calc_theta_beam(survey_specs['D_dish'], redshift,
                                            rest_wavelength).to(u.arcmin)

    V_pix = (1.1e3 * (u.Mpc / Planck18.h)**3
                * (rest_wavelength / (158 * u.um))
                * np.sqrt((1 + redshift) / 8)
                * (theta_beam / (10 * u.arcmin))**2
                * (survey_specs['delta_nu'] / (400 * u.MHz)))

    return V_pix

def calc_t_pix(survey_specs, redshift, rest_wavelength):
    theta_beam = calc_theta_beam(survey_specs['D_dish'], redshift,
                                rest_wavelength).to(u.arcmin)
    Omega_beam = calc_Omega_beam(theta_beam)

    print(Omega_beam)

    t_pix = (survey_specs['t_obs'] * survey_specs['N_spec_eff']
            * Omega_beam / survey_specs['S_A'])

    return t_pix.to(u.s, equivalencies=u.dimensionless_angles())

"""
functions related to instrument beams
(expressions from Padmanabhan et al. 2022 and )
"""
def calc_theta_beam(D_dish, redshift, rest_wavelength):
    """Calculates full width at half maximum (FWHM)

    Parameters
    ----------
    rest_wavelength : units of length
        rest wavelength of the target line
    redshift :
        redshift
    D_dish :
        dish size of instrument

    Returns
    -------
    float
        value of FWHM of instrument beam in radians
    """

    theta_beam = rest_wavelength * (1 + redshift) / D_dish

    return theta_beam.decompose() * u.radian

def calc_Omega_beam(theta_beam=None, sigma_beam=None):
    """Calculates Omega beam

    Parameters
    ----------
    theta_beam : float
    sigma_beam : float

    Returns
    -------
    float
        Omega_beam
    """
    if theta_beam is not None and sigma_beam is not None:
        raise Exception('can only specify either theta or sigma beam, but not both')

    if theta_beam is not None:
        Omega_beam = 2 * np.pi * (theta_to_sigma(theta_beam))**2

    if sigma_beam is not None:
        Omega_beam = 2 * np.pi * sigma_beam**2

    return Omega_beam.to(u.radian**2)

def theta_to_sigma(theta_beam):
    sigma = theta_beam / np.sqrt(8 * np.log(2))

    return sigma.to(u.radian)

def angular_res(D_dish, obs_wavelength):
 # in meters
    angular_res = obs_wavelength / D_dish

    return angular_res

"""
functions for calculating the spatial extent of the survey
"""
def calc_V_survey(survey_specs, redshift, rest_wavelength):
    # from Gong et al. 2011
    S_A = survey_specs['S_A']
    B_nu = survey_specs['B_nu']
    y = rest_wavelength * (1 + redshift)**2 / Planck18.H(redshift)
    R = Planck18.comoving_distance(redshift)

    V_survey = R**2 * y * S_A * B_nu

    return V_survey.decompose().to(u.Mpc**3, equivalencies=u.dimensionless_angles())

def calc_L_para_min(survey_specs, redshift, rest_wavelength):
    nu_rest = rest_wavelength.to(u.Hz, equivalencies=u.spectral())
    nu_obs = utils.calc_nu_obs(nu_rest, redshift)

    L_para = ((const.c / Planck18.H(redshift))
                * (survey_specs['delta_nu'] * (1 + redshift)**2 / nu_obs))

    return L_para.decompose().to(u.Mpc)

def calc_L_para_max(survey_specs, redshift, rest_wavelength):
    nu_rest = rest_wavelength.to(u.Hz, equivalencies=u.spectral())
    nu_obs = utils.calc_nu_obs(nu_rest, redshift)

    L_para = ((const.c / Planck18.H(redshift))
                * (survey_specs['B_nu'] * (1 + redshift)**2 / nu_obs))

    return L_para.decompose().to(u.Mpc)

def calc_L_perp_min(survey_specs, redshift, rest_wavelength):
    # This assumes a square survey
    R_z = Planck18.comoving_distance(redshift)
    theta_beam = calc_theta_beam(survey_specs['D_dish'], redshift,
                                rest_wavelength).to(u.arcmin)
    sigma_beam = theta_to_sigma(theta_beam)

    return (R_z * sigma_beam).to(u.Mpc, equivalencies=u.dimensionless_angles())

def calc_L_perp_max(survey_specs, redshift, rest_wavelength):
    # This assumes a square survey
    R_z = Planck18.comoving_distance(redshift)

    return np.sqrt(R_z**2 * survey_specs['S_A']).to(u.Mpc, equivalencies=u.dimensionless_angles())

def set_L_perp(k_perp_min, redshift):
    R_z = Planck18.comoving_distance(redshift) * Planck18.h
    Omega_surv = (2 * (2 * np.pi)**2 / (k_perp_min * R_z**2)) * u.radian**2

    return Omega_surv.to(u.degree**2)

def set_L_para(k_para_min, redshift, nu_rest):
    B_nu = (2 * np.pi * Planck18.H(redshift) * nu_rest) / \
            (const.c * k_para_min * (1 + redshift)**2) * Planck18.h

    return B_nu

"""
functions for calculating the windowing functions that sets the survey resolution
"""
def calc_sigma_perp(survey_specs, redshift, rest_wavelength):

    theta_beam = calc_theta_beam(survey_specs['D_dish'], redshift,
                                            rest_wavelength).to(u.arcmin)
    sigma_beam = theta_to_sigma(theta_beam)

    sigma_perp = Planck18.comoving_distance(redshift) * sigma_beam

    return sigma_perp.to(u.Mpc, equivalencies=u.dimensionless_angles())

def calc_sigma_para(survey_specs, redshift, nu_obs):

    sigma_para = ((const.c / Planck18.H(redshift))
                * (survey_specs['delta_nu'] * (1 + redshift)**2 / nu_obs))

    return sigma_para.decompose().to(u.Mpc)

def set_sigma_para(sigma_para, redshift, nu_obs):
    # finds the appropriate delta_nu to get specified sigma_parallel
    delta_nu = (sigma_para * Planck18.H(redshift) * nu_obs) / (const.c * (1 + redshift))

    return delta_nu

def calc_W_k(mu, k, sigma_perp, sigma_para):
    W_k = np.exp(-k**2 * sigma_perp**2) * np.exp(-mu**2 * k**2 * (sigma_para**2 - sigma_perp**2))

    return W_k

def calc_W_beam(k, survey_specs, redshift, rest_wavelength):
    k = k / u.Mpc
    W_beam = np.zeros(len(k))
    mu = np.linspace(0,1,int(1e5))
    nu_rest = rest_wavelength.to(u.Hz, equivalencies=u.spectral())
    nu_obs = utils.calc_nu_obs(nu_rest, redshift)

    sigma_perp = calc_sigma_perp(survey_specs, redshift, rest_wavelength)
    sigma_para = calc_sigma_para(survey_specs, redshift, nu_obs)

    for i in range(len(k)):
        W_beam[i] = simps(calc_W_k(mu, k[i], sigma_perp, sigma_para), mu)

    return W_beam

def calc_W_volume(k, k_perp_min, k_para_min):
    k = k / u.Mpc
    W_volume = 1 - (np.sqrt(np.pi) * k_para_min) / (2 * k) * scipy.special.erf(k / k_para_min)

    return W_volume

def calc_W(k, k_perp_min, k_para_min, survey_specs, redshift, rest_wavelength):
    W_beam = calc_W_beam(k, survey_specs, redshift, rest_wavelength)
    W_volume = calc_W_volume(k, k_perp_min, k_para_min)

    W = np.sqrt(W_beam * W_volume)

    return W

"""
others
"""

"""def calc_L_perp(redshift, Omega):
    # This assumes a square survey
    return np.sqrt(Planck18.comoving_distance(redshift)**2 \
            * Omega).to(u.Mpc, equivalencies=u.dimensionless_angles()) * Planck18.h

def calc_L_para(redshift, nu_rest, delta_nu):
    return ((const.c / Planck18.H(redshift)) * delta_nu * (1 + redshift)**2 / \
            nu_rest).to(u.Mpc) * Planck18.h

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
    r = Planck18.comoving_distance(redshift)
    y = (lambda_rest * (1 + redshift)**2) / Planck18.H(redshift)

    V = r**2 * y * A_s * B_nu * Planck18.h**3

    return V.to(u.Mpc**3, equivalencies=u.dimensionless_angles())
"""
