import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import copy as cp
import h5py
import sys
import corner
import copy
import scipy
import astropy.constants as const

from mpl_toolkits.mplot3d import Axes3D
from scipy.special import comb
from astropy import units as u
from zreion import apply_zreion_fast
from astropy.cosmology import Planck15

import analysis
import signals
import estimators
import fitting
import models
import utils
import survey

plt.style.use('seaborn-colorblind')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# Simulations

print('loading simulations')

which_box = 'little'
print('running analysis on', which_box, 'box')

if which_box is 'little':
    rez = 512
    box = h5py.File('L80_halos_z=6.0155.hdf5', 'r')
    print(box.keys())

    redshift = 6.0155
    masses = np.array(box[('mass')])
    pos = np.array(box[('pos')])
    density = np.array(box[('rho')])
    x, y, z = pos.T

    runs = 3
    n_bins = 20

    box_size = 80 # in Mpc
    r = np.linspace(0, box_size, rez)
    r_vec = np.stack((r, r, r))

if which_box is 'big':
    rez = 1024
    box = h5py.File('halos.z8.hdf5', 'r')
    print(box.keys())

    density = np.fromfile('rho.z=07.9589_cic_1024', dtype=np.float64).reshape(rez, rez, rez, order='F')

    #density.max()

    redshift = 7.9589
    masses = np.array(box[('m')])
    x = np.array(box[('x')])
    y = np.array(box[('y')])
    z = np.array(box[('z')])

    runs = 3
    n_bins = 20

    box_size = 160 # in Mpc
    r = np.linspace(0, box_size, rez)
    r_vec = np.stack((r, r, r))

mass_voxels, mass_edges = np.histogramdd([x,y,z], bins=rez,
                                                weights=masses)

masses_hist = plt.hist(np.log(masses), bins=50)

print('generating underlying matter density spectrum')
#print('loading underlying matter density spectrum')

delta = utils.overdensity(density)
#k, P_m = analysis.calc_pspec(r_vec, [delta], n_bins=n_bins, bin_scale='log')
#np.savez('matter_pspec_6.0155', k=k, P_m=P_m)

matter_pspec = np.load('matter_pspec_6.0155.npz')
k = matter_pspec['k']
P_m = matter_pspec['P_m']

print('yay! finished the matter stuff')

H_I_power = 1.3

L = 2.0 / 3.0
M = 1.0
H = 4.0 / 3.0

power_indices = [H_I_power, L, M]

import astropy.constants as const
L_solar=3.828e26
L_CII = 10e6
L_OIII = 10e9

luminosities_L = utils.mass2luminosity(masses, power=L)
luminosities_M = utils.mass2luminosity(masses, power=M)
luminosities_H = utils.mass2luminosity(masses, power=H)

intensities_L = utils.specific_intensity(redshift, L=luminosities_L)
intensities_M = utils.specific_intensity(redshift, L=luminosities_M)
intensities_H = utils.specific_intensity(redshift, L=luminosities_H)

I_fields = np.zeros((runs, rez, rez, rez))
scalings = np.array([3.76387000e-04, 1.06943379e+05, 6.86553108e+01])

for i, power in enumerate(power_indices):
    print('power =', power)
    intensities = utils.specific_intensity(redshift,
                            L=scalings[i] * utils.mass2luminosity(masses, power=power, mass_0=1.0))

    print('mean intensity = ', intensities.mean())
    print(' ')
    # lumens += np.random.lognormal(mean=0.0, sigma=2.0, size=None)
    I_voxels, I_edges = np.histogramdd([x,y,z], bins=rez, weights=intensities)
    I_fields[i] = I_voxels

#k = matter_pspec['k']
#P_m = matter_pspec['P_m']

# parameters
#box = 80.0  # Mpc/h
omegam = Planck15.Om0
omegab = Planck15.Ob0
hubble0 = Planck15.h

alpha = 0.564
k_0 = 0.185 # Mpc/h

# global temperature as a function of redshift
def t0(z):
    return 38.6 * hubble0 * (omegab / 0.045) * np.sqrt(0.27 / omegam * (1 + z) / 10)

def gen_21cm_fields(delta, box_size= 80.0, zmean=7, alpha=0.11, k0=0.05):
    # compute zreion field
    print("computing zreion...")
    zreion = apply_zreion_fast(delta, zmean, alpha, k0, box_size, deconvolve=False)

    return zreion

def get_21cm_fields(z, zreion, delta):
    #print("computing t21 at z=", z, "...")
    ion_field = np.where(zreion > z, 1.0, 0.0)
    t21_field = t0(z) * (1 + delta) * (1 - ion_field)

    return ion_field, t21_field

zreion = gen_21cm_fields(delta)
np.save('zreion_z6.0155', zreion)

ion_field, t21_field = get_21cm_fields(redshift, zreion, delta)

#### Checking unit conversion

def set_I_mean(Lidz_pspec_log, P_x):
    return np.sqrt(Lidz_pspec_log / P_x)


### Power law data
print('generating power law data')
# power law
spectra_pl = analysis.gen_spectra(r_vec, I_fields)

print('generating superfake data')
# indices

### Superfake data
k_indices = [6]
spectra_sf = cp.deepcopy(spectra_pl)

b_21cm = 1
b_CII = 3 * 1.1e3   # Jy/str
b_OIII = 5 * 1.1e3  # Jy/str

biases = [b_21cm, b_CII, b_OIII]
indices = utils.lines_indices()

for i in range(len(indices)):
    print(indices[i][0], indices[i][1])
    spectra_sf[1][i] = biases[int(indices[i][0])] * biases[int(indices[i][1])] * P_m

### Brightness temperature data

I_fields_bt = cp.deepcopy(I_fields)
I_fields_bt[0] = t21_field

print('generating brightness temperature data')
# full simulation
spectra_bt = analysis.gen_spectra(r_vec, I_fields_bt)

### Datasets

np.savez('pspecs_sf_z6.0155', P_21cm_21cm=spectra_sf[1][0], P_21cm_CII=spectra_sf[1][1],
                    P_21cm_OIII=spectra_sf[1][2], P_CII_CII=spectra_sf[1][3],
                    P_CII_OIII=spectra_sf[1][4], P_OIII_OIII=spectra_sf[1][5])

np.savez('pspecs_pl_z6.0155', P_21cm_21cm=spectra_pl[1][0], P_21cm_CII=spectra_pl[1][1],
                    P_21cm_OIII=spectra_pl[1][2], P_CII_CII=spectra_pl[1][3],
                    P_CII_OIII=spectra_pl[1][4], P_OIII_OIII=spectra_pl[1][5])

np.savez('pspecs_bt_z6.0155', P_21cm_21cm=spectra_bt[1][0], P_21cm_CII=spectra_bt[1][1],
                    P_21cm_OIII=spectra_bt[1][2], P_CII_CII=spectra_bt[1][3],
                    P_CII_OIII=spectra_bt[1][4], P_OIII_OIII=spectra_bt[1][5])

#pspecs_sf = np.load('pspecs_sf.npz')
#pspecs_pl = np.load('pspecs_pl.npz')
#pspecs_bt = np.load('pspecs_bt.npz')
#pspecs_bt.files

#spectra_sf = (k, [pspecs_sf['P_21cm_21cm'], pspecs_sf['P_21cm_CII'],
#               pspecs_sf['P_21cm_OIII'], pspecs_sf['P_CII_CII'],
#               pspecs_sf['P_CII_OIII'], pspecs_sf['P_OIII_OIII']])

#spectra_pl = (k, [pspecs_pl['P_21cm_21cm'], pspecs_pl['P_21cm_CII'],
#               pspecs_pl['P_21cm_OIII'], pspecs_pl['P_CII_CII'],
#               pspecs_pl['P_CII_OIII'], pspecs_pl['P_OIII_OIII']])

#spectra_bt = (k, [pspecs_bt['P_21cm_21cm'], pspecs_bt['P_21cm_CII'],
#               pspecs_bt['P_21cm_OIII'], pspecs_bt['P_CII_CII'],
#               pspecs_bt['P_CII_OIII'], pspecs_bt['P_OIII_OIII']])

#### Autocorrelations

P_21cm_21cm = spectra_bt[1][0] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)
P_CII_CII = spectra_bt[1][3] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)
P_OIII_OIII = spectra_bt[1][5] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)

#### Crosscorrelations

P_21cm_CII = spectra_bt[1][1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)
P_21cm_OIII = spectra_bt[1][2] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)
P_CII_OIII = spectra_bt[1][4] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)


# Surveys
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

def calc_L_perp(z, Omega):
    # This assumes a square survey
    return np.sqrt(Planck15.comoving_distance(z)**2 \
            * Omega).to(u.Mpc, equivalencies=u.dimensionless_angles()) * Planck15.h

def calc_L_para(z, nu_rest, delta_nu):
    return ((const.c / Planck15.H(z)) * delta_nu * (1 + z)**2 / \
            nu_rest).to(u.Mpc) * Planck15.h

def calc_V(z, width_perp, width_para, nu_rest):
    L_perp = calc_L_perp(z, width_perp)
    L_para = calc_L_para(z, nu_rest, width_para)

    return L_perp**2 * L_para

def calc_V_vox(z, sigma_beam, delta_nu, nu_rest):
    Omega_beam = sigma_beam**2
    V_vox = calc_V(z, Omega_beam, delta_nu, nu_rest)

    return V_vox

def calc_V_survey(z, Omega_surv, B_nu, nu_rest):
    V_survey = calc_V(z, Omega_surv, B_nu, nu_rest)

    return V_survey

def calc_V_surv_generic(z,lambda_rest, Omega_surv, B_nu):
    A_s = Omega_surv
    B_nu = B_nu
    r = Planck15.comoving_distance(z)
    y = (lambda_rest * (1 + z)**2) / Planck15.H(z)

    V = r**2 * y * A_s * B_nu * Planck15.h**3

    return V.to(u.Mpc**3, equivalencies=u.dimensionless_angles())

def calc_V_surv_ij(z, lambda_i, Omega_surv_j, B_nu_j):
    # units in nanometers, GHz
    A = 3.3e7 * u.Mpc**3 # just a random prefactor (Mpc / h)^3
    V_surv_ij = A * (lambda_i / (158 * u.micron)) * np.sqrt((1 + z) / 8) \
                    * (Omega_surv_j / (16 * u.degree**2)) * (B_nu_j / (20 * u.GHz))

    return V_surv_ij.decompose().to(u.Mpc**3)

### Survey specifications
def calc_L_perp(z, Omega):
    # This assumes a square survey
    return np.sqrt(Planck15.comoving_distance(z)**2 \
            * Omega).to(u.Mpc, equivalencies=u.dimensionless_angles()) * Planck15.h

def calc_L_para(z, nu_rest, delta_nu):
    return ((const.c / Planck15.H(z)) * delta_nu * (1 + z)**2 / \
            nu_rest).to(u.Mpc) * Planck15.h

def set_L_perp(k_perp_min, z):
    R_z = Planck15.comoving_distance(z) * Planck15.h
    Omega_surv = ((2 * np.pi)**2 / (k_perp_min * R_z)**2) * u.radian**2

    return Omega_surv.to(u.degree**2)

def set_L_para(k_para_min, z, nu_rest):
    B_nu = (2 * np.pi * Planck15.H(z) * nu_rest) / \
            (const.c * k_para_min * (1 + z)**2) * Planck15.h

    return B_nu

k_units = k * u.Mpc**(-1)

# max and min k values are partially determined by the simulation box size
k_perp_min_box = (2 * np.pi) / (box_size * u.Mpc)
k_para_min_box = (2 * np.pi) / (box_size * u.Mpc)

k_perp_max_box = (2 * np.pi) / ((box_size * u.Mpc) / 512)
k_para_max_box = (2 * np.pi) / ((box_size * u.Mpc) / 512)

nu_21cm = 1420.4 * u.MHz
nu_21cm_obs = utils.calc_nu_obs(nu_21cm, redshift)

lambda_21cm = utils.nu_to_wavelength(nu_21cm)

nu_CII = 1.9 * u.THz
nu_CII_obs = utils.calc_nu_obs(nu_CII, redshift)

lambda_CII = utils.nu_to_wavelength(nu_CII).to(u.micron) # micrometers

lambda_OIII = 88 * u.micron ## micrometers

nu_OIII = utils.wavelength_to_nu(lambda_OIII)
nu_OIII_obs = utils.calc_nu_obs(nu_OIII, redshift)

sigma_beam = 1.22 * lambda_CII / (3 * u.m)

Omega_surv_21cm = set_L_perp(k_perp_min_box, redshift)
Omega_surv_CII = set_L_perp(k_perp_min_box, redshift)
Omega_surv_OIII = set_L_perp(k_perp_min_box, redshift)

B_21cm = set_L_para(k_para_min_box, redshift, nu_21cm)
B_CII = set_L_para(k_para_min_box, redshift, nu_CII)
B_OIII = set_L_para(k_para_min_box, redshift, nu_OIII)

sigma_perp_21cm = survey.calc_sigma_perp(redshift, sigma_beam)
sigma_perp_CII = survey.calc_sigma_perp(redshift, sigma_beam)
sigma_perp_OIII = survey.calc_sigma_perp(redshift, sigma_beam)

delta_nu_CII = 10 * u.MHz
sigma_para_CII = survey.calc_sigma_para(redshift, nu_CII_obs, delta_nu_CII)

delta_nu_21cm = survey.set_sigma_para(sigma_para_CII, redshift, nu_21cm_obs)
sigma_para_21cm = survey.calc_sigma_para(redshift, nu_21cm_obs, delta_nu_21cm)

delta_nu_OIII = survey.set_sigma_para(sigma_para_CII, redshift, nu_OIII_obs)
sigma_para_OIII = survey.calc_sigma_para(redshift, nu_OIII_obs, delta_nu_OIII)

L_perp_21cm = calc_L_perp(redshift, Omega_surv_21cm)
L_para_21cm = calc_L_para(redshift, nu_21cm, B_21cm)

L_perp_CII = calc_L_perp(redshift, Omega_surv_CII)
L_para_CII = calc_L_para(redshift, nu_CII, B_CII)

L_perp_OIII = calc_L_perp(redshift, Omega_surv_OIII)
L_para_OIII = calc_L_para(redshift, nu_OIII, B_OIII)

k_perp_min_21cm = (2 * np.pi / L_perp_21cm)
k_perp_min_CII = (2 * np.pi / L_perp_CII)
k_perp_min_OIII = (2 * np.pi / L_perp_OIII)

k_para_min_21cm = (2 * np.pi / L_para_21cm)
k_para_min_CII = (2 * np.pi / L_para_CII)
k_para_min_OIII = (2 * np.pi / L_para_OIII)

k_perp_max_21cm = 1 / sigma_perp_21cm
k_perp_max_CII = 1 / sigma_perp_CII
k_perp_max_OIII = 1 / sigma_perp_OIII

k_para_max_21cm = 1 / sigma_para_21cm
k_para_max_CII = 1 / sigma_para_CII
k_para_max_OIII = 1 / sigma_para_OIII

print('initializing window function stuff')
### Weighting function for smoothing power spectrum
W_k_21cm = survey.calc_W_k(k_units, sigma_perp_21cm, sigma_para_21cm)
W_k_21cm

W_k_CII = survey.calc_W_k(k_units, sigma_perp_CII, sigma_para_CII)
W_k_CII

W_k_OIII = survey.calc_W_k(k_units, sigma_perp_OIII, sigma_para_OIII)
W_k_OIII

#plt.axvline(1 / sigma_para_CCATp.to_value(), color='red', ls='--', alpha=.2)
# plt.axvline(1 / sigma_perp_FYST.to_value(), color='b')

### Current and future surveys

specs_CCATp = {'sigma_pix': 0.86 * (u.MJy * u.s**(1/2) / u.steradian),
               'N_det': 20 * u.dimensionless_unscaled,
               'theta_FWMH': 46.0 * u.arcsec,
               'nu_obs_min': 200.0 * u.GHz,
               'nu_obs_max': 300.0 * u.GHz,
               'delta_nu': 2.5 * u.GHz,
               't_obs': 3400 * u.hr,
               'Omega_surv': 1.7 * u.degree**2,
               'AGN Source': 'COSMOS'}

specs_HERA = {'sigma_pix': None,
               'N_det': None,
               'theta_FWMH': None,
               'nu_obs_min': 100.0 * u.GHz,
               'nu_obs_max': 200.0 * u.GHz,
               'delta_nu': 97.8 * u.kHz,
               't_obs': None,
               'Omega_surv': 1440 * u.degree**2,
               'AGN Source': None}

specs_StageII = {'sigma_pix': 0.21 * (u.MJy * u.s**(1/2) / u.steradian),
               'N_det': 16**4 * u.dimensionless_unscaled,
               'theta_FWMH': 30.1 * u.arcsec,
               'nu_obs_min': 200.0 * u.GHz,
               'nu_obs_max': 300.0 * u.GHz,
               'delta_nu': 0.4 * u.GHz,
               't_obs': 2000 * u.hr,
               'Omega_surv': 100.0 * u.degree**2,
               'AGN Source': 'DESI'}

specs_EXCLAIM = {'sigma_pix': 0.2 * (u.MJy * u.s**(1/2) / u.steradian),
               'N_det': 30 * u.dimensionless_unscaled,
               'theta_FWMH': None,
               'sigma_beam': utils.calc_sigma_beam(6, lambda_OIII, 3.0 * u.m),
               'B_nu': 40.0 * u.GHz,
               'nu_obs_min': 420.0 * u.GHz,
               'nu_obs_max': 540.0 * u.GHz,
               'delta_nu': 1000 * u.MHz,
               't_obs': 8.0 * u.hr,
               'Omega_surv': 100.0 * u.degree**2,
               'AGN Source': 'DESI'}


### Errors
V_survey_21cm = calc_V_survey(redshift, Omega_surv_21cm, B_21cm, nu_21cm)
V_survey_CII = calc_V_survey(redshift, Omega_surv_CII, B_CII, nu_CII)
V_survey_OIII = calc_V_survey(redshift, Omega_surv_OIII, B_21cm, nu_OIII)

V_vox_21cm = calc_V_vox(redshift, sigma_beam, delta_nu_21cm, nu_21cm)
V_vox_CII = calc_V_vox(redshift, sigma_beam, delta_nu_CII, nu_CII)
V_vox_OIII = calc_V_vox(redshift, sigma_beam, delta_nu_OIII, nu_OIII)

#### N modes check

N_modes = calc_N_modes(k, V_survey_21cm, align='left')
N_modes

V_survey_21cm / V_vox_21cm

V_survey_CII / V_vox_CII

# Need to update with HERA
t_pix_21cm = calc_t_pix(specs_CCATp['N_det'], specs_CCATp['t_obs'],
                                                   Omega_surv_CII, sigma_beam)

t_pix_CII = calc_t_pix(specs_StageII['N_det'], specs_StageII['t_obs'],
                                                   Omega_surv_CII, sigma_beam)
t_pix_OIII = calc_t_pix(specs_EXCLAIM['N_det'], specs_EXCLAIM['t_obs'],
                                                   Omega_surv_OIII, sigma_beam)

P_N_21cm = calc_P_N(specs_StageII['sigma_pix'], V_vox_21cm, t_pix_21cm)
P_N_CII = calc_P_N(specs_StageII['sigma_pix'], V_vox_CII, t_pix_CII)
P_N_OIII = calc_P_N(specs_EXCLAIM['sigma_pix'], V_vox_OIII, t_pix_OIII)

N_modes = calc_N_modes(k, V_survey_21cm, align='left')

#sigma_21cm_CII = error_bars(P_21cm_CII[:-1], P_21cm_21cm[:-1], P_CII_CII[:-1],
#                                                P_N_21cm, W_k_21cm[:-1], N_modes)

var_21cm_21cm = var_x(P_21cm_21cm[:-1], W_k_21cm[:-1], P_21cm_21cm[:-1], W_k_21cm[:-1],
                     P_N_21cm, P_N_21cm, P_21cm_21cm[:-1], N_modes)
var_CII_CII = var_x(P_CII_CII[:-1], W_k_CII[:-1], P_CII_CII[:-1], W_k_CII[:-1],
                     P_N_CII, P_N_CII, P_CII_CII[:-1], N_modes)
var_OIII_OIII = var_x(P_OIII_OIII[:-1], W_k_OIII[:-1], P_OIII_OIII[:-1], W_k_OIII[:-1],
                     P_N_OIII, P_N_OIII, P_OIII_OIII[:-1], N_modes)

var_21cm_CII = var_x(P_21cm_21cm[:-1], W_k_21cm[:-1], P_CII_CII[:-1], W_k_CII[:-1],
                     P_N_21cm, P_N_CII, P_21cm_CII[:-1], N_modes)
var_21cm_OIII = var_x(P_21cm_21cm[:-1], W_k_21cm[:-1], P_OIII_OIII[:-1], W_k_OIII[:-1],
                     P_N_21cm, P_N_OIII, P_21cm_OIII[:-1], N_modes)
var_CII_OIII = var_x(P_CII_CII[:-1], W_k_CII[:-1], P_OIII_OIII[:-1], W_k_OIII[:-1],
                     P_N_CII, P_N_OIII, P_CII_OIII[:-1], N_modes)

### Comparing noise forecasts with Padmanabhan et al.

# this is just checking sigma / t_pix^(1/2) with Chung et al. (2020) for CCATp

t_pix_CCATp = calc_t_pix(specs_CCATp['N_det'], specs_CCATp['t_obs'],
                                        specs_CCATp['Omega_surv'],
                                        utils.FWHM_to_sigma(specs_CCATp['theta_FWMH']))


sig_t_pix_Chung = 6.2e3 * u.Jy / u.steradian
sig_t_pix_CCATp = specs_CCATp['sigma_pix'] / np.sqrt(t_pix_CCATp)


#### CCAT-p specifications

sigma_beam_CCATp = survey.FWHM_to_sigma(specs_CCATp['theta_FWMH'])
V_surv_CCATp = calc_V_survey(redshift, specs_CCATp['Omega_surv'],
                               specs_CCATp['nu_obs_max'] - specs_CCATp['nu_obs_min'],
                                nu_CII)
V_vox_CCATp = calc_V_vox(redshift, sigma_beam_CCATp, specs_CCATp['delta_nu'], nu_CII)
t_pix_CCATp = calc_t_pix(specs_CCATp['N_det'], specs_CCATp['t_obs'],
                                        specs_CCATp['Omega_surv'],
                                        sigma_beam_CCATp)

N_modes_CCATp = calc_N_modes(k, V_surv_CCATp, align='left')

sigma_perp_CCATp = survey.calc_sigma_perp(redshift, sigma_beam_CCATp)
sigma_para_CCATp = survey.calc_sigma_para(redshift, nu_CII_obs, specs_CCATp['delta_nu'])

P_N_CCATp = calc_P_N(specs_CCATp['sigma_pix'], V_vox_CCATp, t_pix_CCATp)
W_k_CCATp = survey.calc_W_k(k_units, sigma_perp_CCATp, sigma_para_CCATp)

var_auto_CCATp = var_auto(P_CII_CII[:-1], W_k_CCATp[:-1], P_N_CCATp, N_modes_CCATp)
var_x_CCATp = var_x(P_21cm_21cm[:-1], W_k_21cm[:-1], P_CII_CII[:-1], W_k_CCATp[:-1], P_N_21cm,
                     P_N_CCATp, P_21cm_CII[:-1], N_modes_CCATp)


#### FYST (or Stage II) specifications

sigma_beam_StageII = survey.FWHM_to_sigma(specs_StageII['theta_FWMH'])
V_surv_StageII = calc_V_survey(redshift, specs_StageII['Omega_surv'],
                               specs_StageII['nu_obs_max'] - specs_StageII['nu_obs_min'],
                                nu_CII)
V_vox_StageII = calc_V_vox(redshift, sigma_beam_StageII, specs_StageII['delta_nu'], nu_CII)
t_pix_StageII = calc_t_pix(specs_StageII['N_det'], specs_StageII['t_obs'],
                                        specs_StageII['Omega_surv'],
                                        sigma_beam_StageII)

N_modes_StageII = calc_N_modes(k, V_surv_StageII, align='left')

sigma_perp_StageII = survey.calc_sigma_perp(redshift, sigma_beam_StageII)
sigma_para_StageII = survey.calc_sigma_para(redshift, nu_CII_obs, specs_StageII['delta_nu'])

P_N_StageII = calc_P_N(specs_StageII['sigma_pix'], V_vox_StageII, t_pix_StageII)
W_k_StageII = survey.calc_W_k(k_units, sigma_perp_StageII, sigma_para_StageII)

var_auto_StageII = var_auto(P_CII_CII[:-1], W_k_StageII[:-1], P_N_StageII, N_modes_StageII)
var_x_StageII = var_x(P_21cm_21cm[:-1], W_k_21cm[:-1], P_CII_CII[:-1], W_k_StageII[:-1],
                      P_N_21cm, P_N_StageII, P_21cm_CII[:-1], N_modes_StageII)


#### EXCLAIM specifications

V_surv_EXCLAIM = calc_V_survey(redshift, specs_EXCLAIM['Omega_surv'],
                               specs_EXCLAIM['B_nu'],
                                nu_OIII)
V_vox_EXCLAIM = calc_V_vox(redshift, specs_EXCLAIM['sigma_beam'], specs_EXCLAIM['delta_nu'], nu_OIII)
t_pix_EXCLAIM = calc_t_pix(specs_EXCLAIM['N_det'], specs_EXCLAIM['t_obs'],
                                        specs_EXCLAIM['Omega_surv'],
                                        specs_EXCLAIM['sigma_beam'])
N_modes_EXCLAIM = calc_N_modes(k, V_surv_EXCLAIM, align='left')

sigma_perp_EXCLAIM = survey.calc_sigma_perp(redshift, specs_EXCLAIM['sigma_beam'])
sigma_para_EXCLAIM = survey.calc_sigma_para(redshift, nu_OIII_obs, specs_EXCLAIM['delta_nu'])

P_N_EXCLAIM = calc_P_N(specs_EXCLAIM['sigma_pix'], V_vox_EXCLAIM, t_pix_EXCLAIM)
W_k_EXCLAIM = survey.calc_W_k(k_units, sigma_perp_EXCLAIM, sigma_para_EXCLAIM)

var_auto_EXCLAIM = var_auto(P_OIII_OIII[:-1], W_k_EXCLAIM[:-1], P_N_EXCLAIM, N_modes_EXCLAIM)
var_x_EXCLAIM = var_x(P_21cm_21cm[:-1], W_k_21cm[:-1], P_OIII_OIII[:-1], W_k_EXCLAIM[:-1], P_N_21cm,
                     P_N_EXCLAIM, P_21cm_OIII[:-1], N_modes_EXCLAIM)

N_modes_EXCLAIM
### Current and upcoming surveys
sigma_beam_CCATp = survey.FWHM_to_sigma(specs_CCATp['theta_FWMH']) # FWHM_to_sigma(46) #arcsec
sigma_perp_CCATp = survey.calc_sigma_perp(redshift, sigma_beam_CCATp)
sigma_para_CCATp = survey.calc_sigma_para(redshift, nu_CII_obs, specs_CCATp['delta_nu'])

sigma_beam_FYST = 1.22 * lambda_CII / (3 * u.m)

sigma_perp_FYST = survey.calc_sigma_perp(redshift, sigma_beam_FYST.decompose())
sigma_perp_HERA = survey.calc_sigma_perp(redshift, sigma_beam_FYST.decompose())
sigma_perp_CCATp = survey.calc_sigma_perp(redshift, utils.FWHM_to_sigma(specs_CCATp['theta_FWMH']))
sigma_perp_StageII = survey.calc_sigma_perp(redshift, utils.FWHM_to_sigma(specs_StageII['theta_FWMH']))
sigma_perp_EXCLAIM_1 = survey.calc_sigma_perp(redshift, specs_EXCLAIM['sigma_beam'])

sigma_para_FYST = survey.calc_sigma_para(redshift, nu_CII_obs, 300 * u.MHz)
sigma_para_HERA = survey.calc_sigma_para(redshift, nu_21cm_obs, specs_HERA['delta_nu'])
sigma_para_CCATp = survey.calc_sigma_para(redshift, nu_CII_obs, specs_CCATp['delta_nu'])
sigma_para_StageII = survey.calc_sigma_para(redshift, nu_CII_obs, specs_StageII['delta_nu'])
sigma_perp_EXCLAIM_1 = survey.calc_sigma_perp(redshift, specs_EXCLAIM['sigma_beam'])

L_perp_CCATp = (np.sqrt(Planck15.comoving_distance(redshift)**2 \
            * specs_CCATp['Omega_surv'])).to(u.Mpc, equivalencies=u.dimensionless_angles())
L_para_CCATp = ((const.c / Planck15.H(redshift)) * (100 * u.GHz) \
          * (1 + redshift)**2 / nu_CII).to(u.Mpc)

L_perp_StageII = (np.sqrt(Planck15.comoving_distance(redshift)**2 \
            * specs_StageII['Omega_surv'])).to(u.Mpc, equivalencies=u.dimensionless_angles())
L_para_StageII = ((const.c / Planck15.H(redshift)) * (100 * u.GHz) \
          * (1 + redshift)**2 / nu_CII).to(u.Mpc)

L_perp_HERA = (np.sqrt(Planck15.comoving_distance(redshift)**2 \
            * specs_HERA['Omega_surv'])).to(u.Mpc, equivalencies=u.dimensionless_angles())
L_para_HERA = ((const.c / Planck15.H(redshift)) * (100 * u.MHz) \
          * (1 + redshift)**2 / nu_21cm).to(u.Mpc)

L_perp_EXCLAIM_1 = (np.sqrt(Planck15.comoving_distance(redshift)**2 \
            * (100 * u.degree**2))).to(u.Mpc, equivalencies=u.dimensionless_angles())
L_para_EXCLAIM_1 = ((const.c / Planck15.H(redshift)) * (100 * u.GHz) \
          * (1 + redshift)**2 / nu_OIII).to(u.Mpc)

colors = ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442', '#56B4E9']

L_perp_CCATp = (np.sqrt(Planck15.comoving_distance(redshift)**2 \
            * specs_CCATp['Omega_surv'])).to(u.Mpc, equivalencies=u.dimensionless_angles())
L_para_CCATp = ((const.c / Planck15.H(redshift)) * (100 * u.GHz) \
          * (1 + redshift)**2 / nu_CII).to(u.Mpc)

L_perp_FYST = (np.sqrt(Planck15.comoving_distance(redshift)**2 \
            * (100 * u.degree**2))).to(u.Mpc, equivalencies=u.dimensionless_angles())
L_para_FYST = ((const.c / Planck15.H(redshift)) * (100 * u.GHz) \
          * (1 + redshift)**2 / nu_CII).to(u.Mpc)

L_perp_FYST_2 = (np.sqrt(Planck15.comoving_distance(redshift)**2 \
            * (100 * u.degree**2))).to(u.Mpc, equivalencies=u.dimensionless_angles())
L_para_FYST_2 = ((const.c / Planck15.H(redshift)) * (100 * u.GHz) \
          * (1 + redshift)**2 / nu_CII).to(u.Mpc)

print(L_perp_CCATp, L_para_CCATp)
print(L_perp_FYST, L_para_FYST)
print(L_perp_FYST_2, L_para_FYST_2)

print('CCATp:')
print(L_perp_CCATp, L_para_CCATp)

print('StageII:')
print(L_perp_StageII, L_para_StageII)

print('HERA:')
print(L_perp_HERA, L_para_HERA)

print('Box:')
print(box_size * u.Mpc, box_size * u.Mpc)

k_perp_min_CCATp = (2 * np.pi) / L_perp_21cm
k_perp_min_StageII = (2 * np.pi) / L_perp_StageII
k_perp_min_HERA = (2 * np.pi) / L_perp_HERA
k_perp_min_EXCLAIM_1 = (2 * np.pi) / L_perp_EXCLAIM_1
#k_perp_min_EXCLAIM_2 = (2 * np.pi) / L_perp_21cm

k_para_min_CCATp = (2 * np.pi) / L_para_21cm
k_para_min_StageII = (2 * np.pi) / L_para_StageII
k_para_min_HERA = (2 * np.pi) / L_para_HERA
k_para_min_EXCLAIM_1 = (2 * np.pi) / L_para_EXCLAIM_1
#k_para_min_EXCLAIM_2 = (2 * np.pi) / L_para_EXCLAIM_2

k_perp_max_CCATp = 1 / sigma_perp_CCATp
k_perp_max_StageII = 1 / sigma_perp_StageII
k_perp_max_HERA = 1 / sigma_perp_HERA
#k_perp_max_EXCLAIM_1 = 1 / sigma_perp_EXCLAIM_1
#k_perp_max_EXCLAIM_2 = 1 / sigma_perp_EXCLAIM_2

k_para_max_CCATp = 1 / sigma_para_CCATp
k_para_max_StageII = 1 / sigma_para_StageII
k_para_max_HERA = 1 / sigma_para_HERA
#k_perp_max_EXCLAIM_1 = 1 / sigma_para_EXCLAIM_1
#k_perp_max_EXCLAIM_2 = 1 / sigma_para_EXCLAIM_2

def calc_V_surv_generic_check(redshift, nu_rest, Omega_surv, B_nu):
    A_s = Omega_surv
    B_nu = B_nu
    r = Planck15.comoving_distance(redshift)
    y = (const.c * (1 + redshift)**2) / (Planck15.H(z) * nu_rest)

    V = r**2 * y * A_s * B_nu * Planck15.h**3

    return V.to(u.Mpc**3, equivalencies=u.dimensionless_angles())

# Fitting

p_names = np.asarray(['b_i','b_j', 'b_k', 'P_m'])

frac_op = .001
frac_con = .01
frac_pess = .1

var_21cm_21cm = var_x(P_21cm_21cm[:-1], W_k_21cm[:-1], P_21cm_21cm[:-1], W_k_21cm[:-1],
                     P_N_21cm, P_N_21cm, P_21cm_21cm[:-1], N_modes)
var_CII_CII = var_x(P_CII_CII[:-1], W_k_CII[:-1], P_CII_CII[:-1], W_k_CII[:-1],
                     P_N_CII, P_N_CII, P_CII_CII[:-1], N_modes)
var_OIII_OIII = var_x(P_OIII_OIII[:-1], W_k_OIII[:-1], P_OIII_OIII[:-1], W_k_OIII[:-1],
                     P_N_OIII, P_N_OIII, P_OIII_OIII[:-1], N_modes)

var_21cm_CII = var_x(P_21cm_21cm[:-1], W_k_21cm[:-1], P_CII_CII[:-1], W_k_CII[:-1],
                     P_N_21cm, P_N_CII, P_21cm_CII[:-1], N_modes)
var_21cm_OIII = var_x(P_21cm_21cm[:-1], W_k_21cm[:-1], P_OIII_OIII[:-1], W_k_OIII[:-1],
                     P_N_21cm, P_N_OIII, P_21cm_OIII[:-1], N_modes)

#variances = [var_21cm_21cm, var_21cm_CII, var_21cm_OIII, var_CII_OIII, model_params_sf['b_i'] * .25]

print('superfake analysis')

### Superfake data and superfake noise levels

biases_sf = utils.extract_bias(k_indices, spectra_sf[1], P_m)
p_vals_sf = np.asarray([*biases_sf, P_m], dtype=object)

params_sf = dict(zip(p_names, p_vals_sf))
ndim = utils.get_params(params_sf, k_indices).size
model = models.ScalarBias_crossonly(k=spectra_sf[0], params=params_sf)

data_sf_nl, Beane_sf_nl, LSE_sf_nl, MCMC_sf_nl = analysis.run_analysis(k_indices, spectra_sf[1], params_sf,
                                                                N_modes, frac_pess, model, noiseless=True)

data_sf_op, Beane_sf_op, LSE_sf_op, MCMC_sf_op = analysis.run_analysis(k_indices, spectra_sf[1], params_sf,
                                                                N_modes, frac_op, model)

data_sf_con, Beane_sf_con, LSE_sf_con, MCMC_sf_con = analysis.run_analysis(k_indices, spectra_sf[1], params_sf,
                                                                N_modes, frac_con, model)

data_sf_pess, Beane_sf_pess, LSE_sf_pess, MCMC_sf_pess = analysis.run_analysis(k_indices, spectra_sf[1], params_sf,
                                                                N_modes, frac_pess, model)

#analysis.plot_corner('sf_op.pdf', MCMC_sf_op, LSE_sf_op, Beane_sf_op, params_sf, spectra_sf[1][0], k_indices)
#analysis.plot_corner('sf_con.pdf', MCMC_sf_con, LSE_sf_con, Beane_sf_con, params_sf, spectra_sf[1][0], k_indices)
#analysis.plot_corner('sf_pess.pdf', MCMC_sf_pess, LSE_sf_pess, Beane_sf_pess, params_sf, spectra_sf[1][0], k_indices)

np.savez('sf_results_z6.0155', data_sf_nl=data_sf_nl, data_sf_op=data_sf_op, data_sf_con=data_sf_con, data_sf_pess=data_sf_pess,
                    Beane_sf_nl=Beane_sf_nl, Beane_sf_op=Beane_sf_op, Beane_sf_con=Beane_sf_con, Beane_sf_pess=Beane_sf_pess,
                    LSE_sf_nl=LSE_sf_nl, LSE_sf_op=LSE_sf_op, LSE_sf_con=LSE_sf_con, LSE_sf_pess=LSE_sf_pess,
                    MCMC_op_samples=MCMC_sf_op[0], MCMC_con_samples=MCMC_sf_con[0], MCMC_pess_samples=MCMC_sf_pess[0],
                    MCMC_op_logp=MCMC_sf_op[1], MCMC_con_logp=MCMC_sf_con[1], MCMC_pess_logp=MCMC_sf_pess[1],
                    MCMC_nl_samples=MCMC_sf_nl[0], MCMC_nl_logp=MCMC_sf_nl[1])

### Simulated power law data and fractional noise error
print('power law analysis')

biases_pl = utils.extract_bias(k_indices, spectra_pl[1], P_m)
p_vals_pl = np.asarray([*biases_pl, P_m], dtype=object)

params_pl = dict(zip(p_names, p_vals_pl))
ndim = utils.get_params(params_pl, k_indices).size

data_pl_nl, Beane_pl_nl, LSE_pl_nl, MCMC_pl_nl = analysis.run_analysis(k_indices, spectra_pl[1], params_pl,
                                                                N_modes, frac_pess, model, noiseless=True)

data_pl_op, Beane_pl_op, LSE_pl_op, MCMC_pl_op = analysis.run_analysis(k_indices, spectra_pl[1], params_pl,
                                                                N_modes, frac_op, model)

data_pl_con, Beane_pl_con, LSE_pl_con, MCMC_pl_con = analysis.run_analysis(k_indices, spectra_pl[1], params_pl,
                                                                N_modes, frac_con, model)

data_pl_pess, Beane_pl_pess, LSE_pl_pess, MCMC_pl_pess = analysis.run_analysis(k_indices, spectra_pl[1], params_pl,
                                                                N_modes, frac_pess, model)

#analysis.plot_corner('pl_op.pdf', MCMC_pl_op, LSE_pl_op, Beane_pl_op, params_pl, spectra_pl[1][0], k_indices)
#analysis.plot_corner('pl_con.pdf', MCMC_pl_con, LSE_pl_con, Beane_pl_con, params_pl, spectra_pl[1][0], k_indices)
#analysis.plot_corner('pl_pess.pdf', MCMC_pl_pess, LSE_pl_pess, Beane_pl_pess, params_pl, spectra_pl[1][0], k_indices)

np.savez('pl_results_z6.0155', data_pl_nl=data_pl_nl, data_pl_op=data_pl_op, data_pl_con=data_pl_con, data_pl_pess=data_pl_pess,
                    Beane_pl_nl=Beane_pl_nl, Beane_pl_op=Beane_pl_op, Beane_pl_con=Beane_pl_con, Beane_pl_pess=Beane_pl_pess,
                    LSE_pl_nl=LSE_pl_nl, LSE_pl_op=LSE_pl_op, LSE_pl_con=LSE_pl_con, LSE_pl_pess=LSE_pl_pess,
                    MCMC_op_samples=MCMC_pl_op[0], MCMC_con_samples=MCMC_pl_con[0], MCMC_pess_samples=MCMC_pl_pess[0],
                    MCMC_op_logp=MCMC_pl_op[1], MCMC_con_logp=MCMC_pl_con[1], MCMC_pess_logp=MCMC_pl_pess[1],
                    MCMC_nl_samples=MCMC_pl_nl[0], MCMC_nl_logp=MCMC_pl_nl[1])

### Simulated brightness temperature data and fractional noise error
print('brightness temperature analysis')

biases_bt = utils.extract_bias(k_indices, spectra_bt[1], P_m)
p_vals_bt = np.asarray([*biases_bt, P_m], dtype=object)

params_bt = dict(zip(p_names, p_vals_bt))
ndim = utils.get_params(params_bt, k_indices).size

data_bt_nl, Beane_bt_nl, LSE_bt_nl, MCMC_bt_nl = analysis.run_analysis(k_indices, spectra_bt[1], params_bt,
                                                                N_modes, frac_pess, model, noiseless=True)

data_bt_op, Beane_bt_op, LSE_bt_op, MCMC_bt_op = analysis.run_analysis(k_indices, spectra_bt[1], params_bt,
                                                                N_modes, frac_op, model)

data_bt_con, Beane_bt_con, LSE_bt_con, MCMC_bt_con = analysis.run_analysis(k_indices, spectra_bt[1], params_bt,
                                                                N_modes, frac_con, model)

data_bt_pess, Beane_bt_pess, LSE_bt_pess, MCMC_bt_pess = analysis.run_analysis(k_indices, spectra_bt[1], params_bt,
                                                                N_modes, frac_pess, model)

#analysis.plot_corner('bt_op.pdf', MCMC_bt_op, LSE_bt_op, Beane_bt_op, params_bt, spectra_bt[1][0], k_indices)
#analysis.plot_corner('bt_con.pdf', MCMC_bt_con, LSE_bt_con, Beane_bt_con, params_bt, spectra_bt[1][0], k_indices)
#analysis.plot_corner('bt_pess.pdf', MCMC_bt_pess, LSE_bt_pess, Beane_bt_pess, params_bt, spectra_bt[1][0], k_indices)

np.savez('bt_results_z6.0155', data_bt_nl=data_bt_nl, data_bt_op=data_bt_op, data_bt_con=data_bt_con, data_bt_pess=data_bt_pess,
                    Beane_bt_nl=Beane_bt_nl, Beane_bt_op=Beane_bt_op, Beane_bt_con=Beane_bt_con, Beane_bt_pess=Beane_bt_pess,
                    LSE_bt_nl=LSE_bt_nl, LSE_bt_op=LSE_bt_op, LSE_bt_con=LSE_bt_con, LSE_bt_pess=LSE_bt_pess,
                    MCMC_op_samples=MCMC_bt_op[0], MCMC_con_samples=MCMC_bt_con[0], MCMC_pess_samples=MCMC_bt_pess[0],
                    MCMC_op_logp=MCMC_bt_op[1], MCMC_con_logp=MCMC_bt_con[1], MCMC_pess_logp=MCMC_bt_pess[1],
                    MCMC_nl_samples=MCMC_bt_nl[0], MCMC_nl_logp=MCMC_bt_nl[1])

### Fisher analysis
