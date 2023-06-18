import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import copy as cp
import time
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
from astropy.cosmology import Planck15, Planck18

import analysis
import signals
import estimators
import fitting
import models
import utils
import survey
import survey_list

plt.style.use('seaborn-colorblind')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# Simulations

print('loading simulations')

which_box = 'little'
print('running analysis on', which_box, 'box')

if which_box == 'little':
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

if which_box == 'big':
#     rez = 1024
    box = h5py.File('halos.z8.hdf5', 'r')
    print(box.keys())

    density = np.fromfile('sims/rho.z=07.9589_cic_1024', dtype=np.float64).reshape(rez, rez, rez, order='F')

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

#print('generating underlying matter density spectrum')
print('loading underlying matter density spectrum')

#delta = utils.overdensity(density)
#k, P_m = analysis.calc_pspec(r_vec, [delta], n_bins=n_bins, bin_scale='log')
#np.savez('matter_pspec_6.0155', k=k, P_m=P_m)

# matter_pspec = np.load('/home/mcbrie2/projects/def-acliu/mcbrie2/deprism/spectra/matter_pspec_6.0155.npz')
matter_pspec = np.load(f'spectra/matter_pspec_z{redshift}.npz')
k = matter_pspec['k']
P_m = matter_pspec['P_m']

print('yay! finished the matter stuff')

### Datasets

# pspecs_sf = np.load('spectra/pspecs_sf_z6.0155.npy')
# pspecs_pl = np.load('pspecs_pl.npz')
# pspecs_bt = np.load('pspecs_bt.npz')
# pspecs_bt.files
#
spectra_sf = np.load(f'spectra_all_int/spectra_sf_z{redshift}.npy')
spectra_pl = np.load(f'spectra_all_int/spectra_pl_z{redshift}.npy')
spectra_bt = np.load(f'spectra_all_int/spectra_bt_z{redshift}.npy')

# just the above in npz format
pspecs_sf = np.load('spectra_all_int/pspecs_sf_z6.0155.npz')
pspecs_pl = np.load('spectra_all_int/pspecs_pl_z6.0155.npz')
pspecs_bt = np.load('spectra_all_int/pspecs_bt_z6.0155.npz')

# Fitting
p_names = np.asarray(['b_i','b_j', 'b_k', 'P_m'])
k_indices = [6]
k_units = k * Planck18.h / u.Mpc

# Survey specifications now
# set by nature
lambda_OIII = 88 * u.um # micrometers
lambda_CII = 158 * u.um
lambda_21cm = 21 * u.cm # centimeters, of course!

# nu_21cm = 1420 * u.MHz
# nu_CII = 1.900539e12 * u.Hz # Hz
# nu_CO = 115.271203e9 * u.Hz # Hz
# nu_OIII = 88 * u.Hz # Hz

nu_21cm = lambda_21cm.to(u.MHz, equivalencies=u.spectral())
nu_CII = lambda_CII.to(u.GHz, equivalencies=u.spectral()) #158 um micrometers
nu_OIII = lambda_OIII.to(u.GHz, equivalencies=u.spectral())

#lambda_CO = nu_CO.to(u.mm, equivalencies=u.spectral())
# surveys
PAPER = survey_list.specs_PAPER
HERA = survey_list.specs_HERA
HERA_future = survey_list.specs_HERA_future

StageII = survey_list.specs_StageII
StageIII = survey_list.specs_StageIII
CII_future = survey_list.specs_CII_future

EXCLAIM = survey_list.specs_EXCLAIM
OIII_future = survey_list.specs_OIII_future

# extents
L_21cm_PAPER, k_21cm_PAPER = survey.calc_survey_extents(PAPER, redshift, lambda_21cm)
L_21cm_HERA, k_21cm_HERA = survey.calc_survey_extents(HERA, redshift, lambda_21cm)
L_CII_StageII, k_CII_StageII = survey.calc_survey_extents(StageII, redshift, lambda_CII)
L_CII_StageIII, k_CII_StageIII = survey.calc_survey_extents(StageIII, redshift, lambda_CII)
L_OIII_EXCLAIM, k_OIII_EXCLAIM = survey.calc_survey_extents(EXCLAIM, redshift, lambda_OIII)

L_21cm_future, k_21cm_future = survey.calc_survey_extents(HERA_future, redshift, lambda_21cm)
L_CII_future, k_CII_future = survey.calc_survey_extents(CII_future, redshift, lambda_CII)
L_OIII_future, k_OIII_future = survey.calc_survey_extents(OIII_future, redshift, lambda_OIII)

# volumes
V_21cm_HERA = survey.calc_V_survey(HERA, redshift, lambda_21cm)
V_CII_StageII = survey.calc_V_survey(StageII, redshift, lambda_CII)
V_CII_StageIII = survey.calc_V_survey(StageIII, redshift, lambda_CII)
V_OIII_EXCLAIM = survey.calc_V_survey(EXCLAIM, redshift, lambda_OIII)

V_21cm_future = survey.calc_V_survey(HERA_future, redshift, lambda_21cm)
V_CII_future = survey.calc_V_survey(CII_future, redshift, lambda_CII)
V_OIII_future = survey.calc_V_survey(OIII_future, redshift, lambda_OIII)

# instrument noises
P_N_21cm_HERA = survey.calc_P_N_21cm(HERA, k_units, redshift)
P_N_CII_StageII = survey.calc_P_N(StageII, redshift, lambda_CII)
P_N_CII_StageIII = survey.calc_P_N(StageIII, redshift, lambda_CII)
P_N_OIII_EXCLAIM = survey.calc_P_N(EXCLAIM, redshift, lambda_OIII)

P_N_21cm_future = survey.calc_P_N_21cm(HERA_future, k_units, redshift)
P_N_CII_future = survey.calc_P_N(CII_future, redshift, lambda_CII)
P_N_OIII_future = survey.calc_P_N(OIII_future, redshift, lambda_OIII)

# N modes
N_modes_21cm_HERA = survey.calc_N_modes(k, V_21cm_HERA, align='left')
N_modes_CII_StageII = survey.calc_N_modes(k, V_CII_StageII, align='left')
N_modes_CII_StageIII = survey.calc_N_modes(k, V_CII_StageIII, align='left')
N_modes_OIII_EXCLAIM = survey.calc_N_modes(k, V_OIII_EXCLAIM, align='left')

N_modes_21cm_future = survey.calc_N_modes(k, V_21cm_future, align='left')
N_modes_CII_future = survey.calc_N_modes(k, V_CII_future, align='left')
N_modes_OIII_future = survey.calc_N_modes(k, V_OIII_future, align='left')

# Window functions
W_21cm_HERA = survey.calc_W(k, k_21cm_HERA, HERA, redshift, lambda_21cm)
W_CII_StageII = survey.calc_W(k, k_CII_StageII, StageII, redshift, lambda_CII)
W_CII_StageIII = survey.calc_W(k, k_CII_StageIII, StageII, redshift, lambda_CII)
W_OIII_EXCLAIM = survey.calc_W(k, k_OIII_EXCLAIM, EXCLAIM, redshift, lambda_OIII)

W_21cm_future = survey.calc_W(k, k_21cm_future, HERA_future, redshift, lambda_21cm)
W_CII_future = survey.calc_W(k, k_CII_future, CII_future, redshift, lambda_CII)
W_OIII_future = survey.calc_W(k, k_OIII_future, OIII_future, redshift, lambda_OIII)

# Variances
# autos
var_21cm_21cm_HERA = survey.var_x(utils.dimless(k_units[:-1],
                                    pspecs_sf['P_21cm_21cm'][:-1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                  utils.dimless(k_units[:-1],
                                    pspecs_sf['P_21cm_21cm'][:-1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                  P_N_21cm_HERA[:-1],
                                  P_N_21cm_HERA[:-1],
                                  utils.dimless(k_units[:-1],
                                                pspecs_sf['P_21cm_21cm'][:-1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                  N_modes_21cm_HERA,
                                  W_i=W_21cm_HERA[:-1],
                                  W_j=W_21cm_HERA[:-1])

var_21cm_21cm_future = survey.var_x(utils.dimless(k_units[:-1],
                                     pspecs_sf['P_21cm_21cm'][:-1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                    utils.dimless(k_units[:-1],
                                     pspecs_sf['P_21cm_21cm'][:-1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                    P_N_21cm_future[:-1],
                                    P_N_21cm_future[:-1],
                                    utils.dimless(k_units[:-1],
                                     pspecs_sf['P_21cm_21cm'][:-1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                    N_modes_21cm_future,
                                    W_i=W_21cm_future[:-1],
                                    W_j=W_21cm_future[:-1])

var_CII_CII_StageII = survey.var_x(utils.dimless(k_units[:-1],
                                        pspecs_sf['P_CII_CII'][:-1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                   utils.dimless(k_units[:-1],
                                        pspecs_sf['P_CII_CII'][:-1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                   utils.dimless(k_units[:-1], P_N_CII_StageII),
                                   utils.dimless(k_units[:-1], P_N_CII_StageII),
                                   utils.dimless(k_units[:-1],
                                        pspecs_sf['P_CII_CII'][:-1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                   N_modes_CII_StageII,
                                   W_i=W_CII_StageII[:-1],
                                   W_j=W_CII_StageII[:-1])

var_CII_CII_StageIII = survey.var_x(utils.dimless(k_units[:-1],
                                        pspecs_sf['P_CII_CII'][:-1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                    utils.dimless(k_units[:-1],
                                        pspecs_sf['P_CII_CII'][:-1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                    utils.dimless(k_units[:-1], P_N_CII_StageIII),
                                    utils.dimless(k_units[:-1], P_N_CII_StageIII),
                                    utils.dimless(k_units[:-1],
                                        pspecs_sf['P_CII_CII'][:-1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                    N_modes_CII_StageIII,
                                    W_i=W_CII_StageIII[:-1],
                                    W_j=W_CII_StageIII[:-1])

var_CII_CII_future = survey.var_x(utils.dimless(k_units[:-1],
                                        pspecs_sf['P_CII_CII'][:-1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                  utils.dimless(k_units[:-1],
                                        pspecs_sf['P_CII_CII'][:-1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                  utils.dimless(k_units[:-1], P_N_CII_future),
                                  utils.dimless(k_units[:-1], P_N_CII_future),
                                  utils.dimless(k_units[:-1],
                                        pspecs_sf['P_CII_CII'][:-1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                  N_modes_CII_future,
                                  W_i=W_CII_future[:-1],
                                  W_j=W_CII_future[:-1])

var_OIII_OIII_EXCLAIM = survey.var_x(utils.dimless(k_units[:-1],
                                        pspecs_sf['P_OIII_OIII'][:-1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                     utils.dimless(k_units[:-1],
                                        pspecs_sf['P_OIII_OIII'][:-1]* u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                     utils.dimless(k_units[:-1], P_N_OIII_EXCLAIM),
                                     utils.dimless(k_units[:-1], P_N_OIII_EXCLAIM),
                                     utils.dimless(k_units[:-1],
                                        pspecs_sf['P_OIII_OIII'][:-1]* u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                     N_modes_OIII_EXCLAIM,
                                     W_i=W_OIII_EXCLAIM[:-1],
                                     W_j=W_OIII_EXCLAIM[:-1])

var_OIII_OIII_future = survey.var_x(utils.dimless(k_units[:-1],
                                        pspecs_sf['P_OIII_OIII'][:-1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                    utils.dimless(k_units[:-1],
                                        pspecs_sf['P_OIII_OIII'][:-1]* u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                    utils.dimless(k_units[:-1], P_N_OIII_future),
                                    utils.dimless(k_units[:-1], P_N_OIII_future),
                                    utils.dimless(k_units[:-1],
                                        pspecs_sf['P_OIII_OIII'][:-1]* u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                    N_modes_OIII_future,
                                    W_i=W_OIII_future[:-1],
                                    W_j=W_OIII_future[:-1])

# crosses
var_21cm_CII = survey.var_x(utils.dimless(k_units[:-1],
                                pspecs_sf['P_21cm_21cm'][:-1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                            utils.dimless(k_units[:-1],
                                pspecs_sf['P_CII_CII'][:-1]* u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                  P_N_21cm_HERA[1:],
                                  utils.dimless(k_units[:-1], P_N_CII_StageII),
                            utils.dimless(k_units[:-1],
                                pspecs_sf['P_21cm_CII'][:-1]* u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                  N_modes_21cm_HERA,
                                  W_i=W_21cm_HERA[:-1],
                                  W_j=W_CII_StageII[:-1])

var_CII_OIII = survey.var_x(utils.dimless(k_units[:-1],
                                pspecs_sf['P_CII_CII'][:-1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                            utils.dimless(k_units[:-1],
                                pspecs_sf['P_OIII_OIII'][:-1]* u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                  utils.dimless(k_units[:-1], P_N_CII_StageII),
                                  utils.dimless(k_units[:-1], P_N_OIII_EXCLAIM),
                            utils.dimless(k_units[:-1],
                                pspecs_sf['P_CII_OIII'][:-1]* u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                  N_modes_21cm_HERA,
                                  W_i=W_CII_StageII[:-1],
                                  W_j=W_OIII_EXCLAIM[:-1])

var_21cm_CII_StageIII = survey.var_x(utils.dimless(k_units[:-1],
                                pspecs_sf['P_21cm_21cm'][:-1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                            utils.dimless(k_units[:-1],
                                pspecs_sf['P_CII_CII'][:-1]* u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                  P_N_21cm_HERA[1:],
                                  utils.dimless(k_units[:-1], P_N_CII_StageIII),
                            utils.dimless(k_units[:-1],
                                pspecs_sf['P_21cm_CII'][:-1]* u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                  N_modes_21cm_HERA,
                                  W_i=W_21cm_HERA[:-1],
                                  W_j=W_CII_StageIII[:-1])

var_CII_OIII_StageIII = survey.var_x(utils.dimless(k_units[:-1],
                                pspecs_sf['P_CII_CII'][:-1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                            utils.dimless(k_units[:-1],
                                pspecs_sf['P_OIII_OIII'][:-1]* u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                  utils.dimless(k_units[:-1], P_N_CII_StageIII),
                                  utils.dimless(k_units[:-1], P_N_OIII_EXCLAIM),
                            utils.dimless(k_units[:-1],
                                pspecs_sf['P_CII_OIII'][:-1]* u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                  N_modes_21cm_HERA,
                                  W_i=W_CII_StageIII[:-1],
                                  W_j=W_OIII_EXCLAIM[:-1])

var_21cm_OIII = survey.var_x(utils.dimless(k_units[:-1],
                                pspecs_sf['P_21cm_21cm'][:-1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                            utils.dimless(k_units[:-1],
                                pspecs_sf['P_OIII_OIII'][:-1]* u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                  P_N_21cm_HERA[1:],
                                  utils.dimless(k_units[:-1], P_N_OIII_EXCLAIM),
                            utils.dimless(k_units[:-1],
                                pspecs_sf['P_21cm_OIII'][:-1]* u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                  N_modes_21cm_HERA,
                                  W_i=W_21cm_HERA[:-1],
                                  W_j=W_OIII_EXCLAIM[:-1])

var_21cm_CII_future = survey.var_x(utils.dimless(k_units[:-1],
                                pspecs_sf['P_21cm_21cm'][:-1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                            utils.dimless(k_units[:-1],
                                pspecs_sf['P_CII_CII'][:-1]* u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                  P_N_21cm_future[1:],
                                  utils.dimless(k_units[:-1], P_N_CII_future),
                            utils.dimless(k_units[:-1],
                                pspecs_sf['P_21cm_CII'][:-1]* u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                  N_modes_21cm_HERA,
                                  W_i=W_21cm_future[:-1],
                                  W_j=W_CII_future[:-1])

var_CII_OIII_future = survey.var_x(utils.dimless(k_units[:-1],
                                pspecs_sf['P_CII_CII'][:-1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                            utils.dimless(k_units[:-1],
                                pspecs_sf['P_OIII_OIII'][:-1]* u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                  utils.dimless(k_units[:-1], P_N_CII_future),
                                  utils.dimless(k_units[:-1], P_N_OIII_future),
                            utils.dimless(k_units[:-1],
                                pspecs_sf['P_CII_OIII'][:-1]* u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                  N_modes_21cm_HERA,
                                  W_i=W_CII_future[:-1],
                                  W_j=W_OIII_future[:-1])

var_21cm_OIII_future = survey.var_x(utils.dimless(k_units[:-1],
                                pspecs_sf['P_21cm_21cm'][:-1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                            utils.dimless(k_units[:-1],
                                pspecs_sf['P_OIII_OIII'][:-1]* u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                  P_N_21cm_future[1:],
                                  utils.dimless(k_units[:-1], P_N_OIII_future),
                            utils.dimless(k_units[:-1],
                                pspecs_sf['P_21cm_OIII'][:-1]* u.Mpc**3 * u.Jy**2 * u.steradian**(-2)),
                                  N_modes_21cm_HERA,
                                  W_i=W_21cm_future[:-1],
                                  W_j=W_OIII_future[:-1])

### Superfake data and superfake noise levels

# print('superfake temperature analysis')
biases_sf = utils.extract_bias(k_indices, spectra_sf, P_m)
p_vals_sf = np.asarray([*biases_sf, P_m], dtype=object)

params_sf = dict(zip(p_names, p_vals_sf))
ndim = utils.get_params(params_sf, k_indices).size
model = models.ScalarBias_crossonly(k=spectra_sf[0], params=params_sf)
N_modes_small = survey.calc_N_modes(k, 80**3 * u.Mpc**3, align='left')

for i in range(1,k.size-1):
    t0 = time.time()
    print('Now on k-mode',k[i],'%')
    k_indices = [i]
    nsteps = int(1e6)
    n = [var_21cm_21cm_HERA, var_21cm_CII, var_21cm_OIII,
         var_CII_CII_StageIII, var_CII_OIII, var_OIII_OIII_EXCLAIM]
    # if n > .1:
    #     nsteps = int(1e7)

    data_nl, Beane_nl, LSE_nl, MCMC_nl = analysis.keep_P_21(k_indices, spectra_sf, params_sf, n, model,
                                            N_modes=N_modes_small, noiseless=True, nsteps=nsteps,
                                            backend_filename=f'survey_current_kmode{k[i]}_sf_nl_z{redshift}_int.h5',
                                            error_x=False)
    data, Beane, LSE, MCMC = analysis.keep_P_21(k_indices, spectra_sf, params_sf, n, model,
                                            N_modes=N_modes_small, noiseless=False, nsteps=nsteps,
                                            backend_filename=f'survey_current_kmode{k[i]}_sf_z{redshift}_int.h5',
                                            error_x=False)


    np.savez(f'results_all_int/sf_fits/survey_k{k[i]}_sf_nl_z{redshift}_int', data=data_nl, Beane=Beane_nl, LSE=LSE_nl,
                                        samples=MCMC_nl[0], logp=MCMC_nl[1])
    np.savez(f'results_all_int/sf_fits/survey_k{k[i]}_sf_z{redshift}_int', data=data, Beane=Beane, LSE=LSE,
                                        samples=MCMC[0], logp=MCMC[1])

    tf = time.time()
    print(f'run {i} saved to disk')
    print('time to complete superfake analysis run {i} is:', (tf - t0) / 60 / 60, 'hours')

# ### Simulated power law data and fractional noise error
# print('power law analysis')
#
# biases_pl = utils.extract_bias(k_indices, spectra_pl, P_m)
# p_vals_pl = np.asarray([*biases_pl, P_m], dtype=object)
#
# params_pl = dict(zip(p_names, p_vals_pl))
# ndim = utils.get_params(params_pl, k_indices).size
# model = models.ScalarBias_crossonly(k=spectra_pl[0], params=params_pl)
# N_modes_small = survey.calc_N_modes(k, 80**3 * u.Mpc**3, align='left')
#
# for i, n in enumerate(noise):
#     t0 = time.time()
#     print('Now on noise level',n,'%')
#     nsteps = int(1e6)
#     if n > .1:
#         nsteps = int(1e7)
#
#     # data_nl, Beane_nl, LSE_nl, MCMC_nl = analysis.keep_P_21(k_indices, spectra_pl, params_pl, n, model,
#     #                                         N_modes=N_modes_small, noiseless=True, nsteps=nsteps,
#     #                                         backend_filename=f'noise{n}_pl_nl_z{redshift}_int.h5')
#     data, Beane, LSE, MCMC = analysis.keep_P_21(k_indices, spectra_pl, params_pl, n, model,
#                                             N_modes=N_modes_small, noiseless=False, nsteps=nsteps,
#                                             backend_filename=f'noise{n}_pl_z{redshift}_int.h5')
#
#
#     # np.savez(f'results_noises/pl_fits/noise{n}_pl_nl_z{redshift}_int', data=data_nl, Beane=Beane_nl, LSE=LSE_nl,
#     #                                     samples=MCMC_nl[0], logp=MCMC_nl[1])
#     np.savez(f'results_noises/pl_fits/noise{n}_pl_z{redshift}_int', data=data, Beane=Beane, LSE=LSE,
#                                         samples=MCMC[0], logp=MCMC[1])
#
#
#     tf = time.time()
#     print(f'run {i} saved to disk')
#     print('time to complete power law run {i} is:', (tf - t0) / 60, 'minutes')

# ### Simulated brightness temperature data and fractional noise error
# print('brightness temperature analysis')
#
# biases_bt = utils.extract_bias(k_indices, spectra_bt, P_m)
# p_vals_bt = np.asarray([*biases_bt, P_m], dtype=object)
#
# params_bt = dict(zip(p_names, p_vals_bt))
# ndim = utils.get_params(params_bt, k_indices).size
# model = models.ScalarBias_crossonly(k=spectra_bt[0], params=params_bt)
# N_modes_small = survey.calc_N_modes(k, 80**3 * u.Mpc**3, align='left')
#
# for i, n in enumerate(noise):
#     t0 = time.time()
#     print('Now on noise level',n,'%')
#     nsteps = int(1e6)
#     if n > .1:
#         nsteps = int(1e7)
#
#     data_nl, Beane_nl, LSE_nl, MCMC_nl = analysis.keep_P_21(k_indices, spectra_bt, params_bt, n, model,
#                                             N_modes=N_modes_small, noiseless=True, nsteps=nsteps,
#                                             backend_filename=f'noise{n}_bt_nl_z{redshift}_int.h5')
#     data, Beane, LSE, MCMC = analysis.keep_P_21(k_indices, spectra_bt, params_bt, n, model,
#                                             N_modes=N_modes_small, noiseless=False, nsteps=nsteps,
#                                             backend_filename=f'noise{n}_bt_z{redshift}_int.h5')
#
#
#     np.savez(f'results_all_int/bt_fits/noise{n}_bt_nl_z{redshift}_int', data=data_nl, Beane=Beane_nl, LSE=LSE_nl,
#                                         samples=MCMC_nl[0], logp=MCMC_nl[1])
#     np.savez(f'results_all_int/bt_fits/noise{n}_bt_z{redshift}_int', data=data, Beane=Beane, LSE=LSE,
#                                         samples=MCMC[0], logp=MCMC[1])
#
#
#     tf = time.time()
#     print(f'run {i} saved to disk')
#     print(f'time to complete brightness temperature run {i} is:', (tf - t0) / 60, 'minutes')
#
# # # ### Fisher analysis
