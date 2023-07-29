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
    box = h5py.File('sims/L80_halos_z=6.0155.hdf5', 'r')
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
    box = h5py.File('sims/halos.z8.hdf5', 'r')
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

HERA_spacing = survey.calc_spacing(L_21cm_HERA, k_21cm_HERA)
FYST_spacing = survey.calc_spacing(L_CII_StageII, k_CII_StageII)
EXCLAIM_spacing = survey.calc_spacing(L_OIII_EXCLAIM, k_OIII_EXCLAIM)

L_joint = [L_21cm_HERA[0], L_CII_StageII[1],
              L_OIII_EXCLAIM[2], L_21cm_HERA[3]]

k_joint = [k_CII_StageII[0], k_21cm_HERA[1],
              k_21cm_HERA[2], k_OIII_EXCLAIM[3]]

joint_spacing = survey.calc_spacing(L_joint, k_joint)

# load all the HERA noise
data_HERA = np.loadtxt('HERA_optimisticnoise.csv', delimiter=',')

k_HERA = data_HERA[:,0]

nu_21cm_rest = 1420 * u.MHz
nu_21cm_obvs = utils.calc_nu_obs(nu_21cm_rest, redshift)

P21_HERA = ((np.sqrt(data_HERA[:,1]) * u.mK).to(u.Jy / u.steradian,
                      equivalencies=u.brightness_temperature(nu_21cm_obvs)))**2

noise_HERA = P21_HERA / data_HERA[:,2]

print('loaded HERA data...')
spectra_sf_interp = np.zeros((6, k_HERA[1:].size))
for i in range(6):
    spec_interp = np.interp(k_HERA[1:],
                            k[1:-1], utils.dimless(k[1:-1], spectra_sf[i][1:-1]))
    spectra_sf_interp[i] = spec_interp

# calculating k-modes
k_range_sphere = []
k_range_sphere.append(np.concatenate((-np.flip(joint_spacing[0][1:]), joint_spacing[0])))
k_range_sphere.append(np.concatenate((-np.flip(joint_spacing[0][1:]), joint_spacing[0])))
k_range_sphere.append(np.concatenate((-np.flip(joint_spacing[1][1:]), joint_spacing[1])))

K_grid = np.zeros((k_range_sphere[0].size, k_range_sphere[1].size, k_range_sphere[2].size))

for x_i, x in enumerate(k_range_sphere[0]):
    for y_i,y in enumerate(k_range_sphere[1]):
        for z_i,z in enumerate(k_range_sphere[2]):
            K_grid[x_i,y_i,z_i] = np.sqrt(x**2 + y**2 + z**2)

print('min, max k:', K_grid.min(), K_grid.max())

# volumes
V_21cm_HERA = survey.calc_V_survey(HERA, redshift, lambda_21cm)
V_CII_StageII = survey.calc_V_survey(StageII, redshift, lambda_CII)
V_CII_StageIII = survey.calc_V_survey(StageIII, redshift, lambda_CII)
V_OIII_EXCLAIM = survey.calc_V_survey(EXCLAIM, redshift, lambda_OIII)

V_21cm_future = survey.calc_V_survey(HERA_future, redshift, lambda_21cm)
V_CII_future = survey.calc_V_survey(CII_future, redshift, lambda_CII)
V_OIII_future = survey.calc_V_survey(OIII_future, redshift, lambda_OIII)

# N modes
N_modes_joint, N_modes_bins = np.histogram(K_grid, bins=k_HERA)

# some other initalization stuff because of HERA data_nl# resize for survey section
k_units = k_HERA[1:] / u.Mpc
noise_HERA = noise_HERA[1:]

P_N_21cm_PAPER = noise_HERA
P_N_21cm_HERA = noise_HERA
P_N_21cm_future = noise_HERA

spectra_units = spectra_sf_interp * u.Jy**2 * u.steradian**(-2)

# instrument noises
P_N_CII_StageII = utils.dimless(k_units, survey.calc_P_N(StageII, redshift,
                                        lambda_CII)) / np.sqrt(N_modes_joint)
P_N_CII_StageIII = utils.dimless(k_units,
                                survey.calc_P_N(StageIII, redshift,
                                        lambda_CII)) / np.sqrt(N_modes_joint)
P_N_CII_future = utils.dimless(k_units,
                                survey.calc_P_N(CII_future, redshift,
                                        lambda_CII)) / np.sqrt(N_modes_joint)

P_N_OIII_EXCLAIM = utils.dimless(k_units,
                                 survey.calc_P_N(EXCLAIM, redshift,
                                        lambda_OIII)) / np.sqrt(N_modes_joint)
P_N_OIII_future = utils.dimless(k_units,
                                survey.calc_P_N(OIII_future, redshift,
                                        lambda_OIII)) / np.sqrt(N_modes_joint)
                                        
# Window functions
W_21cm_HERA = survey.calc_W(k_HERA[1:], k_21cm_HERA, HERA, redshift, lambda_21cm)
W_CII_StageII = survey.calc_W(k_HERA[1:], k_CII_StageII, StageII, redshift, lambda_CII)
W_CII_StageIII = survey.calc_W(k_HERA[1:], k_CII_StageIII, StageII, redshift, lambda_CII)
W_OIII_EXCLAIM = survey.calc_W(k_HERA[1:], k_OIII_EXCLAIM, EXCLAIM, redshift, lambda_OIII)

W_21cm_future = survey.calc_W(k_HERA[1:], k_21cm_future, HERA_future, redshift, lambda_21cm)
W_CII_future = survey.calc_W(k_HERA[1:], k_CII_future, CII_future, redshift, lambda_CII)
W_OIII_future = survey.calc_W(k_HERA[1:], k_OIII_future, OIII_future, redshift, lambda_OIII)

#===============
# Variances
#===============
# already dimensionless here
P_21cm_21cm = spectra_units[0]
P_21cm_CII = spectra_units[1]
P_21cm_OIII = spectra_units[2]
P_CII_CII = spectra_units[3]
P_CII_OIII = spectra_units[4]
P_OIII_OIII = spectra_units[5]

#============
# autos
#============
# 21cm
var_21cm_21cm_PAPER = survey.var_x(P_21cm_21cm, P_21cm_21cm,
                                    P_N_21cm_PAPER, P_N_21cm_PAPER,
                                    P_21cm_21cm,
                                    N_modes_joint,
                                    W_i=W_21cm_HERA,
                                    W_j=W_21cm_HERA)

var_21cm_21cm_HERA = survey.var_x(P_21cm_21cm, P_21cm_21cm,
                                    P_N_21cm_HERA, P_N_21cm_HERA,
                                    P_21cm_21cm,
                                    N_modes_joint,
                                    W_i=W_21cm_HERA,
                                    W_j=W_21cm_HERA)

var_21cm_21cm_future = survey.var_x(P_21cm_21cm, P_21cm_21cm,
                                    P_N_21cm_future, P_N_21cm_future,
                                    P_21cm_21cm,
                                    N_modes_joint,
                                    W_i=W_21cm_future,
                                    W_j=W_21cm_future)

# C[II]
var_CII_CII_StageII = survey.var_x(P_CII_CII, P_CII_CII,
                                    P_N_CII_StageII, P_N_CII_StageIII,
                                    P_CII_CII,
                                    N_modes_joint,
                                    W_i=W_CII_StageII,
                                    W_j=W_CII_StageII)

var_CII_CII_StageIII = survey.var_x(P_CII_CII, P_CII_CII,
                                    P_N_CII_StageIII,
                                    P_N_CII_StageIII,
                                    P_CII_CII,
                                    N_modes_joint,
                                    W_i=W_CII_StageIII,
                                    W_j=W_CII_StageIII)

var_CII_CII_future = survey.var_x(P_CII_CII, P_CII_CII,
                                    P_N_CII_future, P_N_CII_future,
                                    P_CII_CII,
                                    N_modes_joint,
                                    W_i=W_CII_future,
                                    W_j=W_CII_future)

# O[III]
var_OIII_OIII_EXCLAIM = survey.var_x(P_OIII_OIII, P_OIII_OIII,
                                        P_N_OIII_EXCLAIM,
                                        P_N_OIII_EXCLAIM,
                                        P_OIII_OIII,
                                        N_modes_joint,
                                        W_i=W_OIII_EXCLAIM,
                                        W_j=W_OIII_EXCLAIM)

var_OIII_OIII_future = survey.var_x(P_OIII_OIII, P_OIII_OIII,
                                        P_N_OIII_EXCLAIM,
                                        P_N_OIII_EXCLAIM,
                                        P_OIII_OIII,
                                        N_modes_joint,
                                        W_i=W_OIII_EXCLAIM,
                                        W_j=W_OIII_EXCLAIM)
#===========
# crosses
#===========
# current
var_21cm_CII = survey.var_x(P_21cm_21cm, P_CII_CII,
                            P_N_21cm_HERA, P_N_CII_StageII,
                            P_21cm_CII,
                            N_modes_joint,
                            W_i=W_21cm_HERA,
                            W_j=W_CII_StageII)


var_CII_OIII = survey.var_x(P_CII_CII, P_OIII_OIII,
                            P_N_CII_StageII, P_N_OIII_EXCLAIM,
                            P_CII_OIII,
                            N_modes_joint,
                            W_i=W_CII_StageII,
                            W_j=W_OIII_EXCLAIM)

var_21cm_CII_StageIII = survey.var_x(P_21cm_21cm, P_CII_CII,
                            P_N_21cm_HERA, P_N_CII_StageIII,
                            P_21cm_CII,
                            N_modes_joint,
                            W_i=W_21cm_HERA,
                            W_j=W_CII_StageIII)

var_CII_OIII_StageIII = survey.var_x(P_CII_CII, P_OIII_OIII,
                            P_N_CII_StageIII, P_N_OIII_EXCLAIM,
                            P_CII_OIII,
                            N_modes_joint,
                            W_i=W_CII_StageIII,
                            W_j=W_OIII_EXCLAIM)

var_21cm_OIII = survey.var_x(P_21cm_21cm, P_OIII_OIII,
                            P_N_21cm_HERA, P_N_OIII_EXCLAIM,
                            P_21cm_OIII,
                            N_modes_joint,
                            W_i=W_21cm_HERA,
                            W_j=W_OIII_EXCLAIM)

# future
var_21cm_CII_future = survey.var_x(P_21cm_21cm, P_CII_CII,
                                    P_N_21cm_future, P_N_CII_future,
                                    P_21cm_CII,
                                    N_modes_joint,
                                    W_i=W_21cm_future,
                                    W_j=W_CII_future)

var_CII_OIII_future = survey.var_x(P_CII_CII, P_OIII_OIII,
                                  P_N_CII_future, P_N_OIII_future,
                                  P_CII_OIII,
                                  N_modes_joint,
                                  W_i=W_CII_future,
                                  W_j=W_OIII_future)

var_21cm_OIII_future = survey.var_x(P_21cm_21cm, P_OIII_OIII,
                                    P_N_21cm_future, P_N_OIII_future,
                                    P_21cm_OIII,
                                    N_modes_joint,
                                    W_i=W_21cm_future,
                                    W_j=W_OIII_future)


### Superfake data and superfake noise levels
P_m_interp = np.interp(k_HERA[1:], k, utils.dimless(k, P_m))

for i in range(k_units.value.size):
    t0 = time.time()
    k_indices = [i]
    print('Now on k-mode k=',k[k_indices])

    biases_sf = utils.extract_bias(k_indices, spectra_sf_interp, P_m_interp)
    p_vals_sf = np.asarray([*biases_sf, P_m_interp], dtype=object)
    params_sf = dict(zip(p_names, p_vals_sf))
    ndim = utils.get_params(params_sf, k_indices).size


    p_sf_tot = np.zeros(ndim+1)
    for j in range(ndim):
        p_sf_tot[j] = utils.get_params(params_sf, k_indices)[j]
    p_sf_tot[-1] = spectra_sf_interp[0][k_indices]

    #     np.set_printoptions(precision=6, suppress=True)
    print(f'parameters are', p_sf_tot)
    model = models.ScalarBias_crossonly(k=k_HERA, params=params_sf)

    nsteps = int(1e6)
    n = [var_21cm_21cm_HERA, var_21cm_CII, var_21cm_OIII,
            var_CII_CII_StageIII, var_CII_OIII, var_OIII_OIII_EXCLAIM]

    data_nl, Beane_nl, LSE_nl, MCMC_nl, N = analysis.keep_P_21(k_indices, spectra_sf_interp, params_sf,
                                            n, model,
                                            N_modes=N_modes_joint, noiseless=True, nsteps=nsteps,
                                            backend_filename=f'survey_current_kmode_{k[i]:.2f}_sf_nl_z{redshift:.3f}_int.h5',
                                            error_x=False)
    data, Beane, LSE, MCMC, N = analysis.keep_P_21(k_indices, spectra_sf_interp, params_sf, n, model,
                                            N_modes=N_modes_joint, noiseless=False, nsteps=nsteps,
                                            backend_filename=f'survey_current_kmode_{k[i]:.2f}_sf_z{redshift:.3f}_int.h5',
                                            error_x=False)


    np.savez(f'results_all_int/sf_fits/survey_current_kmode_{k[i]:.2f}_sf_nl_z{redshift:.3f}_int', data=data_nl, Beane=Beane_nl, LSE=LSE_nl,
                                        samples=MCMC_nl[0], logp=MCMC_nl[1], N=N, params=params_sf)
    np.savez(f'results_all_int/sf_fits/survey_current_kmode_{k[i]:.2f}_sf_z{redshift:.3f}_int', data=data, Beane=Beane, LSE=LSE,
                                        samples=MCMC[0], logp=MCMC[1], N=N, params=params_sf)

    tf = time.time()
    print(f'run {i} saved to disk')
    print(f'time to complete superfake analysis run {i} is:', (tf - t0) / 60 / 60, 'hours')

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
