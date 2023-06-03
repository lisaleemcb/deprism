import time
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

#print('generating underlying matter density spectrum')
print('loading underlying matter density spectrum')

delta = utils.overdensity(density)
#k, P_m = analysis.calc_pspec(r_vec, [delta], n_bins=n_bins, bin_scale='log')
#np.savez('matter_pspec_6.0155', k=k, P_m=P_m)

matter_pspec = np.load(f'spectra/matter_pspec_z{redshift}.npz')
k = matter_pspec['k']
P_m = matter_pspec['P_m']

print('yay! finished the matter stuff')

### Datasets

# pspecs_sf = np.load('spectra/pspecs_sf_z6.0155.npy')
# pspecs_pl = np.load('pspecs_pl.npz')
# pspecs_bt = np.load('pspecs_bt.npz')
# pspecs_bt.files

spectra_sf = np.load(f'spectra_all_int/spectra_sf_z{redshift}.npy')
spectra_pl = np.load(f'spectra_all_int/spectra_pl_z{redshift}.npy')
spectra_bt = np.load(f'spectra_all_int/spectra_bt_z{redshift}.npy')

#### Autocorrelations

P_21cm_21cm = spectra_bt[0] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)
P_CII_CII = spectra_bt[3] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)
P_OIII_OIII = spectra_bt[5] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)

#### Crosscorrelations

P_21cm_CII = spectra_bt[1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)
P_21cm_OIII = spectra_bt[2] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)
P_CII_OIII = spectra_bt[4] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)

# Fitting

p_names = np.asarray(['b_i','b_j', 'b_k', 'P_m'])
k_indices = [6]

priors = np.arange(.75,1.25,.05)

print('superfake analysis')

### Superfake data and superfake noise levels

biases_sf = utils.extract_bias(k_indices, spectra_sf, P_m)
p_vals_sf = np.asarray([*biases_sf, P_m], dtype=object)

params_sf = dict(zip(p_names, p_vals_sf))
ndim = utils.get_params(params_sf, k_indices).size
model = models.ScalarBias_crossonly(k=spectra_sf[0], params=params_sf)
N_modes_small = survey.calc_N_modes(k, 80**3 * u.Mpc**3, align='left')

# t0 = time.time()
# print('Flat prior calculation...')
# nsteps = int(1e6)
# n = .05
# if n > .1:
#     nsteps = int(1e7)
#
# data_nl, Beane_nl, LSE_nl, MCMC_nl = analysis.keep_P_21(k_indices, spectra_sf, params_sf, n, model,
#                                         N_modes=N_modes_small, noiseless=True, nsteps=nsteps,
#                                         priors='uniform',
#                                         backend_filename=f'uniformprior_sf_nl_z{redshift}_int.h5')
# data, Beane, LSE, MCMC = analysis.keep_P_21(k_indices, spectra_sf, params_sf, n, model,
#                                         priors='uniform',
#                                         N_modes=N_modes_small, noiseless=False, nsteps=nsteps,
#                                         backend_filename=f'uniformprior_sf_z{redshift}_int.h5')
#
#
# np.savez(f'results_all_int/sf_fits/uniformprior_sf_nl_z{redshift}_int', data=data_nl, Beane=Beane_nl, LSE=LSE_nl,
#                                     samples=MCMC_nl[0], logp=MCMC_nl[1])
# np.savez(f'results_all_int/sf_fits/uniformprior_sf_z{redshift}_int', data=data, Beane=Beane, LSE=LSE,
#                                     samples=MCMC[0], logp=MCMC[1])

# 
# tf = time.time()
# print(f'time to complete uniform prior run is:', (tf - t0) / 60, 'minutes')
#

noise = np.asarray([.001, .005, .01, .05, .1, .15])

for i, n in enumerate(noise):
    t0 = time.time()
    print('Now on noise level',n,'%')
    nsteps = int(1e6)
    if n > .1:
        nsteps = int(1e7)

    print('Prior offset calculation...')
    p = .95

    data_nl, Beane_nl, LSE_nl, MCMC_nl = analysis.keep_P_21(k_indices, spectra_sf, params_sf, n, model,
                                            N_modes=N_modes_small, noiseless=True, nsteps=nsteps,
                                            priors='gaussian', priors_offset=p,
                                            backend_filename=f'prioroffset{p}_noise{n}_bt_nl_z{redshift}_int.h5')
    data, Beane, LSE, MCMC = analysis.keep_P_21(k_indices, spectra_sf, params_sf, n, model,
                                            priors='gaussian', priors_offset=p,
                                            N_modes=N_modes_small, noiseless=False, nsteps=nsteps,
                                            backend_filename=f'prioroffset{p}_noise_{n}_bt_z{redshift}_int.h5')


    np.savez(f'results_all_int/sf_fits/prioroffset{p}_noise{n}_sf_nl_z{redshift}_int', data=data_nl, Beane=Beane_nl, LSE=LSE_nl,
                                        samples=MCMC_nl[0], logp=MCMC_nl[1])
    np.savez(f'results_all_int/sf_fits/prioroffset{p}_noise{n}_sf_z{redshift}_int', data=data, Beane=Beane, LSE=LSE,
                                        samples=MCMC[0], logp=MCMC[1])


    tf = time.time()
    print(f'time to complete prior offset runs is:', (tf - t0) / 60, 'minutes')
