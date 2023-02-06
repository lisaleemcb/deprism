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

#print('generating underlying matter density spectrum')
print('loading underlying matter density spectrum')

delta = utils.overdensity(density)
#k, P_m = analysis.calc_pspec(r_vec, [delta], n_bins=n_bins, bin_scale='log')
#np.savez('matter_pspec_6.0155', k=k, P_m=P_m)

matter_pspec = np.load('spectra/matter_pspec_6.0155.npz')
k = matter_pspec['k']
P_m = matter_pspec['P_m']

print('yay! finished the matter stuff')

### Datasets

# pspecs_sf = np.load('spectra/pspecs_sf_z6.0155.npy')
# pspecs_pl = np.load('pspecs_pl.npz')
# pspecs_bt = np.load('pspecs_bt.npz')
# pspecs_bt.files

spectra_sf = np.load('spectra/pspecs_sf_z6.0155.npy')
spectra_pl = np.load('spectra/pspecs_pl_z6.0155.npy')
spectra_bt = np.load('spectra/pspecs_bt_z6.0155.npy')

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

frac_op = .005
frac_con = .01
frac_pess = .10

noise = np.asarray([.001, .005, .01, .05, .1, .15])

print('superfake analysis')

### Superfake data and superfake noise levels

biases_sf = utils.extract_bias(k_indices, spectra_sf, P_m)
p_vals_sf = np.asarray([*biases_sf, P_m], dtype=object)

params_sf = dict(zip(p_names, p_vals_sf))
ndim = utils.get_params(params_sf, k_indices).size
model = models.ScalarBias_crossonly(k=spectra_sf[0], params=params_sf)
N_modes_small = survey.calc_N_modes(k, 80**3 * u.Mpc**3, align='left')

for i, n in enumerate(noise):
    print('Now on noise level',n,'%')
    nsteps = 5e6
    if n > .1:
        nsteps = 5e7

    data_nl, Beane_nl, LSE_nl, MCMC_nl = analysis.keep_P_21(k_indices, spectra_sf, params_sf, n, model,
                                            N_modes=N_modes_small, noiseless=False, nsteps=nsteps)
    data, Beane, LSE, MCMC = analysis.keep_P_21(k_indices, spectra_sf, params_sf, n, model,
                                            N_modes=N_modes_small, noiseless=True, nsteps=nsteps)


    np.savez(f'noise_{n}_sf_nl_z6.0155', data=data_nl, Beane=Beane_nl, LSE=LSE_nl, MCMC=MCMC_nl)
    np.savez(f'noise_{n}_sf_z6.0155', data=data, Beane=Beane, LSE=LSE, MCMC=MCMC)


### Simulated power law data and fractional noise error
print('power law analysis')

for i, n in enumerate(noise):
    print('Now on noise level',n,'%')
    nsteps = 5e6
    if n > .1:
        nsteps = 5e7

    data_nl, Beane_nl, LSE_nl, MCMC_nl = analysis.keep_P_21(k_indices, spectra_pl, params_pl, n, model,
                                            N_modes=N_modes_small, noiseless=False, nsteps=nsteps)
    data, Beane, LSE, MCMC = analysis.keep_P_21(k_indices, spectra_pl, params_pl, n, model,
                                            N_modes=N_modes_small, noiseless=True, nsteps=nsteps)

    np.savez(f'noise_{n}_pl_nl_z6.0155', data=data_nl, Beane=Beane_nl, LSE=LSE_nl, MCMC=MCMC_nl)
    np.savez(f'noise_{n}_pl_z6.0155', data=data, Beane=Beane, LSE=LSE, MCMC=MCMC)

### Simulated brightness temperature data and fractional noise error
print('brightness temperature analysis')

biases_bt = utils.extract_bias(k_indices, spectra_bt, P_m)
p_vals_bt = np.asarray([*biases_bt, P_m], dtype=object)

params_bt = dict(zip(p_names, p_vals_bt))
ndim = utils.get_params(params_bt, k_indices).size

for i, n in enumerate(noise):
    print('Now on noise level',n,'%')
    nsteps = 5e6
    if n > .1:
        nsteps = 5e7

    data_nl, Beane_nl, LSE_nl, MCMC_nl = analysis.keep_P_21(k_indices, spectra_bt, params_bt, n, model,
                                            N_modes=N_modes_small, noiseless=False, nsteps=nsteps)
    data, Beane, LSE, MCMC = analysis.keep_P_21(k_indices, spectra_bt, params_bt, n, model,
                                            N_modes=N_modes_small, noiseless=True, nsteps=nsteps)


    np.savez(f'noise_{n}_bt_nl_z6.0155', data=data_nl, Beane=Beane_nl, LSE=LSE_nl, MCMC=MCMC_nl)
    np.savez(f'noise_{n}_bt_z6.0155', data=data, Beane=Beane, LSE=LSE, MCMC=MCMC)

### Fisher analysis
