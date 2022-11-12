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

frac_op = .01
frac_con = .10
frac_pess = .25

noise = np.linspace(0, .25, 11)
noise = noise[1:]

print('superfake analysis')

### Superfake data and superfake noise levels

biases_sf = utils.extract_bias(k_indices, spectra_sf, P_m)
p_vals_sf = np.asarray([*biases_sf, P_m], dtype=object)

params_sf = dict(zip(p_names, p_vals_sf))
ndim = utils.get_params(params_sf, k_indices).size
model = models.ScalarBias_crossonly(k=spectra_sf[0], params=params_sf)

P_21_Beane = np.zeros((noise.size, 2))
P_21_MCMC_median = np.zeros((noise.size, 2))
P_21_MCMC_maxlogp = np.zeros((noise.size, 2))

var_21_Beane = np.zeros((noise.size, 2))
var_21_MCMC = np.zeros((noise.size, 2))

# MCMC has three values: median, max logp, and std
for i, n in enumerate(noise):
    print('Now on noise level',n,'%')
    nsteps = 1e6
    if n > .15:
        nsteps = 1e7

    Beane_nl, MCMC_nl = analysis.keep_P_21(k_indices, spectra_sf, params_sf, n, model,
                                            noiseless=False, nsteps=nsteps)
    Beane, MCMC = analysis.keep_P_21(k_indices, spectra_sf, params_sf, .001, model,
                                            noiseless=True, nsteps=nsteps)

    P_21_Beane[i,0] = Beane_nl[0]
    P_21_Beane[i,1] = Beane[0]

    P_21_MCMC_median[i,0] = MCMC_nl[0]
    P_21_MCMC_median[i,1] = MCMC[0]

    P_21_MCMC_maxlogp[i,0] = MCMC_nl[1]
    P_21_MCMC_maxlogp[i,1] = MCMC[1]

    var_21_Beane[i,0] = Beane_nl[1]
    var_21_Beane[i,1] = Beane[1]

    var_21_MCMC[i,0] = MCMC_nl[2]
    var_21_MCMC[i,1] = MCMC[2]

noise_sf_stats = {'P_21_Beane': P_21_Beane,
                'P_21_MCMC_median': P_21_MCMC_median,
                'P_21_MCMC_maxlogp': P_21_MCMC_maxlogp,
                'P_21_MCMC_median': P_21_MCMC_median,
                'var_21_Beane': var_21_Beane,
                'var_21_Beane': var_21_MCMC}

np.savez('noise_sf_stats', P_21_Beane=P_21_Beane,
                           P_21_MCMC_median=P_21_MCMC_median, P_21_MCMC_maxlogp=P_21_MCMC_maxlogp,
                           var_21_Beane=var_21_Beane, var_21_MCMC=var_21_MCMC)

# ### Simulated power law data and fractional noise error
# print('power law analysis')
#
# biases_pl = utils.extract_bias(k_indices, spectra_pl, P_m)
# p_vals_pl = np.asarray([*biases_pl, P_m], dtype=object)
#
# params_pl = dict(zip(p_names, p_vals_pl))
# ndim = utils.get_params(params_pl, k_indices).size
#
# data_pl_nl, Beane_pl_nl, LSE_pl_nl, MCMC_pl_nl = analysis.run_analysis(k_indices, spectra_pl, params_pl,
#                                                                 frac_pess, model, noiseless=True)
#
# data_pl_op, Beane_pl_op, LSE_pl_op, MCMC_pl_op = analysis.run_analysis(k_indices, spectra_pl, params_pl,
#                                                                 frac_op, model)
#
# data_pl_con, Beane_pl_con, LSE_pl_con, MCMC_pl_con = analysis.run_analysis(k_indices, spectra_pl, params_pl,
#                                                                 frac_con, model)
#
# data_pl_pess, Beane_pl_pess, LSE_pl_pess, MCMC_pl_pess = analysis.run_analysis(k_indices, spectra_pl, params_pl,
#                                                                 frac_pess, model)
#
# #analysis.plot_corner('pl_op.pdf', MCMC_pl_op, LSE_pl_op, Beane_pl_op, params_pl, spectra_pl[1][0], k_indices)
# #analysis.plot_corner('pl_con.pdf', MCMC_pl_con, LSE_pl_con, Beane_pl_con, params_pl, spectra_pl[1][0], k_indices)
# #analysis.plot_corner('pl_pess.pdf', MCMC_pl_pess, LSE_pl_pess, Beane_pl_pess, params_pl, spectra_pl[1][0], k_indices)
#
# np.savez('pl_results_z6.0155', data_pl_nl=data_pl_nl, data_pl_op=data_pl_op, data_pl_con=data_pl_con, data_pl_pess=data_pl_pess,
#                     Beane_pl_nl=Beane_pl_nl, Beane_pl_op=Beane_pl_op, Beane_pl_con=Beane_pl_con, Beane_pl_pess=Beane_pl_pess,
#                     LSE_pl_nl=LSE_pl_nl, LSE_pl_op=LSE_pl_op, LSE_pl_con=LSE_pl_con, LSE_pl_pess=LSE_pl_pess,
#                     MCMC_op_samples=MCMC_pl_op[0], MCMC_con_samples=MCMC_pl_con[0], MCMC_pess_samples=MCMC_pl_pess[0],
#                     MCMC_op_logp=MCMC_pl_op[1], MCMC_con_logp=MCMC_pl_con[1], MCMC_pess_logp=MCMC_pl_pess[1],
#                     MCMC_nl_samples=MCMC_pl_nl[0], MCMC_nl_logp=MCMC_pl_nl[1])
#
# ### Simulated brightness temperature data and fractional noise error
# print('brightness temperature analysis')
#
# biases_bt = utils.extract_bias(k_indices, spectra_bt, P_m)
# p_vals_bt = np.asarray([*biases_bt, P_m], dtype=object)
#
# params_bt = dict(zip(p_names, p_vals_bt))
# ndim = utils.get_params(params_bt, k_indices).size
#
# data_bt_nl, Beane_bt_nl, LSE_bt_nl, MCMC_bt_nl = analysis.run_analysis(k_indices, spectra_bt, params_bt,
#                                                                 frac_pess, model, noiseless=True)
#
# data_bt_op, Beane_bt_op, LSE_bt_op, MCMC_bt_op = analysis.run_analysis(k_indices, spectra_bt, params_bt,
#                                                                 frac_op, model)
#
# data_bt_con, Beane_bt_con, LSE_bt_con, MCMC_bt_con = analysis.run_analysis(k_indices, spectra_bt, params_bt,
#                                                                 frac_con, model)
#
# data_bt_pess, Beane_bt_pess, LSE_bt_pess, MCMC_bt_pess = analysis.run_analysis(k_indices, spectra_bt, params_bt,
#                                                                 frac_pess, model)
#
# #analysis.plot_corner('bt_op.pdf', MCMC_bt_op, LSE_bt_op, Beane_bt_op, params_bt, spectra_bt[1][0], k_indices)
# #analysis.plot_corner('bt_con.pdf', MCMC_bt_con, LSE_bt_con, Beane_bt_con, params_bt, spectra_bt[1][0], k_indices)
# #analysis.plot_corner('bt_pess.pdf', MCMC_bt_pess, LSE_bt_pess, Beane_bt_pess, params_bt, spectra_bt[1][0], k_indices)
#
# np.savez('bt_results_z6.0155', data_bt_nl=data_bt_nl, data_bt_op=data_bt_op, data_bt_con=data_bt_con, data_bt_pess=data_bt_pess,
#                     Beane_bt_nl=Beane_bt_nl, Beane_bt_op=Beane_bt_op, Beane_bt_con=Beane_bt_con, Beane_bt_pess=Beane_bt_pess,
#                     LSE_bt_nl=LSE_bt_nl, LSE_bt_op=LSE_bt_op, LSE_bt_con=LSE_bt_con, LSE_bt_pess=LSE_bt_pess,
#                     MCMC_op_samples=MCMC_bt_op[0], MCMC_con_samples=MCMC_bt_con[0], MCMC_pess_samples=MCMC_bt_pess[0],
#                     MCMC_op_logp=MCMC_bt_op[1], MCMC_con_logp=MCMC_bt_con[1], MCMC_pess_logp=MCMC_bt_pess[1],
#                     MCMC_nl_samples=MCMC_bt_nl[0], MCMC_nl_logp=MCMC_bt_nl[1])
#
# ### Fisher analysis
