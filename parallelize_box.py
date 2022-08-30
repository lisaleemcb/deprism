import numpy as np
import copy as cp
import sys
import time
import h5py
import emcee
#import scipy.integrate
import astropy.constants as const

from multiprocessing import Pool
from scipy.special import comb
from astropy import units as u
from astropy.cosmology import Planck15

import analysis
import signals
import estimators
#import fitting
import models
import utils

print('loading simulations')

which_box = 'little'
print('running analysis on ', which_box, ' box')

print('loading underlying matter density spectrum')

if which_box is 'little':
    redshift = 6.0155
    rez = 512
    box = h5py.File('L80_halos_z=6.0155.hdf5', 'r')
    print(box.keys())

    masses = np.array(box[('mass')])
    pos = np.array(box[('pos')])
    density = np.array(box[('rho')])
    x, y, z = pos.T

    delta = utils.overdensity(density)
    matter_pspec = np.load('matter_pspec_6.0155.npz')
    k = matter_pspec['k']
    P_m = matter_pspec['P_m']

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

print('yay! finished the matter stuff. now loading spectra...')

# parameters
#box = 80.0  # Mpc/h
omegam = Planck15.Om0
omegab = Planck15.Ob0
hubble0 = Planck15.H0

# global temperature as a function of redshift
def t0(z):
    return 38.6 * hubble0.value * (omegab / 0.045) * np.sqrt(0.27 / omegam * (1 + z) / 10)

def get_21cm_fields(z, zreion, delta):
    #print("computing t21 at z=", z, "...")
    ion_field = np.where(zreion > z, 1.0, 0.0)
    t21_field = t0(z) * (1 + delta) * (1 - ion_field)

    return ion_field, t21_field

zreion = np.load('zreion.npy') # gen_21cm_fields(delta)
ion_field, t21_field = get_21cm_fields(redshift, zreion, delta)

pspecs_sf = np.load('pspecs_sf.npz')
#pspecs_pl = np.load('pspecs_pl.npz')
#pspecs_bt = np.load('pspecs_bt.npz')
print(pspecs_sf.files)

spectra_sf = (k, [pspecs_sf['P_21cm_21cm'], pspecs_sf['P_21cm_CII'],
               pspecs_sf['P_21cm_OIII'], pspecs_sf['P_CII_CII'],
               pspecs_sf['P_CII_OIII'], pspecs_sf['P_OIII_OIII']])

#spectra_pl = (k, [pspecs_pl['P_21cm_21cm'], pspecs_pl['P_21cm_CII'],
#               pspecs_pl['P_21cm_OIII'], pspecs_pl['P_CII_CII'],
#               pspecs_pl['P_CII_OIII'], pspecs_pl['P_OIII_OIII']])

#spectra_bt = (k, [pspecs_bt['P_21cm_21cm'], pspecs_bt['P_21cm_CII'],
#               pspecs_bt['P_21cm_OIII'], pspecs_bt['P_CII_CII'],
#               pspecs_bt['P_CII_OIII'], pspecs_bt['P_OIII_OIII']])

# Fitting
print('spectra loaded!')
print('starting fitting...')

k_indices = [6]
p_names = np.asarray(['b_i','b_j', 'b_k', 'P_m'])
biases_sf = utils.extract_bias(k_indices, spectra_sf[1], P_m)
p_vals_sf = np.asarray([*biases_sf, P_m], dtype=object)

params_sf = dict(zip(p_names, p_vals_sf))
truths = utils.get_params(params_sf, k_indices)

model = models.ScalarBias_crossonly(k=spectra_sf[0], params=params_sf)
data = utils.fetch_data(k_indices, spectra_sf[1], b_0=params_sf['b_i'])[1:-1]

def log_prob(params):
    b_i, b_j, b_k, P_m = params
    model = np.array([b_i * b_j * P_m, b_j * b_k * P_m, b_i * b_k * P_m])
    diff = model - data

    lp = -(b_i - b_0)**2 / (2 * (b_0 * priors_width)**2)

    return -0.5 * np.dot(diff, np.linalg.solve(N, diff)) + lp

b_0=biases_sf[0]
priors_width = .15
frac_error=.001
ndim = truths.size
burn_in = 1e2
nwalkers = 48
nsteps = 2e7
p0 = truths + np.random.rand(nwalkers, ndim) * .01
N = analysis.estimate_errors(data, frac_error=frac_error)
print('BEFORE IT ALL:', log_prob(truths))
print('DATA: ', data)
print('NOISE: ', np.diag(N))
### MCMC's
print('running MCMC with a priors width of', priors_width, 'and', frac_error, 'fractional error')
print('and', nwalkers, 'walkers running', nsteps, 'steps')

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)

start = time.time()
state = sampler.run_mcmc(p0, burn_in, progress=True)
sampler.reset()
sampler.run_mcmc(state, nsteps, progress=True)

end = time.time()
serial_data_time = end - start
print("Serial took {0:.1f} seconds".format(serial_data_time))

samples = sampler.get_chain(flat=True, thin=100)
logs_p = sampler.get_log_prob(flat=True, thin=100)

np.savez('sf_results_p15_N001', samples=samples, logs_p=logs_p)
