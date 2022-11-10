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

plt.style.use('seaborn-colorblind')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# Simulations

print('loading simulations')

which_box = 'big'
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

    box_size = 160 # in Mpc/h
    r = np.linspace(0, box_size, rez)
    r_vec = np.stack((r, r, r))

mass_voxels, mass_edges = np.histogramdd([x,y,z], bins=rez,
                                                weights=masses)

masses_hist = plt.hist(np.log(masses), bins=50)

print('generating underlying matter density spectrum')
#print('loading underlying matter density spectrum')

delta = utils.overdensity(density)
#k, P_m = analysis.calc_pspec(r_vec, [delta], n_bins=n_bins, bin_scale='log')

if which_box is 'little':
    #np.savez('matter_pspec_6.0155.npz')
    print('loading matter stuff for', which_box, 'box')
    matter_pspec = np.load('matter_pspec_6.0155.npz')
    k = matter_pspec['k']
    P_m = matter_pspec['P_m']

if which_box is 'big':
    #np.savez('matter_pspec_7.9589', k=k, P_m=P_m)
    print('loading matter stuff for', which_box, 'box')
    matter_pspec = np.load('matter_pspec_7.9589.npz')
    k = matter_pspec['k']
    P_m = matter_pspec['P_m']

#matter_pspec = np.load('matter_pspec_6.0155.npz')
#k = matter_pspec['k']
#P_m = matter_pspec['P_m']

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
scalings = np.ones(3) # np.array([3.76387000e-04, 1.06943379e+05, 6.86553108e+01])

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
hubble0 = Planck15.H0

alpha = 0.564
k_0 = 0.185 # Mpc/h

# global temperature as a function of redshift
def t0(z):
    return 38.6 * hubble0.value * (omegab / 0.045) * np.sqrt(0.27 / omegam * (1 + z) / 10)

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

#zreion = gen_21cm_fields(delta)
#ion_field, t21_field = get_21cm_fields(redshift, zreion, delta)
#np.save('zreion_z7.9589.npy', zreion)


### Power law data
# print('generating power law data')
# # power law
# spectra_pl = analysis.gen_spectra(r_vec, I_fields)
#
# print('generating superfake data')
# # indices
#
### Superfake data
# k_indices = [6]
# spectra_sf = cp.deepcopy(spectra_pl)
#
# b_i = np.sqrt(spectra_sf[1][0][k_indices] / P_m[k_indices])
# b_j = np.sqrt(spectra_sf[1][3][k_indices] / P_m[k_indices])
# b_k = np.sqrt(spectra_sf[1][5][k_indices] / P_m[k_indices])
#
# biases = [16, 3 * 1.1e3, 3 * 1.08e3 ] # [b_i, b_j, b_k]
# indices = utils.lines_indices()
#
# for i in range(len(indices)):
#     print(indices[i][0], indices[i][1])
#     spectra_sf[1][i] = biases[int(indices[i][0])] * biases[int(indices[i][1])] * P_m
#
# ### Brightness temperature data
#
# I_fields_bt = cp.deepcopy(I_fields)
# I_fields_bt[0] = t21_field
#
# print('generating brightness temperature data')
# # full simulation
# spectra_bt = analysis.gen_spectra(r_vec, I_fields_bt)
#
# ### Datasets
#
#np.savez('pspecs_sf_z7.9589', P_21cm_21cm=spectra_sf[1][0], P_21cm_CII=spectra_sf[1][1],
#                     P_21cm_OIII=spectra_sf[1][2], P_CII_CII=spectra_sf[1][3],
#                     P_CII_OIII=spectra_sf[1][4], P_OIII_OIII=spectra_sf[1][5])

# k, P_21cm_21cm = analysis.calc_pspec(r_vec,
#                 [I_fields[0], I_fields[0]],
#                 n_bins=n_bins, bin_scale='log')
#
# k, P_CII_CII = analysis.calc_pspec(r_vec,
#                 [I_fields[1], I_fields[1]],
#                 n_bins=n_bins, bin_scale='log')
#
# k, P_OIII_OIII = analysis.calc_pspec(r_vec,
#                 [I_fields[2], I_fields[2]],
#                 n_bins=n_bins, bin_scale='log')
# #
# np.savez('autos_pl_z7.9589', P_21cm_21cm=P_21cm_21cm,
#                                 P_CII_CII=P_CII_CII,
#                                 P_OIII_OIII=P_OIII_OIII)

k, P_21cm_CII = analysis.calc_pspec(r_vec,
                [I_fields[0], I_fields[1]],
                n_bins=n_bins, bin_scale='log')

k, P_CII_OIII = analysis.calc_pspec(r_vec,
                [I_fields[1], I_fields[2]],
                n_bins=n_bins, bin_scale='log')

k, P_21cm_OIII = analysis.calc_pspec(r_vec,
                [I_fields[0], I_fields[2]],
                n_bins=n_bins, bin_scale='log')
#
np.savez('crosses_pl_z7.9589', P_21cm_CII=P_21cm_CII,
                                P_CII_OIII=P_CII_OIII,
                                P_21cm_OIII=P_21cm_OIII)
#
# np.savez('pspecs_bt_z7.9589', P_21cm_21cm=spectra_bt[1][0], P_21cm_CII=spectra_bt[1][1],
#                     P_21cm_OIII=spectra_bt[1][2], P_CII_CII=spectra_bt[1][3],
#                     P_CII_OIII=spectra_bt[1][4], P_OIII_OIII=spectra_bt[1][5])
