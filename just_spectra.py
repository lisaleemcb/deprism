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
import astropy.units as u
import zreion
import time

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

# Simulations

print('loading simulations')

which_box = 'big'
load_files = False
print('running analysis on', which_box, 'box')

if which_box == 'small':
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

    box_size = 160 # in Mpc/h
    r = np.linspace(0, box_size, rez)
    r_vec = np.stack((r, r, r))

print('generating underlying matter density spectrum')
#print('loading underlying matter density spectrum')


delta = utils.overdensity(density)
k, P_m = analysis.calc_pspec(r_vec, [delta], n_bins=n_bins, bin_scale='log')

np.savez(f'spectra/matter_pspec_z{redshift}.npz', k=k, P_m=P_m)

tf = time.time()
print(f'generated matter power spectrum for redshift z={redshift}.')
print(f'it took', (tf - t0) / 60, 'minutes')


#matter_pspec = np.load('matter_pspec_6.0155.npz')
#k = matter_pspec['k']
#P_m = matter_pspec['P_m']

print('yay! finished the matter stuff')

H_I_power = 1.3

L = 2.0 / 3.0
M = 1.0
H = 4.0 / 3.0

power_indices = [H_I_power, L, M]

L_solar=3.828e26
L_CII = 10e6
L_OIII = 10e9

luminosities_L = utils.mass2luminosity(masses, power=L)
luminosities_M = utils.mass2luminosity(masses, power=M)
luminosities_H = utils.mass2luminosity(masses, power=H)

intensities_L = utils.specific_intensity(redshift, L=luminosities_L)
intensities_M = utils.specific_intensity(redshift, L=luminosities_M)
intensities_H = utils.specific_intensity(redshift, L=luminosities_H)

I_fields_pl = np.zeros((runs, rez, rez, rez))
scalings = np.ones(3) # np.array([3.76387000e-04, 1.06943379e+05, 6.86553108e+01])

for i, power in enumerate(power_indices):
    print('power =', power)
    intensities = utils.specific_intensity(redshift,
                            L=scalings[i] * utils.mass2luminosity(masses, power=power, mass_0=1.0))

    print('mean intensity = ', intensities.mean())
    print(' ')
    # lumens += np.random.lognormal(mean=0.0, sigma=2.0, size=None)
    I_voxels, I_edges = np.histogramdd([x,y,z], bins=rez, weights=intensities)
    I_fields_pl[i] = I_voxels


k_indices = [6]
runs=3
n_bins=20
spectra_sf = np.zeros((int(comb(runs, 2) + runs), n_bins))

N_modes_small = survey.calc_N_modes(k, 80**3 * u.Mpc**3, align='left')
indices = utils.lines_indices()

# # parameters
omegam = Planck15.Om0
omegab = Planck15.Ob0
h = Planck15.h

alpha = 0.564
k_0 = 0.185 # Mpc/h

#global temperature as a function of redshift
def t0(z):
    return 38.6 * h * (omegab / 0.045) * np.sqrt(0.27 / omegam * (1 + z) / 10)

def gen_21cm_fields(delta, box_size=box_size, zmean=7, alpha=0.11, k0=0.05):
    # compute zreion field
    print("computing zreion...")
    zreion_field = zreion.apply_zreion_fast(delta, zmean, alpha, k0, box_size, deconvolve=False)

    return zreion_field

def get_21cm_fields(z, zreion_field, delta):
    #print("computing t21 at z=", z, "...")
    ion_field = np.where(zreion_field > z, 1.0, 0.0)
    t21_field = t0(z) * (1 + delta) * (1 - ion_field)

    return ion_field, t21_field

if load_files:
    zreion_fields = np.load(f'zreion_files/zreion_z{redshift}.npz') #gen_21cm_fields(delta)

zreion_field = gen_21cm_fields(delta)
ion_field, t21_field = get_21cm_fields(redshift, zreion_field, delta)

nu_21cm_rest = 1420 * u.MHz
nu_21cm_obvs = utils.calc_nu_obs(nu_21cm_rest, redshift)

i21_field = (t21_field * u.mK).to(u.Jy / u.steradian,
                      equivalencies=u.brightness_temperature(nu_21cm_obvs))

np.savez(f'zreion_files/zreion_z{redshift}', zreion_field=zreion_field,
                                    ion_field=ion_field,
                                    t21_field=t21_field,
                                    i21_field=i21_field)


# ### Power law data
# print('first run of power law data')
# # power law
# t0 = time.time()
# spectra_pl = analysis.gen_spectra(r_vec, I_fields_pl)
#
# tf = time.time()
# print(f'time to complete power law run {i} is:', (tf - t0) / 60, 'minutes')
#
# print('getting 21cm bias factor and scalings')
#
# k, P_21 = analysis.calc_pspec(r_vec, [i21_field.value], n_bins=n_bins, bin_scale='log')
#
# b_21cm = np.sqrt(P_21[k_indices] / P_m[k_indices]) # mK
# b_CII = 3 * 1.1e3   # Jy/str
# b_OIII = 5 * 1.0e3  # Jy/str
# biases = [b_21cm, b_CII, b_OIII]
# print('theoretical biases are', biases)
#
# def calc_scalings(bias, spectra, P_m, k_indices):
#     s2 = (P_m[k_indices] * bias**2) / spectra[k_indices]
#
#     return np.sqrt(s2)
#
# scalings = [calc_scalings(b_21cm, spectra_pl[0], P_m, k_indices),
#             calc_scalings(b_CII, spectra_pl[3], P_m, k_indices),
#             calc_scalings(b_OIII, spectra_pl[5], P_m, k_indices)]
#
# print('generating scaled data')
# print('with scalings:', scalings)
#
# for i, power in enumerate(power_indices):
#     print('power =', power)
#     intensities = utils.specific_intensity(redshift,
#                             L=scalings[i] * utils.mass2luminosity(masses, power=power, mass_0=1.0))
#
#     print('mean intensity = ', intensities.mean())
#     print(' ')
#     # lumens += np.random.lognormal(mean=0.0, sigma=2.0, size=None)
#     I_voxels, I_edges = np.histogramdd([x,y,z], bins=rez, weights=intensities)
#     I_fields_pl[i] = I_voxels
#
# print('generating perfect bias data')
# spectra_sf = cp.deepcopy(spectra_pl)
# for i in range(len(indices)):
#     print(indices[i][0], indices[i][1])
#     spectra_sf[i] = biases[int(indices[i][0])] * biases[int(indices[i][1])] * P_m
#
# ### Power law data
# print('scaled run power law data')
# # power law
# spectra_pl = analysis.gen_spectra(r_vec, I_fields_pl)
#
#
# # print('generating brightness temperature data')
# # # full simulation
# # ### Brightness temperature data
# # I_fields_bt = cp.deepcopy(I_fields_pl)
# # I_fields_bt[0] = i21_field.value
# #
# # spectra_bt = analysis.gen_spectra(r_vec, I_fields_bt)
# #
#
#
# # ### Brightness temperature data
# #
# # I_fields_bt = cp.deepcopy(I_fields)
# # I_fields_bt[0] = t21_field
# #
# # print('generating brightness temperature data')
# # # full simulation
# # spectra_bt = analysis.gen_spectra(r_vec, I_fields_bt)
# #
# # ### Datasets
# #
# np.save(f'spectra_all_int/spectra_sf_z{redshift}', spectra_sf)
#
# np.save(f'spectra_all_int/spectra_pl_z{redshift}', spectra_pl)
#
# np.save(f'spectra_all_int/spectra_bt_z{redshift}', spectra_bt)
#
# # in Npz format
#
# np.savez(f'spectra_all_int/pspecs_sf_z{redshift}', P_21cm_21cm=spectra_sf[0], P_21cm_CII=spectra_sf[1],
#                     P_21cm_OIII=spectra_sf[2], P_CII_CII=spectra_sf[3],
#                     P_CII_OIII=spectra_sf[4], P_OIII_OIII=spectra_sf[5])
#
# # np.savez(f'spectra_all_int/pspecs_pl_z{redshift}', P_21cm_21cm=spectra_pl[0], P_21cm_CII=spectra_pl[1],
# #                      P_21cm_OIII=spectra_pl[2], P_CII_CII=spectra_pl[3],
# #                      P_CII_OIII=spectra_pl[4], P_OIII_OIII=spectra_pl[5])
# #
# # np.savez(f'spectra_all_int/pspecs_bt_z{redshift}', P_21cm_21cm=spectra_bt[0], P_21cm_CII=spectra_bt[1],
# #                     P_21cm_OIII=spectra_bt[2], P_CII_CII=spectra_bt[3],
# #                     P_CII_OIII=spectra_bt[4], P_OIII_OIII=spectra_bt[5])
