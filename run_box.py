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

# this is for the mean voxel intensity to be correct
# but this makes the spectra intensity incorrect
#scalings = np.array([6.36450014002845e-05, 3148.8613192593207, 2.584235977107341])

scalings = np.array([1,1,1])

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

print('loading zreion...')
zreion = np.load('zreion_files/zreion_z6.0155.npy')# gen_21cm_fields(delta)
#np.save('zreion_files/zreion_z6.0155', zreion)

ion_field, t21_field = get_21cm_fields(redshift, zreion, delta)

### Superfake data
k_indices = [6]
runs=3
n_bins=20
spectra_sf = np.zeros((int(comb(runs, 2) + runs), n_bins))

N_modes_small = survey.calc_N_modes(k, 80**3 * u.Mpc**3, align='left')
indices = utils.lines_indices()

### Power law data
print('first run of power law data')
# power law
spectra_pl = analysis.gen_spectra(r_vec, I_fields)

### Brightness temperature data
I_fields_bt = cp.deepcopy(I_fields)
I_fields_bt[0] = t21_field

print('generating brightness temperature data')
# full simulation
spectra_bt = analysis.gen_spectra(r_vec, I_fields_bt)

print('getting 21cm bias factor and scalings')

b_21cm = np.sqrt(spectra_bt[0][k_indices] / P_m[k_indices]) # mK
b_CII = 3 * 1.1e3   # Jy/str
b_OIII = 5 * 1.0e3  # Jy/str
biases = [b_21cm, b_CII, b_OIII]

def calc_scalings(bias, spectra, P_m, k_indices):
    s2 = (P_m[k_indices] * bias**2) / spectra[k_indices]

    return np.sqrt(s2)

scalings = [calc_scalings(b_21cm, spectra_pl[0], P_m, k_indices),
            calc_scalings(b_CII, spectra_pl[3], P_m, k_indices),
            calc_scalings(b_OIII, spectra_pl[5], P_m, k_indices)]

print('generating scaled data')
print('with scalings:', scalings)

for i, power in enumerate(power_indices):
    print('power =', power)
    intensities = utils.specific_intensity(redshift,
                            L=scalings[i] * utils.mass2luminosity(masses, power=power, mass_0=1.0))

    print('mean intensity = ', intensities.mean())
    print(' ')
    # lumens += np.random.lognormal(mean=0.0, sigma=2.0, size=None)
    I_voxels, I_edges = np.histogramdd([x,y,z], bins=rez, weights=intensities)
    I_fields[i] = I_voxels

print('generating perfect bias data')

for i in range(len(indices)):
    print(indices[i][0], indices[i][1])
    spectra_sf[i] = biases[int(indices[i][0])] * biases[int(indices[i][1])] * P_m

### Power law data
print('scaled run power law data')
# power law
spectra_pl = analysis.gen_spectra(r_vec, I_fields)

### Brightness temperature data
I_fields_bt = cp.deepcopy(I_fields)
I_fields_bt[0] = t21_field

print('generating brightness temperature data')
# full simulation
spectra_bt = analysis.gen_spectra(r_vec, I_fields_bt)

### Datasets

np.save('pspecs_sf_z6.0155', spectra_sf)
np.save('pspecs_pl_z6.0155', spectra_pl)
np.save('pspecs_bt_z6.0155', spectra_bt)

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

P_21cm_21cm = spectra_bt[0] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)
P_CII_CII = spectra_bt[3] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)
P_OIII_OIII = spectra_bt[5] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)

#### Crosscorrelations

P_21cm_CII = spectra_bt[1] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)
P_21cm_OIII = spectra_bt[2] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)
P_CII_OIII = spectra_bt[4] * u.Mpc**3 * u.Jy**2 * u.steradian**(-2)

# Fitting

p_names = np.asarray(['b_i','b_j', 'b_k', 'P_m'])

frac_op = .005
frac_con = .01
frac_pess = .10

#variances = [var_21cm_21cm, var_21cm_CII, var_21cm_OIII, var_CII_OIII, model_params_sf['b_i'] * .25]

print('superfake analysis')

### Superfake data and superfake noise levels

biases_sf = utils.extract_bias(k_indices, spectra_sf, P_m)
p_vals_sf = np.asarray([*biases_sf, P_m], dtype=object)

params_sf = dict(zip(p_names, p_vals_sf))
ndim = utils.get_params(params_sf, k_indices).size
model = models.ScalarBias_crossonly(k=spectra_sf[0], params=params_sf)

data_sf_nl, Beane_sf_nl, LSE_sf_nl, MCMC_sf_nl = analysis.run_analysis(k_indices, spectra_sf, params_sf,
                                                                frac_pess, model, N_modes=N_modes_small, noiseless=True)

data_sf_op, Beane_sf_op, LSE_sf_op, MCMC_sf_op = analysis.run_analysis(k_indices, spectra_sf, params_sf,
                                                                frac_op, model, N_modes=N_modes_small)

data_sf_con, Beane_sf_con, LSE_sf_con, MCMC_sf_con = analysis.run_analysis(k_indices, spectra_sf, params_sf,
                                                                frac_con, model, N_modes=N_modes_small)

data_sf_pess, Beane_sf_pess, LSE_sf_pess, MCMC_sf_pess = analysis.run_analysis(k_indices, spectra_sf, params_sf,
                                                                frac_pess, model, N_modes=N_modes_small)

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

biases_pl = utils.extract_bias(k_indices, spectra_pl, P_m)
p_vals_pl = np.asarray([*biases_pl, P_m], dtype=object)

params_pl = dict(zip(p_names, p_vals_pl))
ndim = utils.get_params(params_pl, k_indices).size

data_pl_nl, Beane_pl_nl, LSE_pl_nl, MCMC_pl_nl = analysis.run_analysis(k_indices, spectra_pl, params_pl,
                                                                frac_pess, model, N_modes=N_modes_small, noiseless=True)

data_pl_op, Beane_pl_op, LSE_pl_op, MCMC_pl_op = analysis.run_analysis(k_indices, spectra_pl, params_pl,
                                                                frac_op, model, N_modes=N_modes_small)

data_pl_con, Beane_pl_con, LSE_pl_con, MCMC_pl_con = analysis.run_analysis(k_indices, spectra_pl, params_pl,
                                                                frac_con, model, N_modes=N_modes_small)

data_pl_pess, Beane_pl_pess, LSE_pl_pess, MCMC_pl_pess = analysis.run_analysis(k_indices, spectra_pl, params_pl,
                                                                frac_pess, model, N_modes=N_modes_small)

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

biases_bt = utils.extract_bias(k_indices, spectra_bt, P_m)
p_vals_bt = np.asarray([*biases_bt, P_m], dtype=object)

params_bt = dict(zip(p_names, p_vals_bt))
ndim = utils.get_params(params_bt, k_indices).size

data_bt_nl, Beane_bt_nl, LSE_bt_nl, MCMC_bt_nl = analysis.run_analysis(k_indices, spectra_bt, params_bt,
                                                                frac_pess, model, N_modes=N_modes_small, noiseless=True)

data_bt_op, Beane_bt_op, LSE_bt_op, MCMC_bt_op = analysis.run_analysis(k_indices, spectra_bt, params_bt,
                                                                frac_op, model, N_modes=N_modes_small)

data_bt_con, Beane_bt_con, LSE_bt_con, MCMC_bt_con = analysis.run_analysis(k_indices, spectra_bt, params_bt,
                                                                frac_con, model, N_modes=N_modes_small)

data_bt_pess, Beane_bt_pess, LSE_bt_pess, MCMC_bt_pess = analysis.run_analysis(k_indices, spectra_bt, params_bt,
                                                                frac_pess, model, N_modes=N_modes_small)

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
