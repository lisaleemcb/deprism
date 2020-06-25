import numpy as np
import emcee
import fitting, models

# Power Spectrum initialization
k = np.array([ 0.07853982,  0.10513562,  0.14073752,  0.18839523,  0.25219119,
        0.33759028,  0.45190791,  0.60493672,  0.80978543,  1.08400171,
        1.45107539,  1.94245061,  2.60021941,  3.48072734,  4.6594002 ,
        6.23720508,  8.34929937, 11.17660861, 14.96132484, 20.02765317])
P_m = np.ones(15) + np.random.normal(scale=.1, size=15) #P_m_dimless

# Scalar Bias initialization
pnames_scalarbias = ['b_i', 'b_j', 'b_k']
pvals_scalarbias = 1 + np.random.normal(scale=.1, size=3)
params_scalarbias = dict(zip(pnames_scalarbias, pvals_scalarbias))
params_scalarbias['P_m'] = P_m

# Degree One initialization
pnames_degreeone = ['a_i', 'c_i', 'a_j', 'c_j', 'a_k', 'c_k', 'P_m']
a_i, c_i = [np.random.rand() * 10, np.random.rand() * 10]
a_j, c_j = [np.random.rand() * 10, np.random.rand() * 10]
a_k, c_k = [np.random.rand() * 10, np.random.rand() * 10]
pvals_degreeone = [a_i, c_i, a_j, c_j, a_k, c_k, P_m]
params_degreeone = dict(zip(pnames_degreeone, pvals_degreeone))

# fits initialization
k_indices = [6,12]

model_scalarbias = models.ScalarBias(k, params_scalarbias)
model_degreeone = models.DegreeOneBias(k, params_degreeone)

data_scalarbias = models.ScalarBias(k, params_scalarbias).pspec(k_indices)
data_degreeone = models.DegreeOneBias(k, params_degreeone).pspec(k_indices)

params_init_scalarbias = [params_scalarbias['b_i'], params_scalarbias['b_k'], *params_scalarbias['P_m'][k_indices]]
params_init_degreeone = [params_degreeone['a_j'], params_degreeone['c_j'], params_degreeone['a_k'],
                        params_degreeone['c_k'], *params_degreeone['P_m'][k_indices]]

noise_vars = np.geomspace(1e-3, 10, 30)
noise = [noise_vars[i] * np.identity(len(data_scalarbias)) for i in range(len(noise_vars))]

runs = 200
nwalkers = 48
nsteps = 1e5

fits_scalarbias = [[] for i in range(0,len(noise_vars)),5]
for i in range(0,len(noise_vars),5):
    print('Now on run: ', i)

    fits_scalarbias[i] = fitting.many_realizations(params_init_scalarbias,
                    list(params_scalarbias.keys())[1:], k_indices, data_scalarbias, model_scalarbias,
                     noise[i], params_scalarbias, runs=runs, parallel=True)

np.save('fits_scalarbias.npy', fits_scalarbias)

fits_degreeone = [[] for i in range(0,len(noise_vars)),5]
for i in range(0,len(noise_vars),5):
    print('Now on run: ', i)

    fits_degreeone[i] = fitting.many_realizations(params_init_degreeone,
                    list(params_degreeone.keys())[2:], k_indices, data_degreeone, model_degreeone,
                     noise[i], params_degreeone, runs=runs, parallel=True)

np.save('fits_degreeone.npy', fits_degreeone)
