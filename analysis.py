import numpy as np
import matplotlib.pyplot as plt
import emcee

from functools import reduce # only in Python 3
from scipy.special import comb

def autocorrelation(signal):
    return signal * signal.conj()

def correlation(signal1, signal2):
    return signal1 * signal2.conj()

# k_vector shape is (N_dim, M_samples)
#def bin_k(k_vector, S_k, bins=5):
#k = np.fft.fftshift(np.sqrt((k_vec**2).sum(axis=0)))

def calc_pspec(r_vec, signals, n_bins=5, bin_scale='log'):
    S_r1 = 0
    S_r2 = 0

    if len(signals) is 1:
        S_r1 = signals[0]
        S_r2 = signals[0]

    if len(signals) is 2:
        S_r1 = signals[0]
        S_r2 = signals[1]

    k_vec = np.zeros_like(r_vec)

    for i in range(r_vec.shape[0]):
        k_vec[i] = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(r_vec[i].size,
                    (r_vec[i].max() - r_vec[i].min()) / r_vec[i].size ))

    K_vec = np.meshgrid(*[k_vec[i] for i in range(len(k_vec))])
    K = np.sqrt((np.asarray(K_vec)**2).sum(axis=0))
    #survey_n_pixels = np.asarray([k_vec[i].size for i in range(k_vec[0]).size]).prod()
    #print(survey_n_pixels)
    survey_volume = pixel_volume(r_vec) * K_vec[0].size

    print('voxel size is: ', r_vec[1,1] - r_vec[0,0], ' Mpc')
    print('voxel volume is: ', pixel_volume(r_vec), ' Mpc^3')
    print('survey volume is: ', survey_volume, ' Mpc^3')

    max_k = min([k_vec[i].max() for i in range(k_vec.shape[0])])

    bin_edges = np.zeros(n_bins)
    if bin_scale is 'linear':
        print('bin scale is: linear')
        bin_edges = linear_bins(resolution(r_vec), max_k, n_bins)

    elif bin_scale == 'log':
        print('bin scale is: log')
        bin_edges = log_bins(resolution(r_vec), max_k, n_bins)

    elif bin_scale == 'custom':
        pass

    bins = [ [] for i in range(n_bins) ]

    S_k1 = pixel_volume(r_vec) * np.fft.fftn(S_r1)
    S_k2 = pixel_volume(r_vec) * np.fft.fftn(S_r2)

    S_k1 = np.fft.fftshift(S_k1)
    S_k2 = np.fft.fftshift(S_k2)

    two_point_k = np.abs(S_k1 * np.conj(S_k2))

    print('min resolution: k=', resolution(r_vec))
    print('maximum k-mode:', max_k)
    for index, k in np.ndenumerate(K):
        if k <= max_k:
            if k > resolution(r_vec): # resolution scale of box
                bindex = np.searchsorted(bin_edges,k) - 1
                bins[bindex].append(two_point_k[index])

    for i in range(len(bins)):
        if not bins[i]:
            bins[i] = 0

    pspec = np.asarray([np.asarray(bins[i]).mean() for i in range(n_bins)]) / survey_volume
    # stds = np.asarray([np.asarray(bins[i]).std() for i in range(n_bins)]) / survey_volume
    # counts = np.asarray([np.asarray(bins[i]).size for i in range(n_bins)])
    # plt.plot(bin_edges, bin_avgs / survey_volume)

    return bin_edges, pspec #, stds, counts #, two_point_k / survey_volume

# equal spaced in log(k)
# arbitrary bins
# hybrid-linear log bins


    #index_min = next(i for i,k in enumerate(k) if k>=bin_min)
#    index_max = next(i for i,k in enumerate(k) if k>bin_max) - 1

    #print(K[index], val)

def pixel_volume(r):
    n_dim = len(r)
    lengths = np.zeros((n_dim, 1))
    for i in range(n_dim):
        lengths[i] = np.abs(r[i][0] - r[i][1])

    pixel_volume = reduce(lambda n1, n2: n1 * n2, lengths)

    return pixel_volume

def linear_bins(min_k, max_k, n_bins):
    delta_k = max_k / n_bins
    bins = delta_k * np.arange(n_bins)
    bins[0] = min_k

    return bins

def log_bins(min_k, max_k, n_bins):
    return np.geomspace(min_k, max_k,n_bins)

def resolution(r):
    # for a square volume
    box_length = r[0].max() - r[0].min()

    return (2 * np.pi) / box_length

def log_prob(x, mu, cov):
    diff = x - mu
    return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))

def estimate_errors(signal, frac_error=.01):
    sigma =  frac_error * signal
    N = sigma**2 * np.identity(signal.size)

    return N

def create_noise_matrix(k_indices, variances):
    n_data = 5 # set to 4, 3 crosscorrelations and 1 bias

    N = np.identity(n_data)
    print

    for i in range(4):
        N[i,i] = variances[i][k_indices][0].value

    N[-1,-1] = variances[-1][k_indices]

    return N

def calc_errors(signal, specs):
    pass

def N_pixel_thermal(T_sys, Delta_nu, t_int, f_cover):
    sigma_N = T_sys / (np.sqrt(Delta_nu * t_int) * f_cover)

    return sigma_N

def gen_spectra(r_vec, fields, runs=3, n_bins=20):
    lines_indices = np.zeros((int(comb(runs, 2) + runs), 2))
    pspecs = np.zeros((int(comb(runs, 2) + runs), n_bins))

    counter = 0
    for i in range(runs):
        for j in range(i, runs):
            print('Calculating correlation for Lines', i, ' and ', j)
            k, pspec = calc_pspec(r_vec,
                            [fields[i], fields[j]],
                            n_bins=n_bins, bin_scale='log')

            pspecs[counter] = pspec

            lines_indices[counter,0] = i
            lines_indices[counter,1] = j

            counter += 1

    return k, pspecs

def add_P(samples, k_indices, lines):
    n, m = lines
    n_samples = samples[:,0].size
    n_params = samples[0].size + len(k_indices)

    samples_with_P = np.zeros((n_samples, n_params))

    for i in range(n_params - len(k_indices)):
        samples_with_P[:,i] = samples[:,i]

    for i in range(len(k_indices)):
        samples_with_P[:, n_params - i - 1] = samples[:,n] * samples[:,m] * samples[:,(2 + len(k_indices) - i)]

    return samples_with_P

def var_Pii_Beane_et_al(spectra, P_N_i, P_N_j, P_N_k, N_modes):
    P_ii, P_ij, P_ik, P_jj, P_jk, P_kk = spectra

    P_ii = P_ii[:-1]
    P_jj = P_jj[:-1]
    P_kk = P_kk[:-1]
    P_ij = P_ij[:-1]
    P_jk = P_jk[:-1]
    P_ik = P_ik[:-1]

    P_ii_tot = P_ii + P_N_i
    P_jj_tot = P_jj + P_N_j
    P_kk_tot = P_kk + P_N_k

    var_P_ii = (P_ij / P_ik)**2 * (P_ik**2 + P_ii_tot * P_kk_tot) \
        + (P_ik / P_ij)**2 * (P_ij**2 + P_ii_tot * P_jj_tot) \
        + (P_ij * P_ik / P_jk**2)**2 * (P_jk**2 + P_jj_tot * P_kk_tot) \
        + (P_ij * P_ik / P_jk**2) * (P_ii_tot * P_jk + P_ij * P_ik) \
        - (P_ij**2 * P_ik / P_jk**3) * (P_kk_tot * P_ik + P_ik * P_jk) \
        - (P_ij * P_ik**2 / P_jk**3) * (P_jj_tot * P_ik + P_ij * P_jk) \

    return var_P_ii / N_modes

def check_convergence(mcmc_data):
    results, params, data = mcmc_data
    samples, lnprob = results

    pass

def Fisher_analytic(params, N, k_indices, priors_width=.1):
    b_i = params['b_i']
    b_j = params['b_j']
    b_k = params['b_k']
    P_m = np.float64(params['P_m'][k_indices])

    var_bi = (b_i * priors_width)**2
    var_ii = N[0,0]
    var_ij = N[1,1]
    var_jk = N[2,2]
    var_ik = N[3,3]

    Fisher_matrix = np.array([[(b_j * P_m)**2 / var_ij + (b_k * P_m )**2/ var_ik + var_bi**-1,
                                b_i * b_j * P_m**2 / var_ij,
                                b_i * b_k * P_m**2 / var_ik,
                                b_i * b_j**2 * P_m / var_ij + b_i * b_k**2 * P_m / var_ik],
                     [b_i * b_j * P_m**2 / var_ij,
                                (b_i * P_m)**2 / var_ij + (b_k * P_m)**2 / var_jk,
                                b_j * b_k * P_m**2 / var_jk,
                                b_i**2 * b_j * P_m / var_ij + b_j * b_k**2 * P_m / var_jk],
                     [b_i * b_k * P_m**2 / var_ik,
                                b_j * b_k * P_m**2 / var_jk,
                                (b_i * P_m)**2 / var_ik + (b_j * P_m)**2 / var_jk,
                                b_i**2 * b_k * P_m / var_ik + b_j**2 * b_k * P_m / var_jk],
                     [b_i * b_j**2 * P_m / var_ij + b_i * b_k**2 * P_m / var_ik,
                                b_i**2 * b_j * P_m / var_ij + b_j * b_k**2 * P_m / var_jk,
                                b_i**2 * b_k * P_m / var_ik + b_j**2 * b_k * P_m / var_jk,
                                (b_i * b_j)**2 / var_ij + (b_i * b_k)**2 / var_ik + (b_j * b_k)**2 / var_jk]])


    return Fisher_matrix

def Fisher_num(function, params, noise, k_indices):
    sigma_squared = np.diag(noise)
    n_params = len(params.keys())

    Fisher_matrix = np.zeros((n_params, n_params))
    for i, key_i in enumerate(list(params.keys())):
        for j, key_j in enumerate(list(params.keys())):

            arg = (utils.central_diff(function, params, key_i, k_indices)[:]
                 * utils.central_diff(function, params, key_j, k_indices)[:] / sigma_squared[:])

            Fisher_matrix[i][j] = arg.sum()


    return Fisher_matrix
