import numpy as np
import matplotlib.pyplot as plt
import emcee

from functools import reduce # only in Python 3

def autocorrelation(signal):
    return signal * signal.conj()

def correlation(signal1, signal2):
    return signal1 * signal2.conj()

# k_vector shape is (N_dim, M_samples)
#def bin_k(k_vector, S_k, bins=5):
#k = np.fft.fftshift(np.sqrt((k_vec**2).sum(axis=0)))

def pspec(r_vec, signals, n_bins=5, bin_scale='log'):
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

    max_k = min([k_vec[i].max() for i in range(k_vec.shape[0])])

    bin_edges = np.zeros(n_bins)
    if bin_scale is 'linear':
        bin_edges = linear_bins(resolution(r_vec), max_k, n_bins)

    elif bin_scale == 'log':
        bin_edges = log_bins(resolution(r_vec), max_k, n_bins)

    elif bin_scale == 'custom':
        pass

    bins = [ [] for i in range(n_bins) ]
    bins_dimless = [[] for i in range(n_bins)]

    S_k1 = pixel_volume(r_vec) * np.fft.fftn(S_r1)
    S_k2 = pixel_volume(r_vec) * np.fft.fftn(S_r2)

    S_k1 = np.fft.fftshift(S_k1)
    S_k2 = np.fft.fftshift(S_k2)

    two_point_k = np.abs(S_k1 * np.conj(S_k2))

    print('min resolution: k=', resolution(r_vec))
    for index, k in np.ndenumerate(K):
        if k <= max_k:
            if k > resolution(r_vec): # resolution scale of box
                bindex = np.searchsorted(bin_edges,k) - 1
                bins[bindex].append(two_point_k[index])
                bins_dimless[bindex].append(k**(r_vec.shape[0]) * two_point_k[index])

    for i in range(len(bins)):
        if not bins[i]:
            bins[i] = 0
            bins_dimless[i] = 0

    pspec = np.asarray([np.asarray(bins[i]).mean() for i in range(n_bins)]) / survey_volume
    stds = np.asarray([np.asarray(bins[i]).std() for i in range(n_bins)]) / survey_volume
    counts = np.asarray([np.asarray(bins[i]).size for i in range(n_bins)])
    pspec_dimless = np.asarray([np.asarray(bins_dimless[i]).mean() for i in range(n_bins)]) / survey_volume
    factor = 2 * np.pi**2
    # plt.plot(bin_edges, bin_avgs / survey_volume)

    return bin_edges, pspec / factor, pspec_dimless / factor  #, stds, counts #, two_point_k / survey_volume



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
    box_length = r[0].max() - r[0].min()
    return (2 * np.pi) / box_length

def log_prob(x, mu, cov):
    diff = x - mu
    return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))
