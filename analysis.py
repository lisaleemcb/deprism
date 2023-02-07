import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

import utils
import fitting

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

def estimate_errors(signal, frac_error=.01, priors_width=.10):
    sigma = frac_error * signal
    sigma[-1] = priors_width * signal[-1]
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

    return pspecs

def dimless(k, P):
    return (k**3 / (2 * np.pi**2)) * P

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

def var_Pii_Beane_et_al(spectra, P_N_i, P_N_j, P_N_k, N_modes, k_indices):
    #P_ii, P_ij, P_ik, P_jj, P_jk, P_kk = spectra

    P_ii = spectra[0][k_indices]
    P_jj = spectra[1][k_indices]
    P_kk = spectra[2][k_indices]
    P_ij = spectra[3][k_indices]
    P_jk = spectra[4][k_indices]
    P_ik = spectra[5][k_indices]

    P_ii_tot = P_ii + P_N_i[k_indices]
    P_jj_tot = P_jj + P_N_j[k_indices]
    P_kk_tot = P_kk + P_N_k[k_indices]

    var_P_ii = (P_ij / P_ik)**2 * (P_ik**2 + P_ii_tot * P_kk_tot) \
        + (P_ik / P_ij)**2 * (P_ij**2 + P_ii_tot * P_jj_tot) \
        + (P_ij * P_ik / P_jk**2)**2 * (P_jk**2 + P_jj_tot * P_kk_tot) \
        + (P_ij * P_ik / P_jk**2) * (P_ii_tot * P_jk + P_ij * P_ik) \
        - (P_ij**2 * P_ik / P_jk**3) * (P_kk_tot * P_ik + P_ik * P_jk) \
        - (P_ij * P_ik**2 / P_jk**3) * (P_jj_tot * P_ik + P_ij * P_jk) \

    return var_P_ii / N_modes[k_indices]

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

def get_std(spectra, frac_error, k_indices):
    std = spectra * frac_error

    return std

def initialize_data(spectra, params, N, k_indices):
    data = utils.fetch_data(k_indices, spectra, b_0=params['b_i'])
    data = data + utils.generate_noise(N)

    return data

def plot_corner(MCMC, LSE, Beane, params, logp, P_ii, k_indices, ccolor=None, fig=None, limits=None):
    # truth stuff
    pvals = utils.get_pvals(params, k_indices)
    ndim = len(pvals)
    print([*pvals, *P_ii[k_indices]])

    # MCMC stuff
    samples = MCMC
    samples_00 = add_P(samples, k_indices, (0,0))

    # LSE stuff
    LSE_params, LSE_var = LSE
    LSE_params, LSE_var = utils.add_P_ii([LSE_params, LSE_var])

    # Beane et al stuff
    Beane_params, Beane_var = Beane

    # figure initialization
    plt.style.use('seaborn-colorblind')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if fig:
        fig = corner.corner(samples_00,
                                   truths=[*pvals, *P_ii[k_indices]], truth_color='black',
                                   plot_datapoints=False, color=ccolor, bins=30,
                                   labels=[r'$b_i$', r'$b_j$', r'$b_k$', r'$P_m$', r'$P_{ij}$'],
                                   label_kwargs={'fontsize': 30}, range=limits, fig=fig)

    if fig is None:
        fig = corner.corner(samples_00,
                                   truths=[*pvals, *P_ii[k_indices]], truth_color='black',
                                   plot_datapoints=False, color=ccolor, bins=30,
                                   labels=[r'$b_i$', r'$b_j$', r'$b_k$', r'$P_m$', r'$P_{ij}$'],
                                   label_kwargs={'fontsize': 30}, range=[.99,.99,.99,.99,.99])



    axes = np.array(fig.axes).reshape((ndim+1, ndim+1))
    limits = [axes[i,i].get_xlim() for i in range(ndim+1)]

    # Loop over the diagonal
    for i in range(ndim+1):
        ax = axes[i, i]
        ax.axvline(LSE_params[i], color=ccolor, alpha=.5)
        ax.axvline(LSE_params[i] - np.sqrt(LSE_var[i]), color=ccolor, ls=':', alpha=.5)
        ax.axvline(LSE_params[i] + np.sqrt(LSE_var[i]), color=ccolor, ls=':', alpha=.5)

        #ax.axvline(samples_00[logp.argmax(), i], color='red', alpha=.5)


    # Loop over the histograms
    for yi in range(ndim+1):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(LSE_params[xi], color=ccolor, alpha=.5)
            ax.axvline(LSE_params[xi] - np.sqrt(LSE_var[xi]), color=ccolor, ls=':', alpha=.5)
            ax.axvline(LSE_params[xi] + np.sqrt(LSE_var[xi]), color=ccolor, ls=':', alpha=.5)
            #ax.axvline(value2[xi], color="r")
            ax.axhline(LSE_params[yi], color=ccolor, alpha=.5)
            ax.axhline(LSE_params[yi] - np.sqrt(LSE_var[yi]), color=ccolor, ls=':', alpha=.5)
            ax.axhline(LSE_params[yi] + np.sqrt(LSE_var[yi]), color=ccolor, ls=':', alpha=.5)
            #ax.axhline(value2[yi], color="r")
            #ax.plot(LSE_params[xi], LSE_params[yi], color=color)
           # ax.plot(value2[xi], value2[yi], "sr")

    #line2 = axes[-1,-1].axvline(Beane_params, color=colors[2], ls='--', alpha=.5, dashes=(5, 5))
#    axes[-1,-1].axvline(Beane_params + np.sqrt(Beane_var), color=colors[2], ls=':', alpha=.5)
#    axes[-1,-1].axvline(Beane_params - np.sqrt(Beane_var), color=colors[2], ls=':', alpha=.5)

    #figure.legend()
    return fig, limits

def noisey_spectra(spectra, N=None, frac_error=None, noise='on'):

    if frac_error is not None and N is not None:
        raise Exception("Both noise scenarios provided. Commit to one!")

    if frac_error is not None:
        std_spectra = np.array(spectra) * frac_error

    if noise is 'on':
        std_spectra = np.sqrt(np.diag(N))
        noisey_spectra = np.array(spectra) + np.random.normal(scale=std_spectra, size=std_spectra.shape)

    if noise is 'off':
        noisey_spectra = np.array(spectra)

    return noisey_spectra, std_spectra

def calc_var(P_ii, P_jj, P_ij, N_i, N_j, N_modes):
    # var_ij = P_ii * P_jj + P_ii * N_j + P_jj * N_i + N_i * N_j - P_ij**2

    var_ij =((P_ii + N_i) * (P_jj + N_j) + P_ij**2) / (2 * N_modes)

    return var_ij

def get_noise(k_indices, spectra, b_0, N_modes, frac_error=.1, priors_width=.1):
    k_indices = k_indices[0]

    std = (np.asarray(spectra) * frac_error)
    var_b_0 = (priors_width * b_0)**2

    P_ii = spectra[0][k_indices]
    P_jj = spectra[1][k_indices]
    P_kk = spectra[2][k_indices]
    P_ij = spectra[3][k_indices]
    P_jk = spectra[4][k_indices]
    P_ik = spectra[5][k_indices]

    n = np.asarray([std[0][k_indices], std[1][k_indices], std[4][k_indices], std[2][k_indices], np.sqrt(var_b_0)])
    N = np.identity(n.size)

    N_i, N_j, N_k = std[0][k_indices], std[3][k_indices], std[5][k_indices]

    N_ii = calc_var(P_ii, P_ii, P_ii, N_i, N_i, N_modes[k_indices])
    N_ij = calc_var(P_ii, P_jj, P_ij, N_i, N_j, N_modes[k_indices])
    N_jk = calc_var(P_jj, P_kk, P_jk, N_j, N_k, N_modes[k_indices])
    N_ik = calc_var(P_ii, P_kk, P_ik, N_i, N_k, N_modes[k_indices])

    N = np.array([N_ii, N_ij, N_jk, N_ik, var_b_0]) * N

    return N, np.sqrt(np.asarray([N_i, N_j, N_k]))

def inject_noise(data, N):
    std = np.sqrt(np.diag(N))
    n = np.random.normal(scale=std)

    noisey_data = data + n

    return noisey_data

def run_analysis(k_indices, spectra, params_dict, frac_error, model, N_modes=None,
                    data=None, error_x=True, priors_width=.10, priors_offset=1.0,
                    noiseless=False, nsteps=1e6):

    if data is None:
        data = utils.fetch_data(k_indices, spectra, b_0=params_dict['b_i'])

    if error_x is False:
        N, n = get_noise(k_indices, spectra, params_dict['b_i'], N_modes,
                                            frac_error=frac_error, priors_width=priors_width)
    if error_x is True:
        N = estimate_errors(data, frac_error=frac_error, priors_width=priors_width)
        n = [spectra[0] * frac_error, spectra[3] * frac_error, spectra[5] * frac_error]

    if not noiseless:
        print('adding noise to simulation...')
        data = inject_noise(data, N)
        data[-1] = params_dict['b_i']

    else:
        print('noiseless run, easy breezy!')

    Beane = fitting.Beane_et_al(data, spectra, n[0], n[1], n[2], N_modes, k_indices)
    LSE = fitting.LSE_results(k_indices, data, N)
    MCMC = fitting.MCMC_results(params_dict, k_indices, data, model, N,
                                priors_offset * params_dict['b_i'], priors_width=priors_width,
                                nsteps=nsteps)

    return data, Beane, LSE, MCMC

def keep_P_21(k_indices, spectra, params, noise, model, N_modes=None, nsteps=1e6,
                                        noiseless=False, priors_offset=1.0, priors_width=.1):

    data, Beane, LSE, MCMC = run_analysis(k_indices, spectra, params, noise, model,
                                        N_modes=N_modes, noiseless=noiseless,
                                        priors_offset=priors_offset, priors_width=.1,
                                        nsteps=nsteps)

    samples_00 = add_P(MCMC[0], k_indices, (0,0))


    return data, Beane, LSE, (samples_00, MCMC[1]) #Beane, [np.median(samples_00), samples_00[MCMC[1].argmax(),-1],
                                    #    np.std(samples_00[:,-1])]


def plot_all(samples, k_indices):
    samples_00 = add_P(samples, k_indices, (0,0))
    samples_01 = add_P(samples_00, k_indices, (0,1))
    samples_02 = add_P(samples_01, k_indices, (0,2))
    samples_12 = add_P(samples_02, k_indices, (1,2))

    return samples_12
