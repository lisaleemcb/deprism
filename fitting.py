import numpy as np
import matplotlib.pyplot as plt
import emcee
import copy as cp
import utils
import analysis
import estimators

from multiprocessing import Pool

def log_prior(param_guesses, params, k_indices, model, N, b0_guess,
                priors='uniform', priors_width=.25, positivity=False,
                b_j_prior=False, b_k_prior=False):

    P_m = param_guesses['P_m']
    b_0 = b0_guess #params['b_i'] #* .92 # 2660
    b_i = param_guesses['b_i']
    b_j = param_guesses['b_j']
    b_k = param_guesses['b_k']

    # print(b_0)

    if priors == 'uniform':
        b_0 = params['b_i']
        if positivity:
            if np.any(np.asarray(b_i) < 0):
                return -np.inf
            if np.any(np.asarray(b_j) < 0):
                return -np.inf
            if np.any(np.asarray(b_k) < 0):
                return -np.inf

        if np.any(np.asarray(P_m) < 0):
            return -np.inf

        if b_i < (b_0 - b_0 * priors_width) or b_i > (b_0 + priors_width * b_0):
            return -np.inf

        return 0.0

    if priors == 'upperlimit':
        P_21_upper = 5.0
        P_21_guess = b_i**2 * P_m
        if np.any(np.asarray(P_m) < 0):
            return -np.inf
        if P_21_guess > P_21_upper:
            return -np.inf

        return 0.0

    if priors == 'gaussian':
        if positivity is True:
            if np.any(np.asarray(b_i) < 0):
                return -np.inf
            if np.any(np.asarray(b_j) < 0):
                return -np.inf
            if np.any(np.asarray(b_k) < 0):
                return -np.inf

        #if np.any(np.asarray(P_m) < 0):
        #    return -np.inf
        p_i = np.log(utils.gaussian(b_i, b0_guess, b0_guess * priors_width))
        p_j = 0
        p_k = 0

        if b_j_prior:
            p_j = np.log(utils.gaussian(params['b_j'], b_j, params['b_j'] * priors_width))
        if b_k_prior:

            p_k = np.log(utils.gaussian(params['b_k'], b_k, params['b_k'] * priors_width))

        return p_i + p_j + p_k

    if priors == 'jeffreys':
        #print(param_guesses)
        if np.any(np.asarray(P_m) < 0):
            return -np.inf
        #if b_i < b_0 * .9 or b_i > b_0 * 1.1:
        #    return -np.inf
        Fisher_matrix = utils.Fisher(model.pspec, param_guesses, N, k_indices)
        Fisher_0 = utils.Fisher(model.pspec, params, N, k_indices)

        Jeffreys = np.sqrt(Fisher_matrix[0,0]) / np.sqrt(Fisher_0[0,0])
        if not np.isfinite(Jeffreys):
            #print('Jeffreys: ', Jeffreys)
            return -np.inf
            #print('params: ', param_guesses)
        print(Jeffreys)
        return Jeffreys

    if priors == 'adhoc':
        if np.any(np.asarray(P_m) < 0):
            return -np.inf
        if b_i < b_0 * .75 or b_i > b_0 * 1.25:
            return -np.inf

        return -1 / P_m[k_indices]

def log_likelihood(param_guesses, k_indices, data, model, N,
                    pdf='gaussian'):

    if pdf == 'gaussian':
        pspec = model.pspec(k_indices, params=param_guesses)

        #print('pspec: ', pspec)
        #print('data: ', data)
        diff = data - pspec
    #print('X2 for bias: ', (-.5 * diff[-1]**2) / N[-1,-1])
    #print('log_likelihood: ', -0.5 * np.dot(diff, np.linalg.solve(N, diff)))

        return -0.5 * np.dot(diff, np.linalg.solve(N, diff))

    if pdf == 'quad':
        pspec = model.pspec(k_indices, params=param_guesses)

    #    print('pspec: ', pspec)
    #    print('data: ', data)

        diff = data - pspec
        sigma = np.sqrt(np.diag(N))
        #print(diff)
        #print(sigma)
    #print('X2 for bias: ', (-.5 * diff[-1]**2) / N[-1,-1])
    #print('log_likelihood: ', -0.5 * np.dot(diff, np.linalg.solve(N, diff)))

        return -0.5 * np.dot(diff, np.linalg.solve(N, diff)).sum()

def log_prob(guesses, params, k_indices, data, model, N, b0_guess,
                    priors='gaussian', priors_width=.25,
                    positivity=False, pdf='gaussian', b_j_prior=False, b_k_prior=False):

    param_guesses = cp.deepcopy(params)

    for i, (names, vals) in enumerate(params.items()):
        if names != 'P_m':
            param_guesses[names] = guesses[i]

        if names == 'P_m':
            P_m_params = guesses[i:]
            for j, k in enumerate(k_indices):
                param_guesses['P_m'][k] = P_m_params[j]

    lp = log_prior(param_guesses, params, k_indices, model, N, b0_guess,
                    priors=priors, priors_width=priors_width, positivity=positivity,
                    b_j_prior=b_j_prior, b_k_prior=b_k_prior)
    if not np.isfinite(lp):
        return -np.inf

    #print(lp)
    #print('param_guesses', param_guesses)

    return lp + log_likelihood(param_guesses, k_indices, data, model, N,
                                pdf=pdf)

def start_mcmc(params_init, k_indices, data, model, N, b0_guess, p0_in=None,
                priors='gaussian', priors_width=.1, positivity=False,
                pdf='gaussian', backend_filename=None, nsteps=1e6, nwalkers=48,
                burn_in=1, parallel=False, start_from_backend=False,
                b_j_prior=False, b_k_prior=False):

    print('running mcmc with the following settings:')
    print('fitting data from k: ', k_indices)
    print('burn_in is:', burn_in)
    print('prior is: ', priors)
    print('prior guess is:', b0_guess)
    print('prior width is: ', priors_width)
    print('positivity prior is: ', positivity)
    print('pdf is: ', pdf)
    print('nsteps: ', nsteps)
    print('nwalkers:', nwalkers)
    print('logp of truths is:', log_prob(utils.get_params(params_init, k_indices),
                            params_init, k_indices, data, model, N, b0_guess,
                            priors=priors, priors_width=priors_width,
                            positivity=positivity, pdf=pdf,
                            b_j_prior=b_j_prior, b_k_prior=b_k_prior))

    pvals = np.asarray(list(params_init.values()), dtype=object)

    args = [params_init, k_indices, data, model, N, b0_guess, priors,
                priors_width, positivity, pdf, b_j_prior, b_k_prior]

    ndim = len(pvals) - 1 + len(pvals[-1][k_indices])

    p0 = np.zeros((nwalkers, ndim))

    n_biases = len(pvals) - 1
    for i in range(n_biases):
        p0[:,i] = pvals[i]

    n_bins = len(pvals[-1][k_indices])
    for i in range(n_bins):
        p0[:,-n_bins + i] =  pvals[-1][k_indices[i]]

    delta_p = np.zeros_like(p0)
    for i, val in enumerate(p0[0]):
        std_dev = .1 * val
        delta_p[:,i] = np.random.normal(scale=std_dev, size=nwalkers)

    params0 = p0 + delta_p
    #print(params0)

    if backend_filename is not None:
        if start_from_backend:
            filename = backend_filename
            backend = emcee.backends.HDFBackend(filename)
            #print(backend.chain)
            print('pickup up from backend file...', str(filename))

        if not start_from_backend:
            filename = backend_filename
            backend = emcee.backends.HDFBackend(filename)
            #print(backend.chain)
            backend.reset(nwalkers, ndim)
            print('backend initialized...', str(filename))

    if backend_filename is None:
        backend = None
        print('no backend initialized')

    if not parallel:
        if p0_in is not None:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob,
                    args=args)

            pre_state = sampler.run_mcmc(p0_in, burn_in)

        else:

            sampler = emcee.EnsembleSampler(nwalkers, int(ndim), log_prob,
                    args=args)

            pre_state = sampler.run_mcmc(params0, int(burn_in))

        #print("Mean acceptance fraction during burnin: {0:.3f}".format(
        #np.mean(sampler.acceptance_fraction)))

        sampler.reset()
        state = sampler.run_mcmc(pre_state, int(nsteps), progress=True)

        #print("Mean acceptance fraction: {0:.3f}".format(
        #np.mean(sampler.acceptance_fraction)))

        #print("Mean autocorrelation time: {0:.3f} steps".format(
        #np.mean(sampler.get_autocorr_time())))

        return sampler.get_chain(thin=100, flat=True), sampler.get_log_prob(thin=100, flat=True)

    if parallel:
        print('We are going parallelized! Wooooooo...')
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob,
                    args=args, pool=pool)

            state = sampler.run_mcmc(params0, burn_in)
            sampler.reset()
            check = sampler.run_mcmc(state, nsteps)

            return sampler.get_chain(thin=100, flat=True), sampler.get_log_prob(thin=100, flat=True)

def many_realizations(params_initial, param_names, k_indices,
                                data, model, N, truths,
                                nsteps=1e5, nwalkers=32, burn_in=1000, runs=10, parallel=False):

    samples = np.zeros((runs, len(params_initial)))
    log_prob = np.zeros((runs))

    for i in range(runs):
        data_noise = data + np.random.normal(scale=np.sqrt(N[0,0]), size=len(data))
        samples, log_prob = run_mcmc(params_initial, param_names, k_indices,
                        data, model, N, truths, nsteps=1e5, nwalkers=32, burn_in=1000, parallel=parallel)



    return samples, log_prob

def recover_params_mcmc(k, k_indices, lumen_pspecs, model, density, variances,
            priors='uniform', priors_width=.25, N=None, N_frac_error=None, inject_noise=False,
            pdf='gaussian', positivity=False, nsteps=1e5, nwalkers=72,
            b_j_prior=False, b_k_prior=False):

    data = utils.fetch_data(k, k_indices, lumen_pspecs)

    if N is None:
        if N_frac_error is not None:
            print('Noise level is: ', N_frac_error * 100, '% of specific_intensity')
            N = analysis.estimate_errors(data, frac_error=N_frac_error)

        if N_frac_error is None:
            print('Noise level is set by instrumental specifications')
            N = analysis.create_noise_matrix(k_indices, variances)

    biases = utils.extract_bias(k_indices, lumen_pspecs, density)

    p_names = np.asarray(['b_i','b_j', 'b_k', 'P_m'])
    pvals = np.zeros(len(p_names), dtype=object)

    for i in range(pvals.size-1):
        pvals[i] = biases[i]

    pvals[-1] = density
    data[-1] = biases[0]

    #print(data)
    #print(np.diag(N))

    params = dict(zip(p_names, pvals))

    if inject_noise:
        data = data + utils.generate_noise(N)

    # lopping off the bias
    data_size = model.pspec(k_indices).size
    data = data[1:data_size*len(k_indices)+1]
    N = N[1:data_size*len(k_indices)+1,1:data_size*len(k_indices)+1]
    #print('DATA SIZE: ', data_size)

    print('PARAMS: ', pvals)
    print('DATA: ', data)
    print('NOISE: ',np.diag(N))

    results = start_mcmc(params, k_indices, data, model, N,
                            burn_in=5000, nsteps=nsteps,
                            nwalkers=nwalkers, parallel=False,
                            pdf=pdf, priors=priors, priors_width=priors_width,
                            positivity=positivity, b_j_prior=b_j_prior, b_k_prior=b_k_prior)

    return results, params, data

def recover_params_LSE(k, k_indices, lumen_pspecs, model, density, variances,
                            N=None, inject_noise=False):
    data = utils.fetch_data(k, k_indices, lumen_pspecs)
    biases = utils.extract_bias(k_indices, lumen_pspecs, density)

    data[-1] = biases[0]

    if N == None:
        N = analysis.estimate_errors(data, frac_error=.20)
    #N = analysis.create_noise_matrix(k_indices, variances)

    N[-1,-1] = (biases[0] * .1)**2
    N = N[1:,1:]

    if inject_noise:
        'adding noise'
        data = data + utils.generate_noise(N)
    # print('LSE data: ', data_HI_L_M)
    # print('LSE biases: ,', biases_HI_L_M)

    print('LSE DATA: ', data[1:])
    print('LSE NOISE: ', np.diag(N))
    print('noise has shape ', N.shape)

    LSE = estimators.Estimators(k_indices, data[1:], N)
    results = LSE.LSE_3cross_1bias()

    params, errors = utils.delog_results(results, np.diag(N))

    return params, errors

def LSE_results(k_indices, params_dict, data_noise, data_nl, N):
    p_true = utils.get_params(params_dict, k_indices)

    LSE_noise = N[1:,1:] / data_nl[1:]**2
    LSE = estimators.Estimators(k_indices, data_noise[1:], LSE_noise)
    results = LSE.LSE_3cross_1bias()
    p_, e_ = utils.delog_results(results, np.diag(LSE_noise))

    print('k_index:', k_indices)
    print('LSE DATA: ', data_noise[1:])
    print('LSE DATA / DATA_NL: ', data_noise[1:] / data_nl[1:])
    print('LSE NOISE: ', np.diag(LSE_noise))

    params = np.zeros((p_.size + 1))
    errors = np.zeros((e_.size + 1))

    for i in range(p_.size):
        params[i] = p_[i]
        errors[i] = p_true[i]**2 * e_[i]

    params[-1] = p_[0]**2 * p_[3]

    var_ii = (np.diag(LSE_noise)[0] + np.diag(LSE_noise)[1] + np.diag(LSE_noise)[2]) * data_nl[0]**2
    errors[-1] = var_ii

    return params, errors

def Beane_et_al(data, spectra, P_N_i, P_N_j, P_N_k, N_modes, k_indices):
    P_ij = data[1]
    P_jk = data[2]
    P_ik = data[3]

    P_ii = P_ij * P_ik / P_jk
    var = analysis.var_Pii_Beane_et_al(spectra, P_N_i, P_N_j, P_N_k, N_modes, k_indices)

    return P_ii, var[0]

def MCMC_results(params, k_indices, data, model, N, b0_guess, p0_in=None,
                priors='gaussian', priors_width=.1, positivity=True,
                pdf='gaussian', backend_filename=None, nsteps=1e6, nwalkers=48,
                burn_in=500, parallel=False, b_j_prior=False, b_k_prior=False):
    # lopping off the bias
    #b0_guess = cp.deepcopy(data[-1])
    data_size = model.pspec(k_indices).size
    data = data[1:data_size*len(k_indices)+1]
    N = N[1:data_size*len(k_indices)+1,1:data_size*len(k_indices)+1]
    #print('DATA SIZE: ', data_size)

    print('PARAMS: ', params)
    print('DATA: ', data)
    print('NOISE / DATA: ',np.sqrt(np.diag(N))/ data)
    print('PRIOR RANGE: ', priors_width)
    print('PRIOR GUESS:', b0_guess)
    print('PRIOR on J is:', b_j_prior)
    print('PRIOR on K is:', b_k_prior)

    results = start_mcmc(params, k_indices, data, model, N, b0_guess, p0_in=p0_in,
                            burn_in=burn_in, nsteps=nsteps,
                            nwalkers=nwalkers, parallel=parallel,
                            pdf=pdf, priors=priors, priors_width=priors_width,
                            positivity=positivity, backend_filename=backend_filename,
                            b_j_prior=b_j_prior, b_k_prior=b_k_prior)

    return results
