import numpy as np
import matplotlib.pyplot as plt
import emcee
import copy
import utils

from multiprocessing import Pool

def log_prior(param_guesses, params, k_indices, model, noise,
                priors='uniform', positivity=False):
    P_m = param_guesses['P_m']
    b_0 = 1.0 # params['b_i']
    b_i = param_guesses['b_i']
    b_j = param_guesses['b_j']
    b_k = param_guesses['b_k']

    if priors is 'uniform':
        b_0 = params['b_i']
        if positivity is True:
            if np.any(np.asarray(b_i) < 0):
                return -np.inf
            if np.any(np.asarray(b_j) < 0):
                return -np.inf
            if np.any(np.asarray(b_k) < 0):
                return -np.inf

        if np.any(np.asarray(P_m) < 0):
            return -np.inf

        if b_i < b_0 * .99 or b_i > b_0 * 1.01:
            return -np.inf

        return 0.0

    if priors is 'upperlimit':
        P_21_upper = 5.0
        P_21_guess = b_i**2 * P_m
        if np.any(np.asarray(P_m) < 0):
            return -np.inf
        if P_21_guess > P_21_upper:
            return -np.inf

        return 0.0

    if priors is 'gaussian':
        if positivity is True:
            if np.any(np.asarray(b_i) < 0):
                return -np.inf
            if np.any(np.asarray(b_j) < 0):
                return -np.inf
            if np.any(np.asarray(b_k) < 0):
                return -np.inf

        if np.any(np.asarray(P_m) < 0):
            return -np.inf

        return np.log(utils.gaussian(b_i, b_0, b_0 * 1.0))

    if priors is 'jeffreys':
        #print(param_guesses)
        if np.any(np.asarray(P_m) < 0):
            return -np.inf
        #if b_i < b_0 * .9 or b_i > b_0 * 1.1:
        #    return -np.inf
        Fisher_matrix = utils.Fisher(model.pspec, param_guesses, noise, k_indices)
        Fisher_0 = utils.Fisher(model.pspec, params, noise, k_indices)

        Jeffreys = np.sqrt(Fisher_matrix[0,0]) / np.sqrt(Fisher_0[0,0])
        if not np.isfinite(Jeffreys):
            #print('Jeffreys: ', Jeffreys)
            return -np.inf
            #print('params: ', param_guesses)
        print(Jeffreys)
        return Jeffreys

    if priors is 'adhoc':
        if np.any(np.asarray(P_m) < 0):
            return -np.inf
        if b_i < b_0 * .75 or b_i > b_0 * 1.25:
            return -np.inf

        return -1 / P_m[k_indices]

def log_likelihood(param_guesses, params, k_indices, data, model, noise,
                    pdf='gaussian'):

    if pdf is 'gaussian':
        pspec = model.pspec(k_indices, params=param_guesses)

    #    print('pspec: ', pspec)
    #    print('data: ', data)

        diff = data - pspec
    #print('X2 for bias: ', (-.5 * diff[-1]**2) / noise[-1,-1])
    #print('log_likelihood: ', -0.5 * np.dot(diff, np.linalg.solve(noise, diff)))

        return -0.5 * np.dot(diff, np.linalg.solve(noise, diff))

    if pdf is 'quad':
        pspec = model.pspec(k_indices, params=param_guesses)

    #    print('pspec: ', pspec)
    #    print('data: ', data)

        diff = data - pspec
        sigma = np.sqrt(np.diag(noise))
        #print(diff)
        #print(sigma)
    #print('X2 for bias: ', (-.5 * diff[-1]**2) / noise[-1,-1])
    #print('log_likelihood: ', -0.5 * np.dot(diff, np.linalg.solve(noise, diff)))

        return -0.5 * (np.dot(diff, np.linalg.solve(noise, diff)) + (diff**4 / sigma**4).sum())

def log_probability(guesses, params, k_indices, data, model, noise,
                    priors='gaussian', positivity=False, pdf='gaussian'):

    param_guesses = copy.deepcopy(params)

    for i, (names, vals) in enumerate(params.items()):
        if names != 'P_m':
            param_guesses[names] = guesses[i]

        if names == 'P_m':
            P_m_params = guesses[i:]
            for j, k in enumerate(k_indices):
                param_guesses['P_m'][k] = P_m_params[j]

    lp = log_prior(param_guesses, params, k_indices, model, noise,
                    priors=priors, positivity=positivity)
    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(param_guesses, params, k_indices, data, model, noise,
                                pdf=pdf)

def start_mcmc(params_init, k_indices, data, model, noise,
                priors='gaussian', positivity=False, pdf='gaussian',
                backend_filename=None, nsteps=1000, nwalkers=32,
                 burn_in=1000, parallel=False):

    print('running mcmc with the following settings:')
    print('fitting data from k: ', k_indices)
    print('prior is: ', priors)
    print('pdf is: ', pdf)
    print('positivity prior is: ', positivity)

    pvals = np.asarray(list(params_init.values()), dtype=object)

    args = [params_init, k_indices, data, model, noise, priors,positivity, pdf]

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
        std_dev = .01 * val
        delta_p[:,i] = np.random.normal(scale=std_dev, size=nwalkers)

    params0 = p0 + delta_p
    #print(params0)

    if backend_filename is not None:
        if start_from_backend is True:
            filename = backend_filename
            backend = emcee.backends.HDFBackend(filename)
            #print(backend.chain)
            print('pickup up from backend file...', str(filename))

        if start_from_backend is False:
            filename = backend_filename
            backend = emcee.backends.HDFBackend(filename)
            #print(backend.chain)
            backend.reset(nwalkers, ndim)
            print('backend initialized...', str(filename))

    if backend_filename is None:
        backend = None
        print('no backend initialized')


    if parallel is False:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                args=args)

        state = sampler.run_mcmc(params0, burn_in)
        sampler.reset()
        check = sampler.run_mcmc(state, nsteps)

        return sampler.get_chain(flat=True), sampler.get_log_prob()

    if parallel is True:
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                    args=args, pool=pool)

            state = sampler.run_mcmc(params0, burn_in)
            sampler.reset()
            check = sampler.run_mcmc(state, nsteps)

            return sampler.get_chain(flat=True), sampler.get_log_prob()

def many_realizations(params_initial, param_names, k_indices,
                                data, model, noise, truths,
                                nsteps=1e5, nwalkers=32, burn_in=1000, runs=10, parallel=False):

    samples = np.zeros((runs, len(params_initial)))
    log_prob = np.zeros((runs))

    for i in range(runs):
        data_noise = data + np.random.normal(scale=np.sqrt(noise[0,0]), size=len(data))
        samples, log_prob = run_mcmc(params_initial, param_names, k_indices,
                        data, model, noise, truths, nsteps=1e5, nwalkers=32, burn_in=1000, parallel=parallel)



    return samples, log_prob
