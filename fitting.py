import numpy as np
import matplotlib.pyplot as plt
import emcee
import copy

from multiprocessing import Pool

def log_prior(params, truths, k_indices):
    # P_m = params[-len(k_indices):]
    P_m = params[-len(k_indices):]
      #  print('The sampler is trying to assign a negative value to the parameter P_m')
    if np.all(np.asarray(P_m) > 0):
        return 0.0
    return -np.inf

def log_likelihood(params, param_names, k_indices, data, model, noise, truths):
    param_guesses = copy.deepcopy(truths)
    for i, names in enumerate(param_names):
        if names is not 'P_m':
            param_guesses[names] = params[i]

        if names is 'P_m':
            P_m_params = params[i:]
            for j, l in enumerate(k_indices):
                param_guesses['P_m'][l] = P_m_params[j]

    pspec = model.pspec(k_indices, params=param_guesses)

    diff = data - pspec
    return -0.5 * np.dot(diff, np.linalg.solve(noise, diff))

def log_probability(params, param_names, k_indices, data, model, noise, truths):

    lp = log_prior(params, truths, k_indices)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, param_names, k_indices, data, model, noise, truths)

def run_mcmc(params_initial, param_names, k_indices, data, model, noise, truths,
                                         nsteps=1e5, nwalkers=32, burn_in=1000, parallel=False):

    args = [param_names, k_indices, data, model, noise, truths]

    ndim = len(params_initial)
    p0 = np.zeros((nwalkers, ndim))

    for i, val in enumerate(params_initial):
        std_dev = .1 * val
        p0[:,i] = np.random.normal(scale=std_dev, size=nwalkers)

    # print(p0)
    params0 = params_initial + p0

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
