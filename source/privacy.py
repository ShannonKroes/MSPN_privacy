# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 16:46:40 2022

@author: Shannon
"""

import numpy as np

def PoAC_and_proximity_original(data, sim, no_tests=None):
    """With this function we compute privacy for the original data.
    This function also assess privacy for maximum auxiliary information onl,
    i.e. all variables can be used as background information.
    Optimized by Sander van Rijn <s.j.van.rijn@liacs.leidenuniv.nl> ORCID: 0000-0001-6159-041X
    :param data:      Data for which is want to compute privacy. Order is assumed to be random.
    :param sim:       Simulation specification
    :param no_tests:  Number of tests to perform
    """
    n, d = data.shape
    if no_tests is None:
        no_tests = n

    levels = sim.independent_variables()[-1]
    levels = np.concatenate([levels, [1]]).astype(int)
    ordered = sim.ordered_narrow()

    a_indices = column_exclusion_indices(d)
    func = partial(_single_PoAC_test, data=data, levels=levels, ordered=ordered, a_indices=a_indices)

    with Pool(cpu_count()) as p:
        privacy = p.map(func, range(no_tests))

    return np.array(privacy)


def _single_PoAC_test(i, data, levels, ordered, a_indices):
    """Parallelizable helper function that performs a single test of the
    PoAC_and_proximity_original function
    """
    privacies = []
    aux = data == data[i]
    for j, a_ind in enumerate(a_indices):
        indices = np.all(aux[:, a_ind], axis=1)
        peers_sensitive = data[indices, j]

        if ordered[j] == 1:
            privacy = np.sqrt(np.mean((peers_sensitive - data[i, j]) ** 2))
        else:
            privacy = (np.unique(peers_sensitive).shape[0] - 1) / levels[j]
        privacies.append(privacy)

    return privacies


def column_exclusion_indices(n):
    """Create a series of index vectors to exclude each column once
    :param n: number of columns
    Example:
    >>> column_exclusion_indices(3)
    ... array([[1, 2],   # omits 0
    ...        [0, 2],   # omits 1
    ...        [0, 1]])  # omits 2
    """
    return np.array([
        [x for x in range(n) if x != j]
        for j in range(n)
    ])


def PoAC_and_proximity_mspn(data,mspn, sim, sens= "all", p_reps=500, no_tests=100 ):
    '''
    This algorithm only considers privacy for all combinations of 
    auxiliary information and assumes all variables can be used
    as auxiliary information
    we use sampling to establish proximity 
        
    Parameters
    ----------
    data : np.array
        The data that we want to test privacy for (original data).
    mspn : 
        The mspn for which it will be established how private it is for this specific data set.
    sim : Simulation object
        Simulation object from simulation.py. This contains the information on whether the PoAC or the proximity
        will be computed, dependent on the variable type. 
    sens : np.array, optional
         A range of variable indices that we consider to be sensitive. The default is "all".
    p_reps : int, optional
        The number of samples we take to estimate probabilities for proximity. The default is 500.
    no_tests : int, optional
        The number of individuals for which we evaluate privacy. The default is 100.

    Returns
    -------
    privacy : np.array
        A no_tests by len(sens) array that indicates either PoAC or proximity values for every sensitive variable for every
        of the no_tests individuals.

    '''
    ordered=sim.ordered_narrow()
    n,d= data.shape
    levels= sim.independent_variables()[-1]
    levels= np.concatenate([levels,np.ones(1)])
    levels= np.array(levels, dtype=int)
    privacy= np.zeros((no_tests,d))
    n,d=data.shape
    test_data= data[0:no_tests]

    if sens=="all":
        sens = range(data.shape[1])   

    for j in sens:
        if ordered[j]==0:
            domain=np.unique(data.T[j])
            vs= np.repeat(domain, no_tests).reshape(domain.shape[0], no_tests).T
            to_sample= np.repeat(test_data,(domain.shape[0]),0)
            to_sample.T[j]= vs.reshape(-1)
            probs_sens=np.exp(log_likelihood(mspn,to_sample))
            probs_all_vs= np.sum(probs_sens.reshape(-1, domain.shape[0]), axis=1)
            probs_all_vs_rep= probs_sens.reshape(-1)/np.repeat(probs_all_vs,  domain.shape[0])
            # we count which proportion of values are considered
            # it is assumed here that the true value must be probable. 
            privacy[:,j]= np.sum((probs_all_vs_rep>.01).reshape(-1, domain.shape[0]), axis=1)/domain.shape[0]
                         
        else:                   
            to_sample_aux= copy.deepcopy(test_data)
            to_sample_aux.T[j]= np.nan
            to_sample= np.repeat(to_sample_aux, p_reps, axis=0)
            # compute proximity 
            peers_sensitive= sample_instances(mspn, np.array([to_sample]).reshape(-1, data.shape[1]), RandomState(123+j))
            diffs_from_sens= peers_sensitive.T[j]-np.repeat(test_data.T[j],p_reps)
            sqrt_diffs= (diffs_from_sens)**2
            
            privacy[:,j]=np.sqrt(np.mean(sqrt_diffs.reshape(-1, p_reps), axis=1))

    return privacy
