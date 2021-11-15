# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 18:44:34 2021
@author: Shannon Kroes
In this file the functions are included to generate data,
compute privacy, compute information loss and run the simulations.
throughout this file we use
n = number of samples (e.g. health records/people)
d = number of variables
levels= number of values each value can take on
ordered= whether each variable has a natural ordering (binary and categorical variables do not, whereas continuous or count variables do)
"""

# -*- coding: utf-8 -*-
import os

current_wd = os.getcwd()

import copy
import datetime
import pickle
from functools import partial
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.random.mtrand import RandomState
from scipy.stats import chi2_contingency, ks_2samp
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.LearningWrappers import learn_mspn
from spn.algorithms.Sampling import sample_instances
from spn.algorithms.Statistics import get_structure_stats

from simulation import Simulation


def save_object(obj, filename):
    with open(filename, 'wb') as output: 
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        

def paste_results(result):
    
    name=result[-1]
    information= "\nRMSE_an=" + str(result[0]) +"\n"+ "RMSE_or="+ str(result[1])  +"\n"+"bias_an="+str(result[4])+"\n"+"bias_or="+ str(result[5])+"\nempirical_SE_an="+str(np.std(result[2])) +"\nempirical_SE_or="+str( np.std(result[3]))+"\nmean_SE_an=" + str(result[6]) +"\n"+ "mean_SE_or="+ str(result[7]) +"\nmean_raw_diff="+str(result[10])+"\nabs_cor_diff="+str(result[11])+"\nuniv_prop_same="+str(result[12])+"\n"           
    # take the average privacy result over all variables
    # and over all individualss
    # and over all repetitions

    # mean_PPP_an= np.mean(result[10])
    # mean_PPP_or= np.mean(result[12])

    # mean_PPP_per_var_an=  np.mean(result[10], axis=(0,1))
    # mean_PPP_per_var_or=  np.mean(result[12], axis=(0,1))
    
    # #See if ppp is above threshold 0
    # PPP_p_an=  np.mean(result[10]>0, axis=(0,1))
    # PPP_p_or=  np.mean(result[12]>0, axis=(0,1))
    
    # p_PPP_per_var_an=  np.mean(result[10]>0, axis=(0,1))
    # p_PPP_per_var_or=  np.mean(result[12]>0, axis=(0,1))
    
    # mean_proximity_per_var_an=  np.mean(result[11], axis=(0,1))
    # mean_proximity_per_var_or=  np.mean(result[13], axis=(0,1))
    mean_privacy_an = np.mean(result[13],(0,1))
    mean_privacy_or = np.mean(result[14],(0,1))
    
    prop_privacy_an = np.mean(result[13]>0,(0,1))
    prop_privacy_or = np.mean(result[14]>0,(0,1))

    privacy= "\nprivacy_or="+str( mean_privacy_or)+ "\nprivacy_an="+str( mean_privacy_an)+'\nprop_privacy_an'+str(prop_privacy_an)+'\nprop_privacy_or'+str(prop_privacy_or)
    
    #result= name+" \ninformation loss\n"+information+"\nprivacy\n"+privacy+"\n" +"\n" +"\n" +"\n" +"\n" +"\n" +"\n" +"\n" 
    result= name+" \ninformation loss\n"+information+privacy+"\n" +"\n" +"\n" +"\n" +"\n" +"\n" +"\n" +"\n" 

    return result  


def paste_results_original(result):
    
    name=result[-1]
    information= "\n"+ "RMSE_or="+ str(result[0])  +"\n"+"bias_or="+ str(result[2])+"\nempirical_SE_or="+str( np.std(result[1]))+"\n"+ "mean_SE_or="+ str(result[3]) +"\n"           
    # take the average privacy result over all variables
    # and over all individualss
    # and over all repetitions

    mean_PPP_or= np.mean(result[5])

    mean_PPP_per_var_or=  np.mean(result[5], axis=(0,1))
    
    #See if ppp is above threshold 0
    PPP_p_or=  np.mean(result[5]>0, axis=(0,1))
    
    p_PPP_per_var_or=  np.mean(result[5]>0, axis=(0,1))
    
    mean_proximity_per_var_or=  np.mean(result[6], axis=(0,1))
    
    privacy=  "\np_PPP_per_var_or="+ str(p_PPP_per_var_or)+       "\nmean_proximity_per_var_or="+ str(mean_proximity_per_var_or)+       "\nmean_PPP_or="+str( mean_PPP_or)+      "\nPPP_p_or="+str( PPP_p_or)  
    
    result= name+" \ninformation loss\n"+information+"\nprivacy\n"+privacy+"\n" +"\n" +"\n" +"\n" +"\n" +"\n" +"\n" +"\n" 
    return result      


def cor_matrix(data):
    d=data.shape[1]
    cors=np.zeros((d,d), dtype=float)
    for i in range(d):
        cor_i=np.zeros((d), dtype=float)
        for j in range(d):
            cor_i[j]= pearsonr(np.transpose(data)[i], np.transpose(data)[j])[0]
        cors[i]=cor_i
        print(cors)
        
    return cors


def get_ks_p_values(data, AN):
    d=data.shape[1]
    ks_p_values= np.zeros((d), dtype=float)
    for i in range(d):
        ks_p_values[i]=ks_2samp(np.transpose(AN)[i], np.transpose(data)[i]).pvalue
    return ks_p_values
    

def compute_freqs(variable):
    df= pd.DataFrame(variable, columns = ['variable'])
    freq= np.array(df['variable'].value_counts())
    return freq


def get_chi_p_values(data, AN):
    d= data.shape[1]
    chi_p_values= np.zeros((d), dtype=float)
    # fist compute which 
    
    AN=np.round(AN)
    data= np.round(data)
       
    for i in range(d):
        unique_AN= np.unique(AN.T[i])
        unique_data= np.unique(data.T[i])
        levels= np.union1d(unique_AN, unique_data)
        no_levels= levels.shape[0]
        an_ind= np.zeros((no_levels))
        or_ind= np.zeros((no_levels))

        for l in range(levels.shape[0]):
            an_ind[l]= np.any(levels[l]==unique_AN)
            or_ind[l]= np.any(levels[l]==unique_data)

        AN_freq_i= compute_freqs(AN.T[i])
        
        AN_freq_full= np.zeros((no_levels))
        AN_freq_full[an_ind==1]= AN_freq_i
        data_freq_i= compute_freqs(data.T[i])
        data_freq_full= np.zeros((no_levels))
        data_freq_full[or_ind==1]= data_freq_i

        chi_p_values[i]=chi2_contingency(np.array([AN_freq_full,data_freq_full]))[1]
        
    return chi_p_values


def save_regression_coefficients(sample):
    n,d=sample.shape

    y_ind=d-1
    x_ind= np.concatenate([range(0,y_ind), range(y_ind,d-1)])

    xs=sample[:,np.array(x_ind, dtype=int)]
    y=sample[:,y_ind]
    
    reg = LinearRegression().fit(xs, y)

    coefs= reg.coef_
    intercepts=reg.intercept_
    estimates= np.hstack([intercepts, coefs])    

    return [estimates]


def Regression_Simulation(mspn, sims, an_sample_size, data, x_ind, seed, toround):#, x_ind=0):
    # in this function we will estimate a regression on different samples of 
    # spn is the sum product network from which we will sample data
    # sims are the number of times we will repeat the procedure
    # we will assume that y will be predicted by all other x values. 
    # sample size is the number of records we want to have to perform the regression analysis on
    # y_ind is the index that indicates the dependent variable
    # all other variables are treated as independent variables.
    d=data.shape[1]
    
    #x_ind= np.concatenate([range(1,y_ind), range(y_ind,p-1)])
    # we compute the regression coefficient for every variable in the data
    # and also save the corresponding standard error
    estimates= np.zeros((sims,1))
    SEs= np.zeros((sims,1))

    y_ind=d-1
    
    for i in range(sims):
        # generate anonymized data 
        AN= sample_instances(mspn, np.array([[np.repeat(np.nan,d)] * an_sample_size]).reshape(-1, data.shape[1]), RandomState(seed+204*i))
        #xs=sample[:,np.array(x_ind, dtype=int)]
        
        AN=np.array(AN)

        if toround==True :
            AN= np.round(AN,2)

        xs=AN[:,x_ind]
        # we add ones so that we have an intercept
        xs= np.hstack([np.ones(an_sample_size).reshape(an_sample_size, 1), xs])
        y=AN[:,y_ind]
        
        result = sm.OLS(y,xs).fit()
        
        estimates[i]=result.params[1]
        SEs[i]= result.bse[1]

    # estimates are the beta parameters that we estimated with the anonymized data
    # SEs are the standard errors corresponding to these beta parameters
    # AN is the last anonymized data set we generated
    return [estimates, SEs, AN]


def perform_regression_on_anonymized_data(mspn, an_sample_size, data, x_ind, seed, toround, sim):#, x_ind=0):
    # in this function we will estimate a regression on different samples of the mixed sum product network (MSPN) 
    # spn is the sum product network from which we will sample data
    # sims are the number of times we will repeat the procedure
    # we will assume that y will be predicted by all other x values. 
    # sample size is the number of records we want to have to perform the regression analysis on
    # y_ind is the index that indicates the dependent variable
    # all other variables are treated as independent variables.
    
    n,d=data.shape
    no_sample_sizes= an_sample_size.shape[0]
    
    levels= sim.independent_variables()[-1]
    levels= np.concatenate([levels,np.ones(1)])
    levels= np.array(levels, dtype=int)
    ordered=sim.ordered()
    wide_d= np.sum(levels)
    
    # x_ind= np.concatenate([range(1,y_ind), range(y_ind,p-1)])
    # we compute the regression coefficient for every variable in the data
    # and also save the corresponding standard error
    estimates= np.zeros((no_sample_sizes))
    SEs_corr= np.zeros((no_sample_sizes))

    y_ind=sim.d-1
    AN_narrow= sample_instances(mspn, np.array([[np.repeat(np.nan,d)] * an_sample_size[-1]]).reshape(-1, data.shape[1]), RandomState(seed))
    #xs=sample[:,np.array(x_ind, dtype=int)]

    AN_narrow=np.array(AN_narrow)
    AN_narrow= np.round(AN_narrow,2)
    
    AN= np.zeros((an_sample_size[-1],wide_d))
    
    for i in range(9):  
        for j in range(levels[i]):
            ind_wide=np.sum(levels[0:i])+j
            if ordered[ind_wide]==0:
                AN.T[ind_wide]= (AN_narrow.T[i]==(j+1))
            else:
                AN.T[ind_wide]= AN_narrow.T[i]
                
    for i in range(no_sample_sizes):
        # generate anonymized data (first in the narrow format, with the raw categorical variables)
        AN_i= AN[0:an_sample_size[i]]  

        xs=AN_i[:,x_ind]
        # we add ones so that we have an intercept
        xs= np.hstack([np.ones(an_sample_size[i]).reshape(an_sample_size[i], 1), xs])
        y=AN_i[:,y_ind]
        
        result = sm.OLS(y,xs).fit()
        
        estimates[i]=result.params[1]
        # we correct the standard errors for the increase in sample size
        SEs_corr[i]= result.bse[1]*np.sqrt(an_sample_size[i])/np.sqrt(n)

    # estimates are the beta parameters that we estimated with the anonymized data
    # SEs are the standard errors corresponding to these beta parameters
    # AN is the last anonymized data set we generated
    return [estimates, SEs_corr, AN_narrow]


def sample_from_mspn(mspn, d=9, seed=204, sample_size=1000):
        
    sample= sample_instances(mspn, np.array([[np.repeat(np.nan,d)] * sample_size]).reshape(-1, d), RandomState(seed))
    
    return sample


def PPP_and_proximity_mspn(data, mspn, sens= "all", p1=25, p2=75, p_reps=500 ):
    # data is the data that we want to test privacy for (original data)
    # mspn is the mspn that we want to know of how private it is
    # for this specific data set
    # sens is a range of variable indices that we consider to be sensitive
    # the default is that all variables are considered sensitive
    
    # this algorithm only considers privacy for all combinations of 
    # auxiliary information and assumes all variables can be used
    # as auxiliary information
    # we still use sampling to establish proximity 
    PPP= np.zeros((data.shape))
    probs= np.zeros((data.shape))
    proximity= np.zeros((data.shape))
    # this function tests privacy only for maximum auxiliary information.
    if sens=="all":
        sens = range(data.shape[1])   
    
    for i in range(data.shape[0]):
        for j in sens:
            to_sample_total= copy.deepcopy(data[i])
            to_sample_total= to_sample_total.reshape(-1, data.shape[1])
            total_prob=  np.exp(log_likelihood(mspn,to_sample_total))
            
            to_sample_aux= copy.deepcopy(data[i])
            to_sample_aux[j]= np.nan
            to_sample_aux= to_sample_aux.reshape(-1, data.shape[1])
            PPP[i,j]=  1-total_prob/np.exp(log_likelihood(mspn,to_sample_aux))
                       
            # compute proximity 
            sample= sample_instances(mspn, np.array([[to_sample_aux] * p_reps]).reshape(-1, data.shape[1]), RandomState(123+i))
            # here we must select only the jth variable???
            percs= np.transpose(np.percentile(sample, [p1,p2], axis=0))[j]
            proximity[i,j]=np.absolute(percs[1]-percs[0])
            # later on it might be good to standardize this proximity value

    return [PPP,proximity]


def PPP_and_proximity_sample(dat, mspn, s, p1=25, p2=95):
    # sample er s per waarde om de prob voor uit te rekenen

    PPP= np.zeros((dat.shape))
    proximity= np.zeros((dat.shape))
    for i in range(dat.shape[0]):
        for j in range(dat.shape[1]):
            to_sample= copy.deepcopy(dat[i])
            to_sample[j]= np.nan
            sample= sample_instances(mspn, np.array([[to_sample] * s]).reshape(-1, dat.shape[1]), RandomState(123+i))
            sample= np.round(sample)
            percs= np.percentile(sample, [p1,p2])
            proximity[i,j] =np.absolute(percs[1]-percs[0])
            PPP[i,j]= 1-sum(sample[:,j]==dat[i,j])/s
            
    return [PPP,proximity]


# dit is dezelfde funcite maar dan voor de originele data 
def PPP_and_proximity_original(data, p1=25, p2=75 , sens= "all"):
    # With this function we compute privacy for the original data.
    # data is the data for which is want to compute privacy
    # p1 and p2 are percentile values for which we want to assess the  distance in the distribution of the sensitive variable
    # conditional on background information 
    # by using percentiles it is more robust against outliers and skewness. 
    
    d=data.shape[1]
    PPP= np.zeros((data.shape))
    proximity= np.zeros((data.shape))
    
    if sens=="all":
        sens = range(data.shape[1])   
    
    for i in range(data.shape[0]):
        for j in sens:
            # save the indices the variables that will serve as auxiliary information
            a_ind=np.concatenate([ range(0,j), range(j+1,d)])
            # paste the auxiliary information of every individual to find the  peers
            aux=np.apply_along_axis(np.array2string,1, data[:,np.array(a_ind, dtype=int)])
            # save the sensitive values of the peers
            peers_sensitive= data[aux==aux[i],j]
 
            PPP[i,j]= 1-np.sum(peers_sensitive==data[i,j])/peers_sensitive.shape[0]
                     
            percs= np.percentile(peers_sensitive, [p1,p2])
            proximity[i,j] =np.absolute(percs[1]-percs[0])

    return [PPP,proximity]


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
    # data is the data that we want to test privacy for (original data)
    # mspn is the mspn that we want to know of how private it is
    # for this specific data set
    # sens is a range of variable indices that we consider to be sensitive
    # the default is that all variables are considered sensitive
    
    # this algorithm only considers privacy for all combinations of 
    # auxiliary information and assumes all variables can be used
    # as auxiliary information
    # we still use sampling to establish proximity 
    ordered=sim.ordered_narrow()
    n,d= data.shape
    levels= sim.independent_variables()[-1]

    levels= np.concatenate([levels,np.ones(1)])
    levels= np.array(levels, dtype=int)
    privacy= np.zeros((no_tests,d))

    # no_tests is the number of individuals for which we evaluate privacy
    # the default is for the first 100 individuals
       
    # this function tests privacy only for maximum auxiliary information.
    if sens=="all":
        sens = range(data.shape[1])   
    
    for i in range(no_tests):
        for j in sens:
            
            if ordered[j]==0:
                # for unordered variables, compute the probability of each possible value
                # given maximum auxiliary information
                # save these probabilities in vector probs_sens
                
                domain=np.unique(data.T[j])
                domain_false= domain[domain!=data[i,j]]
                probs_sens=np.zeros((domain_false.shape[0]), dtype=float)
                for v in range(domain_false.shape[0]):
                    to_sample_aux= copy.deepcopy(data[i])
                    to_sample_aux[j]= domain_false[v]
                    to_sample_aux= to_sample_aux.reshape(-1, data.shape[1])
                    probs_sens[v]=np.exp(log_likelihood(mspn,to_sample_aux))
                    
                privacy[i,j]=np.sum(probs_sens>0)/domain_false.shape[0]
                     
            else:                   
                to_sample_aux= copy.deepcopy(data[i])
                to_sample_aux[j]= np.nan
                to_sample_aux= to_sample_aux.reshape(-1, data.shape[1])
                # compute proximity 
                peers_sensitive= sample_instances(mspn, np.array([[to_sample_aux] * p_reps]).reshape(-1, data.shape[1]), RandomState(123+i))
                # here we must select only the jth variable???
                privacy[i,j]=np.sqrt(np.mean((peers_sensitive-data[i,j])**2))
                # later on it might be good to standardize this proximity value

    return privacy


def simulation_spn_privacy( n=100, seed=1919, repetitions=1, an_sample_size=None, p_reps=500, distr="normal", sparse=False, mis=100, priv_or_runs=0, priv_an_runs=0, rows=None, H0=False,  or_res=True, pois_real=False, threshold=0.3, leaves=None, toround=True, save=True, cpus=-1, cor_univ=True, col_test="rdc",  hist_source="numpy", cols="rdc", save_inter=True, pois_allreal=False, non_parametric=True, no_tests=100, no_clusters=2, standardize=False, ecdf=False):    

    # we can make a distinction for the input between:
    # 1. generic simulation settings
    # sims: the number of anonymized data sets we generate for 1 original data set, which is always set to 1 in practice
    # repetitions: the number of original data sets we generate (and for each of these we generate anonymized data and compute information loss for both)
    # to_round: whether we round (important for privacy computation), which is in practice always true
    # or_res: whether we want results for the original data (because the time it takes to run this relative to the anonymized data is very small we always set this to true)

    # 2. data generation
    
    # distr: indicates the data generating model: "normal", "poisson", "categorical" or "mixed
    # n: the sample size of the original data set(s)
    # H0: whether the variable of interest has an effect on the outcome variable (i.e. null or alternative scenario)

    # 3. settings for privacy
    # p_reps: the number of samples we take from the mspn to estimate proximity
    # priv_or_runs and priv_an_runs: the amount of runs for which we evaluate privacy for original and anonymized data, respectively
    # we do this because it takes a lot of time to evaluate privacy
    # no_tests: amount of people we test privacy for 
    
    # 4. setings to generate the anonymized data 
    # an_sample_size is either one or multiple sample sizes for the anonymized data we want to investigate.
    # in the final experiment we use: an_sample_size= np.array([1000, 5000, 15000, 30000, 50000, 75000, 100000])
    # threshold: the RDC threshold for the MSPN algorithm, between 0 and 1, we set it to .01 always, but the default is 0.3
    # rows: splitting method for horizontal splits, e.g. "kmeans", "rdc", see the function get_splitting_functions in spn/algorithms/LearningWrappers.py for all available options
    # mis: minimum number of instances/people/rows that need to be in a cluster before another horizontal split is made. 
    # pois_real and pois_allreal: these govern the context chosen for the poisson variables. Though they are discrete variables
    # these are two options to consider part of them or all of them to be continuous when generating the mspn (see Simulation.ds_context)
    # hist_source: the 
    # col_test: 
    # cols: the splitting method chosen to split variables (vertical splitting), see spn\algorithms\splitting\RDC.py for the options
    
    sim= Simulation(n=n, distr=distr, pois_real=pois_real, pois_allreal= pois_allreal, H0=H0, non_parametric=non_parametric, sparse=sparse)
    true_param= sim.true_param


    d=9
    is_discrete= sim.is_discrete()
        
    if rows==None:
        if distr=="normal":
            rows="kmeans"
        elif distr=="poisson" or distr=="categorical":
            rows="rdc+"
        elif distr=="mixed":
            rows= "rdc+kmeans"

    no_an_sample_sizes=an_sample_size.shape[0]
    # if we are going to assess the effect ofincreaseing the anonymized sample
    # we do not reevaluate the effects on the correlation and univariate distibutions
    # because we know this will only improve?
    if no_an_sample_sizes>1:
        cor_univ=False

    estimates_an=np.zeros((repetitions,no_an_sample_sizes))
    SEs_an=np.zeros((repetitions,no_an_sample_sizes))

    # save the true parameters
    # voor nu doen we dit even via approximation
    estimates_or=np.zeros((repetitions))
    SEs_or=np.zeros((repetitions))

    # create matrices to store privacy results
    privacy=np.zeros((np.min([repetitions,priv_an_runs]),no_tests,d))
    privacy_original=np.zeros((np.min([repetitions,priv_or_runs]),no_tests,d))

    diffs=np.zeros((repetitions), dtype=float)
    cor_diffs=np.zeros((repetitions,d,d), dtype=float)
    univs=np.zeros((repetitions,d), dtype=float)  
    raw_diffs=np.zeros((repetitions), dtype=float)
    os.chdir(current_wd+"/Results")
    
    levels= sim.independent_variables()[-1]
    levels= np.concatenate([levels,np.ones(1)])
    levels= np.array(levels, dtype=int)
    ordered=sim.ordered()
    wide_d= np.sum(levels)

    x_ind= range(0,wide_d-1)
    np.random.seed(seed)
    
    if distr!="mixed":
        Distr=distr
    elif sparse:
        Distr="mixed_sparse"
    else: 
        Distr="mixed"
        
    for s in range(repetitions):
        
        np.random.seed(seed+s)
        sim= Simulation(n=n, distr=distr, pois_real=pois_real, H0=H0, non_parametric=non_parametric)
             
        data_wide= sim.generate_data()

        # bereken eerst regr in sample om te kijken of het wel goed gaat met afronden naar 2 decimalen
        if or_res:

            xs=data_wide[:,x_ind]
            # we add ones so that we have an intercept
            xs= np.hstack([np.ones(n).reshape(n,1), xs])
            y=data_wide[:,wide_d-1]
        # we only take the index of the independent variable
        # which we assume to be the first variable
        # we save the beta parameter estimate and the SE of the first variable
            reg_result = sm.OLS(y,xs).fit()
            estimates_or[s]=np.array([reg_result.params[1]])
            SEs_or[s]= np.array([reg_result.bse[1]])
            
        # change the data from wide to narrow format   
        data= np.zeros((n,9))
        data_transform= np.zeros((data_wide.shape))
        for i in range(9):  
            for j in range(levels[i]):
                ind_wide=np.sum(levels[0:i])+j
                data_transform.T[ind_wide]=data_wide.T[ind_wide]*(j+1)           
                data.T[i]= np.sum(data_transform.T[np.sum(levels[0:i]):(np.sum(levels[0:(i+1)]))],0)
            
        ds_context=sim.ds_context()
        ds_context.add_domains(data)

        mspn = learn_mspn(data, ds_context, min_instances_slice=mis, rows=rows, threshold=threshold, leaves=leaves, cpus=cpus, hist_source=hist_source, col_test=col_test, cols=cols, no_clusters=no_clusters, standardize=standardize, ecdf=ecdf)    

        # Use this function to 1. create anonymized data with the MSPN and 2. Perform the regression and extract the parameters. 
        estimates_an[s], SEs_an[s], AN=perform_regression_on_anonymized_data(mspn, an_sample_size, data, x_ind, seed=seed+s, toround=toround, sim=sim)

        # in addition to estimating information loss we want to measure privacy increase
        # for the original data
        if s<priv_or_runs:
            privacy_original[s]= PoAC_and_proximity_original(data, sim=sim,  no_tests=no_tests)

        # and for the spn
        if s<priv_an_runs:
            privacy[s]= PoAC_and_proximity_mspn(data, mspn,sim,  p_reps=p_reps, no_tests=no_tests)

        if cor_univ:
                        
            an_cor=cor_matrix(AN)
            or_cor= cor_matrix(data)
        
            cor_diffs[s]= an_cor-or_cor
            diffs[s]= np.mean(np.absolute(cor_diffs[s]))    
            raw_diffs[s]= np.mean(cor_diffs[s])

            univs[s][is_discrete]= get_chi_p_values(data.T[is_discrete].T, AN.T[is_discrete].T)
            univs[s][is_discrete==0]= get_ks_p_values(data.T[is_discrete==0].T, AN.T[is_discrete==0].T)
 
    # to save object while running the simulation, so that partial results can be retrieved
        if save_inter:
            RMSE_an= np.sqrt(np.mean((estimates_an-true_param)**2))
            RMSE_or=  np.sqrt(np.mean((estimates_or-true_param)**2)) 
        
            bias_an= np.mean(estimates_an-true_param)
            bias_or= np.mean(estimates_or-true_param)
        
            SE_mean_an= np.mean(SEs_an)*np.sqrt(n)/np.sqrt(an_sample_size)
            SE_mean_or= np.mean(SEs_or)*np.sqrt(n)/np.sqrt(an_sample_size)
                    
            univ_same_prop= np.mean(univs>.05)
            raw_cor_diff=np.mean(diffs)
            abs_cor_diff=np.mean(raw_diffs)
                
            # save some of the characteristics of the mspn
            print(get_structure_stats(mspn))
            print(os.getcwd())
            H='H1'
            if H0==True: H='H0' 
            pois="pois_discrete"
            if pois_real:
                pois= "pois_real"
            if pois_allreal:
                pois="pois_allreal"
            
           # save output as object
            name= str(H)+ str(Distr)+'n'+str(n)+'mis'+str(mis)+str(rows)+'col_test'+col_test+'t'+ str(threshold)+pois+'col_test'+col_test+'hist_source'+hist_source+"an_sample_size"+str(an_sample_size)+"no_clusters="+str(no_clusters)+"_"+"standardize"+str(standardize)+"ecdf"+str(ecdf)+str(repetitions)+"repetitions"+"save_inter"
        
            result=[RMSE_an,  RMSE_or,  estimates_an,   estimates_or, bias_an,   bias_or, SE_mean_an, SE_mean_or,SEs_an ,SEs_or, raw_cor_diff, abs_cor_diff, univ_same_prop, privacy,  privacy_original, s, mspn, name]
            save_object(result, name)
      
            # also log how far along we are
            file1 = open("log_progress.txt","a") 
            now = datetime.datetime.now()
            file1.write(str(s)+"___"+str(now)+"___"+name+"\n")
            if s==repetitions-1:
                file1.write("\n \n \n \n ")
            file1.close()

        print(s, "n =", n, "mis=", mis, "distr=", Distr, "sparse=", sparse, "rows=", rows, "threshold=", threshold, pois , "H0=", H0, "col_test=",col_test, "hist_source=", hist_source, "no_clusters=", no_clusters)        
        
    RMSE_or=  np.sqrt(np.mean((estimates_or-true_param)**2)) 
    bias_or= np.mean(estimates_or-true_param)
    SE_mean_or= np.mean(SEs_or)

    RMSE_an= np.sqrt(np.mean((estimates_an-true_param)**2,0))
    bias_an= np.mean(estimates_an-true_param,0)
    SE_mean_an= np.mean(SEs_an, 0)

    univ_same_prop= np.mean(univs>.05)
    raw_cor_diff=np.mean(diffs)
    abs_cor_diff=np.mean(raw_diffs)
    
    # save some of the characteristics of the mspn
    print(get_structure_stats(mspn))
    print(os.getcwd())
    H='H1'
    if H0==True: H='H0' 
    # save output as object
    name= str(H)+ str(Distr)+'n'+str(n)+'mis'+str(mis)+str(rows)+'t'+ str(threshold)+pois+'col_test'+col_test+'hist_source'+hist_source+"an_sample_size"+str(an_sample_size)+"_"+"standardize"+str(standardize)+"ecdf"+str(ecdf)+str(no_clusters)+"_clusters="+str(repetitions)+"repetitions"

    if save:           
        result=[RMSE_an,  RMSE_or,  estimates_an,   estimates_or, bias_an,   bias_or, SE_mean_an, SE_mean_or,SEs_an ,SEs_or, raw_cor_diff, abs_cor_diff, univ_same_prop,privacy,  privacy_original, name]
        save_object(result, name)

        # save summary of results in output text file
        file1 = open("output.txt","a") 
        file1.write(paste_results(result))

    result=[RMSE_an,  RMSE_or,  estimates_an, estimates_or, bias_an, bias_or, SEs_an ,SEs_or, SE_mean_an, SE_mean_or, raw_cor_diff, abs_cor_diff, univ_same_prop, privacy,  privacy_original, cor_diffs, raw_diffs, univs, data, mspn, name]

    return result


def View_results(result, priv=False):
    print(["RMSE_an=" ,result[0]])
    print(["RMSE_or=", result[1]])

    print(["bias_an=", result[4]])
    print(["bias_or=", result[5]])
    
    # take the average privacy result over all variables
    # and over all individuals
    # and over all repetitions 
    
    if priv==True:
                  
        print()
        mean_PPP_an= np.mean(result[6])
        mean_PPP_or= np.mean(result[8])
    
        mean_PPP_per_var_an=  np.mean(result[6], axis=(0,1))
        mean_PPP_per_var_or=  np.mean(result[8], axis=(0,1))
        
        #See if ppp is above threshold 0
        PPP_p_an=  np.mean(result[6]>0, axis=(0,1))
        PPP_p_or=  np.mean(result[8]>0, axis=(0,1))
        
        p_PPP_per_var_an=  np.mean(result[6]>0, axis=(0,1))
        p_PPP_per_var_or=  np.mean(result[8]>0, axis=(0,1))
        
        mean_proximity_per_var_an=  np.mean(result[7], axis=(0,1))
        mean_proximity_per_var_or=  np.mean(result[9], axis=(0,1))
    
        print(["p_PPP_per_var_an=", p_PPP_per_var_an])
        print(["p_PPP_per_var_or=", p_PPP_per_var_or])
        
        print(["mean_proximity_per_var_an=", mean_proximity_per_var_an])
        print(["mean_proximity_per_var_or=", mean_proximity_per_var_or])
        
        print(["mean_PPP_an=", mean_PPP_an])
        print(["mean_PPP_or=", mean_PPP_or])
    
        print(["PPP_p_an=", PPP_p_an])
        print(["PPP_p_or=", PPP_p_or])


def load_object(filename):
    import _pickle as cPickle
    with open(filename, "rb") as input_file:
        object= cPickle.load(input_file)

        
def check_convergence(vect):
    means=np.zeros((vect.shape[0]), dtype=float)
    for i in range(vect.shape[0]):
        means[i]= np.mean(vect[0:i+1])
        
    plt.plot(range(vect.shape[0]), means)
        

def simulation_original_data( n=100, seed=1919, sims=1, repetitions=1,  distr="normal", priv_or_runs=0,  H0=False,pois_real=False, save=False):    
    # mean is average of the true distribution
    # covariance is covariantie die elke variabele heeft met alle andere
    # n is the sample size of the original data set
    # sims is the number of anonymized data sets we generate for 1 original data set
    # repetitions is the number of original data sets we generate
    # to round is whether we round (important for privacy computation)
    # p_reps the number of samples we take from the mspn to estimate privacy
    # distr indicates the data generating model
    # priv_or_runs and priv_an_runs are is whether we only evaluate privacy of 1 original data set
    # we do this because it takes a lot of time to evaluate privacy
    # but the result is practically equal (for 8/9 variables in the normal scenario is has variance 0)
    # if we want to compare different settings for anonymization it is unneccessary to compute the results for the original data every time
    # threshold is the RDC threshold for the MSPN algorithm
    
    sim= Simulation(n=n, distr=distr, pois_real=pois_real, H0=H0)
    true_param= sim.true_param

    estimates_or=np.zeros((repetitions))
    SEs_or=np.zeros((repetitions))

    x_ind= range(0,sim.d-1)
    np.random.seed(seed)
    
    # create matrices to store privacy results
    PPP_original=np.zeros((priv_or_runs,n,sim.d))
    proximity_original=np.zeros((priv_or_runs,n,sim.d))

    for s in range(repetitions):
        
        np.random.seed(seed+s)
        sim= Simulation(n=n, distr=distr, pois_real=pois_real, H0=H0)
        data= sim.generate_data()

        # bereken eerst regr in sample om te kijken of het wel goed gaat met afronden naar 2 decimalen

        xs=data[:,x_ind]
        # we add ones so that we have an intercept
        xs= np.hstack([np.ones(n).reshape(n,1), xs])
        y=data[:,sim.d-1]
    # we only take the index of the independent variable
    # which we assume to be the first variable
    # we save the beta parameter estimate and the SE of the first variable
        reg_result = sm.OLS(y,xs).fit()
        estimates_or[s]=np.array([reg_result.params[1]])
        SEs_or[s]= np.array([reg_result.bse[1]])

        if s<priv_or_runs:
            privacy_original= PPP_and_proximity_original(data)
            PPP_original[s]=privacy_original[0]
            proximity_original[s]= privacy_original[1] 

    RMSE_or=  np.sqrt(np.mean((estimates_or-true_param)**2)) 

    bias_or= np.mean(estimates_or-true_param)
    
    SE_mean_or= np.mean(SEs_or)
    
    # save some of the characteristics of the mspn

    H='H1'
    if H0==True: H='H0' 
    # save output as object
    name= 'original'+str(H)+ str(distr)+'n'+str(n)+str(repetitions)+"repetitions "

    if save:           
        result=[ RMSE_or,    estimates_or,   bias_or, SE_mean_or,SEs_or,   PPP_original, proximity_original, name]
        os.chdir(current_wd+"/Results")
        save_object(result, name)

        # save summary of results in output text file
        file1 = open("output_original.txt","a") 
        file1.write(paste_results_original(result))
        file1.close() 
    else:
        result=[  RMSE_or,    estimates_or,   bias_or, SEs_or, SE_mean_or,  PPP_original, proximity_original, data, name]     

    return result


def compute_CI(result):
    no_reps= result[2].shape[0]
    lower_bound= result[4]-2*np.std(result[2])/np.sqrt(no_reps)
    upper_bound= result[4]+2*np.std(result[2])/np.sqrt(no_reps)
    
    return [lower_bound, upper_bound]


def print_an_information(result, priv=False):
    print(["RMSE_an=" ,result[0]])
    print(["bias_an=", result[4]]) 
    print(["empirical_SE_an=", np.std(result[2])])
    print(["mean_SE_an=" , result[6]])
    sd= np.std(result[2])
    mean= np.mean(result[2])
    pop=np.random.normal(0, sd, 1000000)
    p= ks_2samp(result[2], pop).pvalue 
    print(["normal=",p>.05 ])
    print(["CI=",compute_CI(result) ])

    # take the average privacy result over all variables
    # and over all individuals
    # and over all repetitions 
    
    if priv==True:
                  
        print()
        mean_PPP_an= np.mean(result[6])
        mean_PPP_or= np.mean(result[8])
    
        mean_PPP_per_var_an=  np.mean(result[6], axis=(0,1))
        mean_PPP_per_var_or=  np.mean(result[8], axis=(0,1))
        
        #See if ppp is above threshold 0
        PPP_p_an=  np.mean(result[6]>0, axis=(0,1))
        PPP_p_or=  np.mean(result[8]>0, axis=(0,1))
        
        p_PPP_per_var_an=  np.mean(result[6]>0, axis=(0,1))
        p_PPP_per_var_or=  np.mean(result[8]>0, axis=(0,1))
        
        mean_proximity_per_var_an=  np.mean(result[7], axis=(0,1))
        mean_proximity_per_var_or=  np.mean(result[9], axis=(0,1))
    
        print(["p_PPP_per_var_an=", p_PPP_per_var_an])
        print(["p_PPP_per_var_or=", p_PPP_per_var_or])
        
        print(["mean_proximity_per_var_an=", mean_proximity_per_var_an])
        print(["mean_proximity_per_var_or=", mean_proximity_per_var_or])
        
        print(["mean_PPP_an=", mean_PPP_an])
        print(["mean_PPP_or=", mean_PPP_or])
    
        print(["PPP_p_an=", PPP_p_an])
        print(["PPP_p_or=", PPP_p_or])


def compute_CI_ans(bias, estimates, no_reps):
    
    lower_bound= bias-2*np.std(estimates)/np.sqrt(no_reps)
    upper_bound=bias+2*np.std(estimates)/np.sqrt(no_reps)
    
    return [lower_bound, upper_bound]


def print_an_information_ans(result, priv=False):
    ps= np.zeros((int(result[2].shape[1])),dtype=float)
    CIs= np.zeros((int(result[2].shape[1])),dtype=list)
    print(["RMSE_an=" ,result[0]])
    print(["bias_an=", result[4]]) 
    print(["empirical_SE_an=", np.std(result[2],0)])
    print(["mean_SE_an=" , result[6]])

    for i in range(result[2].shape[1]):
        sd= np.std(result[2].T[i])
        mean= np.mean(result[2].T[i])
        pop=np.random.normal(0, sd, 1000000)
        ps[i]= ks_2samp(result[2].T[i], pop).pvalue 
        CIs[i]= compute_CI_ans(result[4][i], result[2].T[i], result[2].T[0].shape[0])
    print(["normal=",ps>.05 ])
       
    print(["CIs=", CIs])

        
def print_inter_information_ans(result, true_param=.3):
    no_reps=result[2].T[-1].shape[0]- np.sum(result[2].T[-1]==0)
    CIs= np.zeros((result[2].shape[1]), dtype=list)
    for i in range(result[2].shape[1]):
        result_i=result[2].T[i][0:no_reps]
                   
        lower_bound= np.mean(result_i)-2*np.std(result_i)/np.sqrt(no_reps) # 0.29108434164613006
        upper_bound= np.mean(result_i)+2*np.std(result_i)/np.sqrt(no_reps) # 0.30379596690869626
        CIs[i]= [lower_bound, upper_bound]
    #  empirical SE
    print(" empirical SE:")
    print(np.std(result[2][0:no_reps], 0))
    
    # bias
    print(" mean parameter:")
    print(np.mean(result[2][0:no_reps], 0))
    
    print(" bias")
    print(np.mean(result[2][0:no_reps], 0)-true_param)
    
    print(" mean estimated SE:")
    print(np.mean(result[8][0:no_reps], 0))
    print("CIs:")
    print(CIs)


def compute_CI_or(result):
    no_reps= result[3].shape[0]
    lower_bound= np.mean(result[3])-2*np.std(result[3])/np.sqrt(no_reps)
    upper_bound= np.mean(result[3])+2*np.std(result[3])/np.sqrt(no_reps)
    
    return [lower_bound, upper_bound]
