# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 11:49:15 2022

@author: Shannon
"""
import os
import matplotlib.pyplot as plt
import spn as spn
import math
import numpy as np
import copy
from numpy.random.mtrand import RandomState
from spn.algorithms.Sampling import sample_instances
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from spn.algorithms.LearningWrappers import learn_mspn
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from spn.algorithms.Statistics import get_structure_stats
import statsmodels.api as sm
from scipy.stats import ks_2samp
from scipy.stats.stats import pearsonr
import datetime  
import pandas as pd
from scipy.stats import chi2_contingency, norm, binom, poisson, expon
import pickle
from spn.algorithms.Inference import log_likelihood
from base import *
from privacy import *
from simulation import Simulation

def simulation_spn_privacy( n=100, seed=1919, repetitions=1, an_sample_size=None, p_reps=500, distr="normal", sparse=False, mis=100, priv_or_runs=0, priv_an_runs=0, rows=None, H0=False,  or_res=True, pois_real=False, threshold=0.3, toround=True, save=True, cor_univ=True, save_inter=True, no_tests=100, no_clusters=2, standardize=False):    
    '''
    we can make a distinction for the input between:
    1. generic simulation settings
    sims: the number of anonymized data sets we generate for 1 original data set, which is always set to 1 in practice
    repetitions: the number of original data sets we generate (and for each of these we generate anonymized data and compute information loss for both)
    to_round : bool, whether we round (important for privacy computation), which is in practice always true
    or_res : bool, whether we want results for the original data (because the time it takes to run this relative to the anonymized data is very small we always set this to true)

    2. data generation
    
    distr : str, indicates the data generating model: "normal", "poisson", "categorical" or "mixed
    n : int, the sample size of the original data set(s)
    H0 : bool, whether the variable of interest has an effect on the outcome variable (i.e. null or alternative scenario)


    3. settings for privacy
    p_reps : int, the number of samples we take from the mspn to estimate proximity
    priv_or_runs and priv_an_runs : int, the amount of runs for which we evaluate privacy for original and anonymized data, respectively
    we do this because it takes a lot of time to evaluate privacy
    no_tests : int, amount of people we test privacy for 
    
    4. setings to generate the anonymized data 
    an_sample_size : np.array, either one or multiple sample sizes for the anonymized data we want to investigate.
    in the final experiment we use: an_sample_size= np.array([100000])
    threshold : float,  the RDC threshold for the MSPN algorithm, between 0 and 1, we set it to .01 always, but the default is 0.3
    rows : str, splitting method for horizontal splits, e.g. "kmeans", "rdc", see the function get_splitting_functions in spn/algorithms/LearningWrappers.py for all available options
    mis : int, minimum number of instances/people/rows that need to be in a cluster before another horizontal split is made. 
    pois_real : bool, these govern the context chosen for the poisson variables. Though they are discrete variables
    these are two options to consider part of them or all of them to be continuous when generating the mspn (see Simulation.ds_context)   
    '''
    # Generate the simulation object which contains everything we need for the original (privacy-sensitive) data
    sim=Simulation(n=n, distr=distr,pois_real=pois_real, pois_allreal= pois_allreal, H0=H0, non_parametric=non_parametric, sparse=sparse)
    true_param= sim.true_param
    d=9
    is_discrete= sim.is_discrete()
    no_an_sample_sizes=an_sample_size.shape[0]
    
    # if we are going to assess the effect of increaseing the anonymized sample
    # we do not reevaluate the effects on the correlation and univariate distibutions  
    if no_an_sample_sizes>1:
        cor_univ=False
                
    estimates_an=np.zeros((repetitions,no_an_sample_sizes))
    SEs_an=np.zeros((repetitions,no_an_sample_sizes))
    # save the true parameters
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
        
    for s in range(repetitions):
        
        np.random.seed(seed+s)
        sim=Simulation(n=n, distr=distr,pois_real=pois_real, H0=H0, non_parametric=non_parametric)
        data_wide= sim.generate_data()
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
        mspn = learn_mspn(data, ds_context, min_instances_slice=mis, rows=rows, threshold=threshold, no_clusters=no_clusters, standardize=standardize)    

        # Use this function to 1. create anonymized data with the MSPN and 2. Perform the regression and extract the parameters. 
        estimates_an[s], SEs_an[s], AN=perform_regression_on_anonymized_data(mspn, an_sample_size, data, x_ind, seed=seed+s, toround=toround, sim=sim)
        os.chdir(current_wd)
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
            result=[RMSE_an,  RMSE_or,  estimates_an,   estimates_or, bias_an,   bias_or, SE_mean_an, SE_mean_or,SEs_an ,SEs_or, raw_cor_diff, abs_cor_diff, univ_same_prop, privacy,  privacy_original, s, mspn, data, AN, name]
            save_object(result, name)
            # also log how far along we are
            file1 = open("log_progress.txt","a") 
            now = datetime.datetime.now()
            file1.write(str(s)+"___"+str(now)+"___"+name+"\n")
            if s==repetitions-1:
                file1.write("\n \n \n \n ")
            file1.close() 

        print(s, "n =", n, "mis=", mis, "distr=", Distr, "sparse=", sparse, "rows=", rows, "threshold=", threshold, pois , "H0=", H0, "col_test=",col_test, "hist_source=", hist_source, "no_clusters=", no_clusters)        
        
    # compute information loss
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
    H='H1'
    if H0==True: H='H0' 
    # save output as object
    name= str(H)+ str(Distr)+'n'+str(n)+'mis'+str(mis)+str(rows)+'t'+ str(threshold)+pois+'col_test'+col_test+'hist_source'+hist_source+"an_sample_size"+str(an_sample_size)+"_"+"standardize"+str(standardize)+"ecdf"+str(ecdf)+str(no_clusters)+"_clusters="+"no_tests="+str(no_tests)+"_"+str(priv_an_runs)+"priv_an_runs"+"__"+str(repetitions)+"repetitions"

    if save:           
        result=[RMSE_an,  RMSE_or,  estimates_an,   estimates_or, bias_an,   bias_or, SE_mean_an, SE_mean_or,SEs_an ,SEs_or, raw_cor_diff, abs_cor_diff, univ_same_prop,privacy,  privacy_original, AN, mspn, data, name]
        save_object(result, name)
        # save summary of results in output text file
        file1 = open("output.txt","a") 
        file1.write(paste_results(result))
    result=[RMSE_an,  RMSE_or,  estimates_an, estimates_or, bias_an, bias_or, SEs_an ,SEs_or, SE_mean_an, SE_mean_or, raw_cor_diff, abs_cor_diff, univ_same_prop, privacy,  privacy_original, cor_diffs, raw_diffs, univs, data, mspn, name, AN]     
        
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
    
    sim=Simulation(n=n, distr=distr,pois_real=pois_real, H0=H0)
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
        sim=Simulation(n=n, distr=distr,pois_real=pois_real, H0=H0)
        data= sim.generate_data()
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
