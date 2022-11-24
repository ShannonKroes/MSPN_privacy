# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 10:06:51 2021

@author: shann
"""
# in this file we compute the required error variance (evar) to obtain an R2 of 0.3


import os
os.chdir(r"C:\Users\shann\Documents\Sanquin\Project 4\SPN\Simulation\DSLab")
from all_functions import *



n=10000000

def compute_and_check_evar(distr,n=n):
    
    sim=Simulation(distr=distr, n=n)
    X,sds,evar,levels= sim.independent_variables()       
    X_wide= np.zeros((sim.n, sim.d-1))
    ordered=sim.ordered()
    
    for i in range(8):  
        for j in range(levels[i]):
            ind_wide=np.sum(levels[0:i])+j
                 
            if ordered[ind_wide]==0:
                X_wide.T[ind_wide]= (X.T[i]==(j+1))
            else:
                X_wide.T[ind_wide]= X.T[i]      
    
    beta_raw=0.3
    levels_long= np.concatenate(np.asarray(list(map(lambda x: np.repeat(x,x), levels))))
    betas_raw=np.repeat(beta_raw, sum(levels))
    betas_raw_levels= betas_raw/levels_long
    betas= betas_raw_levels/sds
    
    Y_hat=np.sum(betas*X_wide, axis=1)   
    evar= np.var(Y_hat)/.3- np.var(Y_hat) 
    
    # check whether evar indeed leads to R2 of 0.3:
    Y= Y_hat+ np.random.normal(0,math.sqrt(evar),sim.n)
    Xs= np.hstack([np.ones(n).reshape(n,1), X_wide])
    reg_result = sm.OLS(Y,Xs).fit()
     
    return [evar,reg_result.rsquared]





''' normal data '''
evar_normal= compute_and_check_evar("normal", n)

# run the function with this evar to see whether R2 is correct
sim=Simulation(distr="normal", n=n)
data= sim.generate_data()
x_ind= range(0,sim.d-1)
xs=data[:,x_ind]
xs= np.hstack([np.ones(n).reshape(n,1), xs])
y=data[:,sim.d-1]
reg_result = sm.OLS(y,xs).fit()
reg_result.rsquared 




''' poisson data '''
evar_poisson= compute_and_check_evar("poisson", n)

# run the function with this evar to see whether R2 is correct
sim=Simulation(distr="poisson", n=n)
data= sim.generate_data()
x_ind= range(0,sim.d-1)
xs=data[:,x_ind]
xs= np.hstack([np.ones(n).reshape(n,1), xs])
y=data[:,sim.d-1]
reg_result = sm.OLS(y,xs).fit()
reg_result.rsquared 





''' categorical data '''
evar_categorical= compute_and_check_evar("categorical", n)

# run the function with this evar to see whether R2 is correct
sim=Simulation(distr="categorical", n=n)
data= sim.generate_data()
x_ind= range(0,sim.d-1)
xs=data[:,x_ind]
xs= np.hstack([np.ones(n).reshape(n,1), xs])
y=data[:,sim.d-1]
reg_result = sm.OLS(y,xs).fit()
reg_result.rsquared 




''' mixed data '''
evar_mixed= compute_and_check_evar("mixed", n)

# run the function with this evar to see whether R2 is correct
sim=Simulation(distr="mixed", n=n)
data= sim.generate_data()
x_ind= range(0,sim.d-1)
xs=data[:,x_ind]
xs= np.hstack([np.ones(n).reshape(n,1), xs])
y=data[:,sim.d-1]
reg_result = sm.OLS(y,xs).fit()
reg_result.rsquared 

