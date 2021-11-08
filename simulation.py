# -*- coding: utf-8 -*-
"""
Created on Sat Nov 06 23:13:42 2021
@author: Shannon Kroes
In this file the functions are included to generate data,
compute privacy, compute information loss and run the simulations.
throughout this file we use
n = number of samples (e.g. health records/people)
d = number of variables
levels= number of values each value can take on
ordered= whether each variable has a natural ordering (binary and categorical variables do not, whereas continuous or count variables do)
"""


import copy
import math

import numpy as np
from scipy.stats import norm, poisson, expon
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType


class Simulation():
    # We use this class to generate data for the simulation

    def __init__(self, distr="normal", n=10000, H0=False, logistic=False, sparse=False, pois_real=False, pois_allreal=False, non_parametric=True):
        self.distr=distr
        self.n=n
        self.H0=H0
        self.logistic=logistic
        self.sparse=sparse
        self.pois_real=pois_real
        self.pois_allreal=pois_allreal
        self.non_parametric= non_parametric


    @property
    def d(self):
        # this value d is the number of variables if the data are in its wide format
        # so when there are dummy variables
        if self.distr=="normal" or self.distr=="poisson":
            d=9
        elif self.distr=="mixed":
            d= 15
        elif self.distr=="categorical":
            d=20
        return d


    @staticmethod
    def sample_bin(p):
            x=np.random.choice([0, 1], size=1, p=[1-p,p])
            return x


    @staticmethod
    def logit(x):
        return np.log(x/(1-x))


    def generate_correlated_normal(self, var=1, covariance= 0.3, mean=0, CDF=True ):
        # Depending on whether we sample the values to use them as data
        # or later transform them and only generate them to create a certain level of dependence
        # toround and CDF should be altered, that is, in the latter case they should be False and True

        # the number of independent variables (d_ind) equals d-1
        #dmin1=self.d-1
        dmin1=8

        mean =np.repeat(mean ,dmin1)

        cov = np.reshape(np.repeat(np.repeat(covariance, dmin1), dmin1), (dmin1,dmin1))  # correlatie van .3?
        np.fill_diagonal(cov,var)

        X= np.random.multivariate_normal(mean, cov, self.n)

        if CDF==True:
            CDF_values= norm.cdf(X)
            result=copy.deepcopy(CDF_values)
        else:
            result= np.round(X,2)

        return result


    def ordered(self):
        # a function that indicates which variable is ordered
        # which is needed to transform the data from narrow to wide
        # always add a one at the end for the dependent variable, which is always positive
        if self.distr=="normal" or self.distr=="poisson":
            ordered= np.ones(self.d)

        elif self.distr=="categorical":
            ordered= np.concatenate([ np.zeros(self.d-1), np.ones(1)])
        elif self.distr=="mixed":
            ordered= np.concatenate([np.ones(4), np.zeros(self.d-5), np.ones(1)])

        return ordered


    def ordered_narrow(self):
        # a function that indicates which variable is ordered
        # which is needed to select an appropriate privacy metric

        if self.distr=="normal" or self.distr=="poisson":
            ordered= np.ones(9)

        elif self.distr=="categorical":
            ordered= np.concatenate([ np.zeros(8), np.ones(1)])
        elif self.distr=="mixed":
            ordered= np.concatenate([np.ones(3), np.zeros(5), np.ones(1)])

        return ordered


    def is_discrete(self):
        # a function that indicates which variable is ordered
        # which is needed to select an appropriate privacy metric
        if self.distr=="normal" :
            is_discrete= np.zeros(9, dtype=bool)
        elif self.distr=="categorical"or self.distr=="poisson":
            is_discrete= np.concatenate([np.ones(8, dtype=bool), np.zeros(1, dtype=bool)])
        elif self.distr=="mixed":
            is_discrete= np.concatenate([np.zeros(3, dtype=bool), np.ones(5, dtype=bool), np.zeros(1, dtype=bool)])

        return is_discrete


    def independent_variables(self):
        # with this function we generate:
        # evar: the error variance that ensures R squared of 0.3 (which we computed in the file compute_sds.py)
        # X: the correlated indepenent variables in a narrow format
        # levels: the number of levels each variable has (=1 for continuous variables)
        # sds: the standard deviation of each variable which we use to standardize the beta coefficients
        if self.distr=="normal":

            sds=(np.repeat(1,self.d-1))
            if self.logistic==True:
                evar=0.720
            else:
                evar=5.208
            X= self.generate_correlated_normal(CDF=False)
            levels=np.repeat(1, self.d-1)

        elif self.distr=="poisson":
            params= np.array([1,1,1,4,4,4,10,10])
            sds= np.sqrt(params)
            if self.logistic==True:
                evar=0.372
            else:
                evar=4.95
            CDF_values= self.generate_correlated_normal()
            X= poisson.ppf(CDF_values, params)
            levels=np.repeat(1, self.d-1)

        elif self.distr=="categorical":
            #params= np.array([0.2, 0.3, 0.4, 0.5, 0.6,.2, 1/7, 1/7])

            params= np.concatenate([[0.2, 0.3, 0.4, 0.5, 0.6,.2,.2], np.repeat(1/7,12)])

            CDF_values= self.generate_correlated_normal()
            t_CDF_values= np.transpose(CDF_values)

            if self.logistic==True:
                evar=0
            else:
                evar=1.172

            levels= np.concatenate([np.repeat(1, 5),[ 2, 6, 6]])
            X= np.empty((8,self.n), dtype=float)

            param_inds= np.concatenate([np.zeros(1), np.cumsum(levels)])

            for i in range(8):

                X[i]= np.digitize(t_CDF_values[i],np.cumsum(params[int(param_inds[i]):int(param_inds[i+1])]))

            X= np.transpose(X)
            sds= np.sqrt(params*(1-params))

        elif self.distr=="mixed":
            params= np.concatenate([[15,4,15,6, 0.3,.2,.3], np.repeat(1/7,6), [.1]])
            bin_ind_short= range(4,8)

            CDF_values= self.generate_correlated_normal()
            t_CDF_values= np.transpose(CDF_values)
            X1= norm.ppf(t_CDF_values[0], 60, 15).reshape(self.n,1)
            X2= expon.ppf(t_CDF_values[1],scale= 4).reshape(self.n,1)
            X3= norm.ppf(t_CDF_values[2], 120, 15).reshape(self.n,1)
            X4= poisson.ppf(t_CDF_values[3], 6).reshape(self.n,1)

            levels= np.concatenate([np.repeat(1, 4),[ 1,2,6,1]])
            bin_params=params[4:]
            bin_levels= levels[bin_ind_short]
            param_inds= np.concatenate([np.zeros(1), np.cumsum(bin_levels)])

            X_bin= np.empty((len(bin_ind_short),self.n), dtype=float)
            for i in range(len(bin_ind_short)):
                X_bin[i]= np.digitize(t_CDF_values[i],np.cumsum(bin_params[int(param_inds[i]):int(param_inds[i+1])]))

            X=  np.hstack([X1, X2, X3, X4, np.transpose(X_bin)])

            if self.sparse==True:
                if self.logistic==True:
                    evar=0.298
                else:
                    evar=1.395
            else:
                if self.logistic==True:
                    evar=0.343
                else:
                    #evar=3.445
                    evar=2.122

            sds= np.zeros(self.d-1)
            sds[0]=15
            sds[1]= 4
            sds[2]=15
            sds[3]=np.sqrt(6)
            sds[4:]= params[4:]*(1-params[4:])

        return [X, sds, evar, levels]


    def sds_levels(self):
        if self.distr=="normal":
            sds=(np.repeat(1,self.d-1))
            levels=np.repeat(1, self.d-1)
        if self.distr=="poisson":
            params= np.array([1,1,1,4,4,4,10,10])
            sds= np.sqrt(params)
            levels=np.repeat(1, self.d-1)

        if self.distr=="categorical":
            params= np.concatenate([[0.2, 0.3, 0.4, 0.5, 0.6,.2,.2], np.repeat(1/7,12)])
            sds= np.sqrt(params*(1-params))
            levels= np.concatenate([np.repeat(1, 5),[ 2, 6, 6]])

        if self.distr=="mixed":
            params= np.concatenate([[15,4,15,6, 0.3,.2,.3], np.repeat(1/7,6), [.1]])
            sds= np.zeros(self.d-1)

            # we have to replace this with the true sds
            sds[0]=15
            sds[1]= 4
            sds[2]=15
            sds[3]=np.sqrt(6)
            sds[4:]= params[4:]*(1-params[4:])

            levels= np.concatenate([np.repeat(1, 4),[ 1,2,6,1]])

        return [sds, levels]


    def generate_data(self):
        # with this function we compute the betas
        # dependent on whether we are running a logistic regression
        # and on the sds of the
        X,sds,evar,levels= self.independent_variables()
        # transform X so that it has dummy variables
        X_wide= np.zeros((self.n, self.d-1))
        ordered=self.ordered()

        for i in range(8):
            for j in range(levels[i]):
                ind_wide=np.sum(levels[0:i])+j

                if ordered[ind_wide]==0:
                    X_wide.T[ind_wide]= (X.T[i]==(j+1))
                else:
                    X_wide.T[ind_wide]= X.T[i]

        if self.logistic==True:
            beta_raw=-.19
            intercept=-.85
        else:
            beta_raw=0.3
            intercept=0

        levels_long= np.concatenate(np.asarray(list(map(lambda x: np.repeat(x,x), levels))))
        betas_raw=np.repeat(beta_raw, sum(levels))
        betas_raw_levels= betas_raw/levels_long
        betas= betas_raw_levels/sds

        if self.sparse:
            betas[1]=0
            betas[range(7,14)]=0

        if self.H0==True:
            betas[0]=0

        Y_hat=intercept+ np.sum(betas*X_wide, axis=1)
        Y= Y_hat+ np.random.normal(0,math.sqrt(evar),self.n)

        if self.logistic==True:
            # compute the probability of 1 for every individual and then
            # sample 0 and 1 with these probabilities
            Y= map(self.sample_bin, self.logit(Y))

        data= np.hstack([X_wide, np.round(Y,2).reshape(self.n,1)])

        levels= np.concatenate([levels,np.ones(1)])
        levels= np.array(levels, dtype=int)

        return data

    @property
    def true_param(self):

        sds, levels= self.sds_levels()

        if self.H0==True:
            beta_raw=0
        else:
            if self.logistic==True:
                if self.distr=="mixed":
                     beta_raw=-0.16
                     intercept= -188.4
                beta_raw=-.19
                intercept=-.85
            else:
                beta_raw=0.3
                intercept=0
        levels_long= np.concatenate(np.asarray(list(map(lambda x: np.repeat(x,x), levels))))
        betas_raw=np.repeat(beta_raw, sum(levels))
        betas_raw_levels= betas_raw/levels_long
        betas= betas_raw_levels/sds
        true_param=betas[0]

        return true_param


    def ds_context(self):

        if self.non_parametric==True:
            if self.distr== "categorical":
                ds_context=Context(meta_types=np.concatenate([np.repeat(MetaType.DISCRETE,8),[MetaType.REAL]]))
            elif self.distr=="normal":
                ds_context = Context(meta_types=[MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL])
            elif self.distr=="poisson":
                if self.pois_real==True:
                    ds_context = Context(meta_types=[MetaType.DISCRETE, MetaType.DISCRETE, MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL])
                elif self.pois_allreal==True :
                    ds_context = Context(meta_types=[MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL])
                else:
                    ds_context = Context(meta_types=[MetaType.DISCRETE, MetaType.DISCRETE, MetaType.DISCRETE, MetaType.DISCRETE, MetaType.DISCRETE, MetaType.DISCRETE, MetaType.DISCRETE, MetaType.DISCRETE, MetaType.REAL])
            elif self.distr=="mixed":
                ds_context=Context(meta_types=np.concatenate([[MetaType.REAL,MetaType.REAL,MetaType.REAL,MetaType.REAL],np.repeat(MetaType.DISCRETE, 4), [MetaType.REAL]]))

        return ds_context