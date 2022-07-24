# -*- coding: utf-8 -*-
"""
@author: ShannonKroes
"""
import numpy as np
import copy
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from scipy.stats import chi2_contingency, norm, binom, poisson, expon

class Simulation():
    '''
    We use this class to generate data for the simulation and to indicate variable types, which will be used as input for the construction of the MSPN.  
    '''           
    def __init__(self, distr="normal", n=10000, H0=False, sparse=False, pois_real=False):
        """
        Parameters
        ----------
        distr : str, optional
            The distribution of the independent variables, either "normal", "poisson", "categorical", or "mixed". The default is "normal", poisson data corresponds to count data in the paper.
        n : int, optional
            Sample size, in the article varied between 1000, 10,000 and 100,000. The default is 10000.
        H0 : bool, optional
            Whether the first independent variable has an effect on the outcome variable. If False, it has no effect, if it equals True the null hypothesis is false and it has a standardized effect of 0.3. The default is False.
        pois_real : bool, optional
            Whether the poisson variables with more than 7 levels are considered to be real valued in the MSPN construction. The default is False.
        non_parametric : bool, optional
            Whether the MSPN will use nonparametric estimation methods for the univariate probabilities. The default is True.

        Returns
        -------
        None.

        """
        self.distr=distr 
        self.n=n
        self.H0=H0
        self.pois_real=pois_real
        
    @property
    def d(self):
        '''
        Extract the number of variables in wide format. 

        Returns
        -------
        d : int
            The number of variables if the data are in its wide format (when there are dummy variables).
        '''
        if self.distr=="normal" or self.distr=="poisson":
            d=9
        elif self.distr=="mixed":
            d= 15
        elif self.distr=="categorical":
            d=20
        return d
    
    def sample_bin(p):
            x=np.random.choice([0, 1], size=1, p=[1-p,p])
            return x
      
    def logit(x):
        return np.log(x/(1-x))
               
    def generate_correlated_normal(self, CDF=True ):    
        '''
        Parameters
        ----------
        CDF : bool, optional
            Whether CDF values are computed. This is the case for all distr types, except for normal, as in this case the variables will 
            not be transformed to another distribution. The default is True.

        Returns
        -------
        result : numpy array
            An n by d array with correlated normal data, which will be transformed into the distribution (distr).
        '''
        dmin1=8
        mean =np.repeat(0 ,dmin1)
        cov = np.reshape(np.repeat(np.repeat(0.3, dmin1), dmin1), (dmin1,dmin1))  # correlatie van .3?
        np.fill_diagonal(cov,1)
        X= np.random.multivariate_normal(mean, cov, self.n)    
    
        if CDF==True:
            CDF_values= norm.cdf(X)
            result=copy.deepcopy(CDF_values)
        else:
            result= np.round(X,2)
        
        return result
        
    def ordered(self):
        '''
        A function that yields an array that indicates which variable is ordered, which is needed to transform the data from narrow to wide.
        Always add a 1 at the end for the dependent variable, which is always ordered.

        Returns
        -------
        ordered : numpy array consisting of bool elements.

        '''
        if self.distr=="normal" or self.distr=="poisson":
            ordered= np.ones(self.d)
        elif self.distr=="categorical":
            ordered= np.concatenate([ np.zeros(self.d-1), np.ones(1)])
        elif self.distr=="mixed":
            ordered= np.concatenate([np.ones(4), np.zeros(self.d-5), np.ones(1)])
            
        return ordered
    
    def ordered_narrow(self):
        '''
        A function that yields an array that indicates which variable is ordered, which is needed to select an appropriate privacy metric.
        Always add a 1 at the end for the dependent variable, which is always ordered.

        Returns
        -------
        ordered : numpy array consisting of bool elements.

        '''
        if self.distr=="normal" or self.distr=="poisson":
            ordered= np.ones(9)

        elif self.distr=="categorical":
            ordered= np.concatenate([ np.zeros(8), np.ones(1)])
        elif self.distr=="mixed":
            ordered= np.concatenate([np.ones(3), np.zeros(5), np.ones(1)])
            
        return ordered

    def independent_variables(self):
        '''
        Generate the independent variables by transforming the correlated normal data
        to the chosen "distr". 

        Returns
        -------
        With this function we generate a list containing:
        evar : float
            The error variance that ensures R squared of 0.3 (which we computed in the file compute_evars.py).
        X : np.array
            The correlated indepenent variables in a narrow format.
        levels : np.array[int]
            The number of levels each variable has (=1 for continuous variables).
        sds : np.array[float]
            The standard deviation of each variable which we use to standardize the beta coefficients.

        '''
        if self.distr=="normal":
            
            sds=(np.repeat(1,self.d-1))
            evar=5.208
            X= self.generate_correlated_normal(CDF=False)
            levels=np.repeat(1, self.d-1)
            

        elif self.distr=="poisson":
            params= np.array([1,1,1,4,4,4,10,10])
            sds= np.sqrt(params)
            evar=4.95
            CDF_values= self.generate_correlated_normal()
            X= poisson.ppf(CDF_values, params)
            levels=np.repeat(1, self.d-1)                    
            
        elif self.distr=="categorical":
            params= np.concatenate([[0.2, 0.3, 0.4, 0.5, 0.6,.2,.2], np.repeat(1/7,12)])
            CDF_values= self.generate_correlated_normal()
            t_CDF_values= np.transpose(CDF_values)
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
            evar=2.122
            sds= np.zeros(self.d-1)
            sds[0]=15
            sds[1]= 4
            sds[2]=15
            sds[3]=np.sqrt(6)
            sds[4:]= params[4:]*(1-params[4:])
            
        return [X, sds, evar, levels]      
        

    def sds_levels(self):
        '''
        Returns
        -------
        list
            sds : np.array[float]
                The standard deviation of every independent variable.
            levels : np.array[int]
                The number of values each variable can take on (1 if continuous)

        '''
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
        '''
        With this function we actually generate the data, using the other functions in the class.
        This function generates the independent variables and combines categorical vectors that represent
        different levels of the same variable. 
        Returns
        -------
        data : np.array
            An n by d array which functions as the original, privacy-sensitive data in the article. 
        '''
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
        beta_raw=0.3
        intercept=0
        levels_long= np.concatenate(np.asarray(list(map(lambda x: np.repeat(x,x), levels))))
        betas_raw=np.repeat(beta_raw, sum(levels))
        betas_raw_levels= betas_raw/levels_long
        betas= betas_raw_levels/sds
    
        if self.H0==True:
            betas[0]=0            
        Y_hat=intercept+ np.sum(betas*X_wide, axis=1)
        Y= Y_hat+ np.random.normal(0,math.sqrt(evar),self.n)        
        data= np.hstack([X_wide, np.round(Y,2).reshape(self.n,1)])                          
        levels= np.concatenate([levels,np.ones(1)])
        levels= np.array(levels, dtype=int)

        return data
        
    @property
    def true_param(self):
        '''
        This function indicates the true beta parameter value of the independent variable of interest.
        This is used to estimate bias.

        Returns
        -------
        true_param : float
            True beta parameter value.

        '''
        sds, levels= self.sds_levels()
        
        if self.H0==True:
            beta_raw=0
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
        '''
        Extract input required to construct the MSPN. This is an array with 9 elements that indicates for each variable
        whether it is discrete or real-valued using the class MetaType from the spflow package.

        Returns
        -------
        ds_context : Context class from spn.structure.Base

        '''
        if self.distr== "categorical":
            ds_context=Context(meta_types=np.concatenate([np.repeat(MetaType.DISCRETE,8),[MetaType.REAL]]))
        elif self.distr=="normal":
            ds_context = Context(meta_types=[MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL])
        elif self.distr=="poisson":
            if self.pois_real==True:
                ds_context = Context(meta_types=[MetaType.DISCRETE, MetaType.DISCRETE, MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL])
            else:
                ds_context = Context(meta_types=[MetaType.DISCRETE, MetaType.DISCRETE, MetaType.DISCRETE, MetaType.DISCRETE, MetaType.DISCRETE, MetaType.DISCRETE, MetaType.DISCRETE, MetaType.DISCRETE, MetaType.REAL])          
        elif self.distr=="mixed":
            ds_context=Context(meta_types=np.concatenate([[MetaType.REAL,MetaType.REAL,MetaType.REAL,MetaType.DISCRETE],np.repeat(MetaType.DISCRETE, 4), [MetaType.REAL]]))

        return ds_context        
    