# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:14:40 2022

@author: Shannon
"""



import os
file_wd = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_wd)
import numpy as np 
from spn.algorithms.Sampling import sample_instances
current_wd = os.getcwd()
from numpy.random.mtrand import RandomState
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from spn.algorithms.LearningWrappers import learn_mspn
from base import save_object


def anonymize_data(data, an_sample_size=100000, discrete=None, n_clusters=None, seed=1901, save_mspn=False, file_dest=None, rows = "kmeans", parties = None):
    '''
    Function to anonymize a data set. 
    
    data = numpy array: with the variables in the columns and records (people) in the rows.
           the categorical variables must be included as dummy variables.
    an_sample_size = int: the size of the anonymized data set. Recommended to be larger than the original data set. 
    n_clusters = int: the number of clusters created when constructing the mspn (mixed sum-product network)
                  this induces a privacy-utility trade-off; more clusters is higher utility but lower privacy.
                  If left empty, the algorithm chooses the number of clusters such that there is approximately
                  an average of 25 individuals per cluster.
    discrete = numpy array: contains the variable indices for the discrete variables. Those that are not discrete
             are treated as real variables. For example, if the first two variables are discrete and the rest is real-valued
             it would look like: np.array([0,1]). If left as None, all variables are considered to be real-valued. 
    seed = int: specfying the seed (optional). We use the same seed for clustering and sampling. 
    save_mspn = bool: whether the mspn is saved in destination folder "file_dest"
    file_dest = str: path to folder where mspn should be saved. 
    rows = str: the method used to cluster the data. 
    parties = list: distribution of variables between parties, only required to test the approach in a distributed setting. 
    
    Output
    AN = numpy array: contains synthetic, anonymized data with the same number of variables as "data" and 
         an_sample_size records. 
    '''
    
    n,d = data.shape
    meta_types = np.repeat(MetaType.REAL,d)
    if discrete is not None:
        meta_types[discrete]=MetaType.DISCRETE
    ds_context=Context(meta_types=meta_types)
    ds_context.add_domains(data)

    if n_clusters==None:
        n_clusters = int(n/25)
        # if rows == "vertical_kmeans":
            # vert_n_clusters = int(n_clusters*4)
    print("working on mspn...")
    mspn = learn_mspn(data, ds_context, min_instances_slice=n-1, rows=rows, threshold=-1, n_clusters=n_clusters, standardize=True, parties = parties, rand_gen = seed)    
    if save_mspn:
        os.chdir(file_dest)
        save_object(mspn, "mspn")
        
    print("generating synthetic data...")
    AN= sample_instances(mspn, np.array([[np.repeat(np.nan,d)] * an_sample_size]).reshape(-1, data.shape[1]), RandomState(seed))
    
    if rows == "vertical_kmeans":
        final_no_clusters = len(mspn.children)
        print("eventual number of clusters is "+str(final_no_clusters)+" instead of "+str(n_clusters))
    
    return AN

if __name__ == "__main__":
    # This is an example with normally distributed data with 1000 records. 
    from simulation import Simulation
    sim=Simulation()
    data= sim.generate_data()
    anonymize_data(data)
