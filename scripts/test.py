# -*- coding: utf-8 -*-
"""
Test run for the simulation with only a few evaluations for normally distributed data with 10 repetitions and evaluating privacy for the first ten individuals of the first run. 
"""
import os
from evaluate_privacy_information_loss import simulation_spn_privacy
ans= np.array([100000])
result= simulation_spn_privacy(distr="normal", repetitions=10, threshold=-1, mis=999, n=1000, rows="kmeans", no_clusters=40, H0=True, an_sample_size=ans, standardize=True, priv_an_runs= 1, no_tests=10, priv_or_runs=1)
