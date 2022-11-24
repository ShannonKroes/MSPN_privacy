# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 15:58:31 2021

@author: Shannon
"""
import os
os.chdir(r"~\source")
from base import *
from simulation import Simulation
from scipy import stats
from statsmodels.stats.power import tt_solve_power
import pickle as cPickle
os.chdir(r"~\results\information_loss")
sim_n=Simulation()
mean_diff= sim_n.true_param/20

''' final results'''

with open(r"H0normaln1000mis999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse40_clusters=5500repetitions", "rb") as input_file:
    n30_40 = cPickle.load(input_file)

with open(r"H1normaln1000mis999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse40_clusters=5500repetitions", "rb") as input_file:
    n31_40 = cPickle.load(input_file)
       
# How many repetitions do we need?
sd_diff=np.std(n30_40[2].reshape(n30_40[2].shape[0])-n30_40[3]) 
n_n30 = tt_solve_power(effect_size=mean_diff/sd_diff, alpha=0.05, power=0.8, alternative='two-sided')
# 111.88428566909673

sd_diff=np.std(n31_40[2].reshape(n31_40[2].shape[0])-n31_40[3]) 
n_n31 = tt_solve_power(effect_size=mean_diff/sd_diff, alpha=0.05, power=0.8, alternative='two-sided')
# 119.30037503711544

# so lets do 200 for both
n3=200

tStat, pValue =  stats.ttest_rel(n30_40[2].reshape(n30_40[2].shape[0])[0:n3], n30_40[3][0:n3])
print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) 
# P-Value:0.1332460876482874 T-Statistic:1.5075871735803232

tStat, pValue =  stats.ttest_rel(n31_40[2].reshape(n31_40[2].shape[0])[0:n3], n31_40[3][0:n3])
print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) 
# P-Value:0.6086845599399623 T-Statistic:0.5127647995922548
######################################################################################################################################

with open(r"H0normaln10000mis9999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse400_clusters=no_tests=100_0priv_an_runs__100repetitions", "rb") as input_file:
    n40_40 = cPickle.load(input_file)
with open(r"H1normaln10000mis9999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse400_clusters=no_tests=100_0priv_an_runs__100repetitions", "rb") as input_file:
    n41_40 = cPickle.load(input_file)
       
# How many repetitions do we need?
sd_diff=np.std(n40_40[2].reshape(n40_40[2].shape[0])-n40_40[3]) 
n_n40 = tt_solve_power(effect_size=mean_diff/sd_diff, alpha=0.05, power=0.8, alternative='two-sided')
# 4.023613946464105

sd_diff=np.std(n41_40[2].reshape(n41_40[2].shape[0])-n41_40[3]) 
n_n41 = tt_solve_power(effect_size=mean_diff/sd_diff, alpha=0.05, power=0.8, alternative='two-sided')
# 5.147600253707778

# so lets do 10 for both
n4=10

tStat, pValue =  stats.ttest_rel(n40_40[2].reshape(n40_40[2].shape[0])[0:n4], n40_40[3][0:n4])
print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) 
# P-Value:0.5314878415457867 T-Statistic:-0.6507181954337973

tStat, pValue =  stats.ttest_rel(n41_40[2].reshape(n41_40[2].shape[0])[0:n4], n41_40[3][0:n4])
print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) 
#P-Value:0.8512948430795342 T-Statistic:0.19293320933442193
######################################################################################################################################

with open(r"H0normaln100000mis99999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse4000_clusters=75repetitions", "rb") as input_file:
    n50_40 = cPickle.load(input_file)
       

with open(r"H1normaln100000mis99999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse4000_clusters=75repetitions", "rb") as input_file:
    n51_40 = cPickle.load(input_file)
       
# How many repetitions do we need?
sd_diff=np.std(n50_40[2].reshape(n50_40[2].shape[0])-n50_40[3]) 
n_n50 = tt_solve_power(effect_size=mean_diff/sd_diff, alpha=0.05, power=0.8, alternative='two-sided')
# 4.023613946464105

sd_diff=np.std(n51_40[2].reshape(n51_40[2].shape[0])-n51_40[3]) 
n_n51 = tt_solve_power(effect_size=mean_diff/sd_diff, alpha=0.05, power=0.8, alternative='two-sided')
# 5.147600253707778

# so lets do 10 for both
n5=10

tStat, pValue =  stats.ttest_rel(n50_40[2].reshape(n50_40[2].shape[0])[0:n5], n50_40[3][0:n5])
print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) 
# P-Value:0.5314878415457867 T-Statistic:-0.6507181954337973

tStat, pValue =  stats.ttest_rel(n51_40[2].reshape(n51_40[2].shape[0])[0:n5], n51_40[3][0:n5])
print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) 
#P-Value:0.8512948430795342 T-Statistic:0.19293320933442193




