# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 19:26:53 2021

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
sim_n=Simulation(distr="mixed")
with open(r"H0mixedn1000mis999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse40_clusters=4000repetitions", "rb") as input_file:
    m30_40 = cPickle.load(input_file)

with open(r"H1mixedn1000mis999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse40_clusters=4000repetitions", "rb") as input_file:
    m31_40 = cPickle.load(input_file)
       
# Number of repetitions required
sd_diff=np.std(m30_40[2].reshape(m30_40[2].shape[0])-m30_40[3]) 
mean_diff= sim_n.true_param/20
n_m30 = tt_solve_power(effect_size=mean_diff/sd_diff, alpha=0.05, power=0.8, alternative='two-sided')
#  158.53583878672893

sd_diff=np.std(m31_40[2].reshape(m31_40[2].shape[0])-m31_40[3]) 
mean_diff= sim_n.true_param/20
n_m31 = tt_solve_power(effect_size=mean_diff/sd_diff, alpha=0.05, power=0.8, alternative='two-sided')
#  161.9250592757955
m3=200
tStat, pValue =  stats.ttest_rel(m30_40[2].reshape(m30_40[2].shape[0])[0:m3], m30_40[3][0:m3])
print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) 
# P-Value:7.473105282044227e-05 T-Statistic:4.045297153559246
tStat, pValue =  stats.ttest_rel(m31_40[2].reshape(m31_40[2].shape[0])[0:m3], m31_40[3][0:m3])
print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) 
#P-Value:2.4370353526135185e-36 T-Statistic:-15.591603168715517
######################################################################################################################################
with open(r"H0mixedn10000mis9999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse400_clusters=150repetitions", "rb") as input_file:
    m40_40 = cPickle.load(input_file)
with open(r"H1mixedn10000mis9999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse400_clusters=150repetitions", "rb") as input_file:
    m41_40 = cPickle.load(input_file)
       
# How many repetitions do we need?
sd_diff=np.std(m40_40[2].reshape(m40_40[2].shape[0])-m40_40[3]) 
mean_diff= sim_n.true_param/20
n_m40 = tt_solve_power(effect_size=mean_diff/sd_diff, alpha=0.05, power=0.8, alternative='two-sided')
# 7.125032490225395

sd_diff=np.std(m41_40[2].reshape(m41_40[2].shape[0])-m41_40[3]) 
mean_diff= sim_n.true_param/20
n_m41 = tt_solve_power(effect_size=mean_diff/sd_diff, alpha=0.05, power=0.8, alternative='two-sided')
# 7.029434774390546
m4=10
tStat, pValue =  stats.ttest_rel(m40_40[2].reshape(m40_40[2].shape[0])[0:m4], m40_40[3][0:m4])
print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) 
#P-Value:0.3290291199868086 T-Statistic:-1.0319481598950981

tStat, pValue =  stats.ttest_rel(m41_40[2].reshape(m41_40[2].shape[0])[0:m4], m41_40[3][0:m4])
print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) 
#P-Value:0.36028053197258114 T-Statistic:0.9638965549143051
######################################################################################################################################
with open(r"H0mixedn100000mis99999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse4000_clusters=150repetitions", "rb") as input_file:
    m50_40 = cPickle.load(input_file)
with open(r"H1mixedn100000mis99999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse4000_clusters=150repetitions", "rb") as input_file:
    m51_40 = cPickle.load(input_file)
######################################################################################################################################
sd_diff=np.std(m50_40[2].reshape(m50_40[2].shape[0])-m50_40[3]) 
mean_diff= sim_n.true_param/20
n_m50 = tt_solve_power(effect_size=mean_diff/sd_diff, alpha=0.05, power=0.8, alternative='two-sided')
# 4.779806523027332
sd_diff=np.std(m51_40[2].reshape(m51_40[2].shape[0])-m51_40[3]) 
mean_diff= sim_n.true_param/20
n_m51 = tt_solve_power(effect_size=mean_diff/sd_diff, alpha=0.05, power=0.8, alternative='two-sided')
# 4.627486595440289

m5=5
tStat, pValue =  stats.ttest_rel(m50_40[2].reshape(m50_40[2].shape[0])[0:m5], m50_40[3][0:m5])
print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) 

tStat, pValue =  stats.ttest_rel(m51_40[2].reshape(m51_40[2].shape[0])[0:m5], m51_40[3][0:m5])
print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) 
# P-Value:0.0001234258016860374 T-Statistic:14.736085659456734
# P-Value:0.0002858280129035911 T-Statistic:11.897790146352667
######################################################################################################################################




