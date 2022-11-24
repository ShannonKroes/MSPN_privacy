"""
In this file we extract the privacy results for data with categorical predictors. 
"""
import os
import pickle as cPickle
import numpy as np
os.chdir(r"C:/Users/Shannon/Documents/Sanquin/Project 4/MSPN_privacy/results/privacy/mspn")
with open(r"H0categoricaln1000mis999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse40_clusters=no_tests=1000_50priv_an_runs__50repetitions", "rb") as input_file:
    c30_40 = cPickle.load(input_file)
with open(r"H1categoricaln1000mis999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse40_clusters=no_tests=1000_50priv_an_runs__50repetitions", "rb") as input_file:
    c31_40 = cPickle.load(input_file)
      # -5 is privacy original and -6 is mspn privacy 
priv30= c30_40[-6]
priv31= c31_40[-6]

print(np.mean(priv30,(0,1)))
print(np.mean(priv30>0,(0,1)))
# [0.87816    0.90394    0.92759    0.93708    0.93389    0.91612
#  0.91153429 0.91562286 1.37701089]
# [1. 1. 1. 1. 1. 1. 1. 1. 1.]

print(np.mean(priv31,(0,1)))
print(np.mean(priv31>0,(0,1)))
# privacy_an=[0.84966    0.90361    0.92395    0.93649    0.9303     0.91586667
#  0.91109429 0.91652286 1.38582166]
# prop_privacy_an[1. 1. 1. 1. 1. 1. 1. 1. 1.]
######################################################################################################################################
with open(r"H0categoricaln10000mis9999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse400_clusters=no_tests=1000_50priv_an_runs__50repetitions", "rb") as input_file:
    c40_40 = cPickle.load(input_file)
with open(r"H1categoricaln10000mis9999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse400_clusters=no_tests=1000_50priv_an_runs__50repetitions", "rb") as input_file:
    c41_40 = cPickle.load(input_file)
      # -5 is privacy original and -6 is mspn privacy 
priv40= c40_40[-6]
priv41= c41_40[-6]

print(np.mean(priv40,(0,1)))
print(np.mean(priv40>0,(0,1)))
# [0.89422    0.90864    0.92559    0.92457    0.91722    0.91067333
#  0.90991143 0.9106     1.36528314]
# [1. 1. 1. 1. 1. 1. 1. 1. 1.]
print(np.mean(priv41,(0,1)))
print(np.mean(priv41>0,(0,1)))
# [0.86521    0.9086     0.92396    0.92413    0.91694    0.91183333
#  0.91102571 0.91238857 1.36702495]
# [1. 1. 1. 1. 1. 1. 1. 1. 1.]
######################################################################################################################################
with open(r"H0categoricaln100000mis99999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse4000_clusters=no_tests=1000_50priv_an_runs__50repetitions", "rb") as input_file:
    c50_50 = cPickle.load(input_file)
with open(r"H1categoricaln100000mis99999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse4000_clusters=no_tests=1000_50priv_an_runs__50repetitions", "rb") as input_file:
    c51_50 = cPickle.load(input_file)
      # -6 is privacy original and -7 is mspn privacy 
priv50= c50_50[-6][0:10]
priv51= c51_50[-6][0:10]

print(np.mean(priv50,(0,1)))
print(np.mean(priv50>0,(0,1)))
# [0.9078     0.9202     0.93425    0.9341     0.9245     0.9112
#  0.90565714 0.9049     1.42556174]
# [1. 1. 1. 1. 1. 1. 1. 1. 1.]
print(np.mean(priv51,(0,1)))
print(np.mean(priv51>0,(0,1)))
# [0.888      0.92295    0.93275    0.9347     0.92795    0.9132
#  0.90814286 0.90667143 1.42642306]
# [1. 1. 1. 1. 1. 1. 1. 1. 1.]