# -*- coding: utf-8 -*-
"""
In this file we extract the privacy results for normally distributed data.
"""
import os
import pickle as cPickle
os.chdir(r"C:\Users\Shannon\Documents\Sanquin\Project 4\Results\Final privacy results")

with open(r"H0normaln1000mis999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse40_clusters=no_tests=1000_50priv_an_runs__50repetitions", "rb") as input_file:
    n30_40 = cPickle.load(input_file)
with open(r"H1normaln1000mis999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse40_clusters=no_tests=1000_50priv_an_runs__50repetitions", "rb") as input_file:
    n31_40 = cPickle.load(input_file)
# -5 is privacy original and -6 is mspn privacy 
priv30= n30_40[-6]
priv31= n31_40[-6]

print(np.mean(priv30,(0,1)))
print(np.mean(priv30>0,(0,1)))
# [0.99227134 0.9800129  0.98808288 0.98442749 0.98533261 0.99068817
#  0.98735725 0.98676502 2.58114714]
# [1. 1. 1. 1. 1. 1. 1. 1. 1.]

[1. 1. 1. 1. 1. 1. 1. 1. 1.]
print(np.mean(priv31,(0,1)))
print(np.mean(priv31>0,(0,1)))
# [0.98735245 0.98364139 0.98605177 0.988062   0.9883588  0.99616767
#  0.99099536 0.99223445 2.62381827]
# [1. 1. 1. 1. 1. 1. 1. 1. 1.]
######################################################################################################################################
with open(r"H0normaln10000mis9999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse400_clusters=no_tests=1000_50priv_an_runs__50repetitions", "rb") as input_file:
    n40_40 = cPickle.load(input_file)
with open(r"H1normaln10000mis9999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse400_clusters=no_tests=1000_50priv_an_runs__50repetitions", "rb") as input_file:
    n41_40 = cPickle.load(input_file)
      # -5 is privacy original and -6 is mspn privacy 
priv40= n40_40[-6]
priv41= n41_40[-6]

print(np.mean(priv40,(0,1)))
print(np.mean(priv40>0,(0,1)))
# [0.94909825 0.94468158 0.94834477 0.94745933 0.94563632 0.95159436
#  0.94797727 0.94515242 2.47757534]
# [1. 1. 1. 1. 1. 1. 1. 1. 1.]

print(np.mean(priv41,(0,1)))
print(np.mean(priv41>0,(0,1)))
######################################################################################################################################
with open(r"H0normaln100000mis99999kmeanscol_testrdct-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]no_clusters=4000_standardizeTrueecdfFalse50repetitionssave_inter", "rb") as input_file:
    n50_50 = cPickle.load(input_file)
with open(r"H1normaln100000mis99999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse4000_clusters=no_tests=1000_50priv_an_runs__50repetitions", "rb") as input_file:
    n51_50 = cPickle.load(input_file)
      # -6 is privacy original and -7 is mspn privacy 
priv50= n50_50[-7][0:10]
priv51= n51_50[-7][0:10]

print(np.mean(priv50,(0,1)))
print(np.mean(priv50>0,(0,1)))
# [0.98748112 0.97731901 0.98100755 0.97506466 0.97376466 0.97767711
#  0.97174326 0.97413188 2.55036991]
# [1. 1. 1. 1. 1. 1. 1. 1. 1.]


print(np.mean(priv51,(0,1)))
print(np.mean(priv51>0,(0,1)))



np.sum(priv50==0, (0,1))
 array([10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000])