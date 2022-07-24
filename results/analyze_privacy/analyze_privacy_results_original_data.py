# -*- coding: utf-8 -*-
"""
In this file we extract the original data privacy results 
"""
import os
import pickle as cPickle
import numpy as np
os.chdir(r"C:/Users/Shannon/Documents/Sanquin/Project 4/MSPN_privacy/results/privacy/original_data")
######################################################################################################################################
###############################     normal          ###################################################################
######################################################################################################################################
with open(r"originalH0normaln100050priv_or_runs__50repetitions", "rb") as input_file:
    n30_40 = cPickle.load(input_file)
with open(r"originalH1normaln100050priv_or_runs__50repetitions", "rb") as input_file:
    n31_40 = cPickle.load(input_file)
priv30= n30_40[0]
priv31= n31_40[0]

print(np.mean(priv30,(0,1)))
print(np.mean(priv30>0,(0,1)))
# [0. 0. 0. 0. 0. 0. 0. 0. 0.]
# [0. 0. 0. 0. 0. 0. 0. 0. 0.]
print(np.mean(priv31,(0,1)))
print(np.mean(priv31>0,(0,1)))
# [0. 0. 0. 0. 0. 0. 0. 0. 0.]
# [0. 0. 0. 0. 0. 0. 0. 0. 0.]
######################################################################################################################################
with open(r"originalH0normaln1000050priv_or_runs__50repetitions", "rb") as input_file:
    n40_40 = cPickle.load(input_file)
with open(r"originalH1normaln1000050priv_or_runs__50repetitions", "rb") as input_file:
    n41_40 = cPickle.load(input_file)
priv40= n40_40[0]
priv41= n41_40[0]

print(np.mean(priv40,(0,1)))
print(np.mean(priv40>0,(0,1)))
# [0. 0. 0. 0. 0. 0. 0. 0. 0.]
# [0. 0. 0. 0. 0. 0. 0. 0. 0.]
print(np.mean(priv41,(0,1)))
print(np.mean(priv41>0,(0,1)))
# [0. 0. 0. 0. 0. 0. 0. 0. 0.]
# [0. 0. 0. 0. 0. 0. 0. 0. 0.]
######################################################################################################################################
with open(r"originalH0normaln10000050priv_or_runs__50repetitions", "rb") as input_file:
    n50_40 = cPickle.load(input_file)
with open(r"originalH1normaln10000050priv_or_runs__50repetitions", "rb") as input_file:
    n51_40 = cPickle.load(input_file)
priv50= n50_40[0][0:10]
priv51= n51_40[0][0:10]

print(np.mean(priv50,(0,1)))
print(np.mean(priv50>0,(0,1)))
# [0. 0. 0. 0. 0. 0. 0. 0. 0.]
# [0. 0. 0. 0. 0. 0. 0. 0. 0.]
print(np.mean(priv51,(0,1)))
print(np.mean(priv51>0,(0,1)))
# [0. 0. 0. 0. 0. 0. 0. 0. 0.]
# [0. 0. 0. 0. 0. 0. 0. 0. 0.]
######################################################################################################################################
###############################     poisson (count)          ###################################################################
######################################################################################################################################
with open(r"originalH0poissonn100050priv_or_runs__50repetitions", "rb") as input_file:
    p30_40 = cPickle.load(input_file)
with open(r"originalH1poissonn100050priv_or_runs__50repetitions", "rb") as input_file:
    p31_40 = cPickle.load(input_file)
priv30= p30_40[0]
priv31= p31_40[0]

print(np.mean(priv30,(0,1)))
print(np.mean(priv30>0,(0,1)))
# [0.         0.         0.         0.         0.         0.
#  0.         0.         0.00300662]
# [0.     0.     0.     0.     0.     0.     0.     0.     0.0018]
print(np.mean(priv31,(0,1)))
print(np.mean(priv31>0,(0,1)))
# [0.         0.         0.         0.         0.         0.
#  0.         0.         0.00300662]
# [0.     0.     0.     0.     0.     0.     0.     0.     0.0018]
######################################################################################################################################

with open(r"originalH0poissonn1000050priv_or_runs__50repetitions", "rb") as input_file:
    p40_40 = cPickle.load(input_file)
with open(r"originalH1poissonn1000050priv_or_runs__50repetitions", "rb") as input_file:
    p41_40 = cPickle.load(input_file)
priv40= p40_40[0]
priv41= p41_40[0]

print(np.mean(priv40,(0,1)))
print(np.mean(priv40>0,(0,1)))
# [0.         0.         0.         0.         0.         0.
#  0.00141421 0.         0.03624827]
# [0.     0.     0.     0.     0.     0.     0.0004 0.     0.02  ]
print(np.mean(priv41,(0,1)))
print(np.mean(priv41>0,(0,1)))
# [0.         0.         0.         0.         0.         0.
#  0.00141421 0.         0.03624827]
# [0.     0.     0.     0.     0.     0.     0.0004 0.     0.02  ]
######################################################################################################################################
with open(r"originalH0poissonn10000050priv_or_runs__50repetitions", "rb") as input_file:
    p50_40 = cPickle.load(input_file)
with open(r"originalH1poissonn10000050priv_or_runs__50repetitions", "rb") as input_file:
    p51_40 = cPickle.load(input_file)
priv50= p50_40[0][0:10]
priv51= p51_40[0][0:10]

print(np.mean(priv50,(0,1)))
print(np.mean(priv50>0,(0,1)))
# [0.         0.00070711 0.00070711 0.         0.00070711 0.00353553
#  0.00707107 0.00282843 0.23915044]
# [0.    0.001 0.001 0.    0.001 0.002 0.003 0.002 0.136]
print(np.mean(priv51,(0,1)))
print(np.mean(priv51>0,(0,1)))
# [0.         0.00070711 0.00070711 0.         0.00070711 0.00353553
#  0.00707107 0.00282843 0.23915044]
# [0.    0.001 0.001 0.    0.001 0.002 0.003 0.002 0.136]
######################################################################################################################################
###############################     categorical          ###################################################################
######################################################################################################################################

with open(r"originalH0categoricaln100050priv_or_runs__50repetitions", "rb") as input_file:
    c30_40 = cPickle.load(input_file)
with open(r"originalH1categoricaln100050priv_or_runs__50repetitions", "rb") as input_file:
    c31_40 = cPickle.load(input_file)
priv30= c30_40[0]
priv31= c31_40[0]

print(np.mean(priv30,(0,1)))
print(np.mean(priv30>0,(0,1)))
# [0.0006     0.001      0.0008     0.0028     0.0026     0.0012
#  0.00206667 0.00173333 0.45995451]
# [0.0006 0.001  0.0008 0.0028 0.0026 0.0024 0.0122 0.0102 0.4432]
print(np.mean(priv31,(0,1)))
print(np.mean(priv31>0,(0,1)))
# [0.0008     0.001      0.0008     0.0028     0.0026     0.0012
#  0.00206667 0.00173333 0.45995451]
# [0.0008 0.001  0.0008 0.0028 0.0026 0.0024 0.0122 0.0102 0.4432]
######################################################################################################################################
with open(r"originalH0categoricaln1000050priv_or_runs__50repetitions", "rb") as input_file:
    c40_40 = cPickle.load(input_file)
with open(r"originalH1categoricaln1000050priv_or_runs__50repetitions", "rb") as input_file:
    c41_40 = cPickle.load(input_file)
priv40= c40_40[0]
priv41= c41_40[0]

print(np.mean(priv40,(0,1)))
print(np.mean(priv40>0,(0,1)))
# [0.0078     0.0086     0.0156     0.017      0.0272     0.0103
#  0.01993333 0.0227     1.09408713]
# [0.0078 0.0086 0.0156 0.017  0.0272 0.0204 0.105  0.1186 0.8854]
print(np.mean(priv41,(0,1)))
print(np.mean(priv41>0,(0,1)))
# [0.008      0.0086     0.0156     0.017      0.0272     0.0103
#  0.01993333 0.0227     1.09408713]
# [0.008  0.0086 0.0156 0.017  0.0272 0.0204 0.105  0.1186 0.8854]
######################################################################################################################################
with open(r"originalH0categoricaln10000050priv_or_runs__50repetitions", "rb") as input_file:
    c50_40 = cPickle.load(input_file)
with open(r"originalH1categoricaln10000050priv_or_runs__50repetitions", "rb") as input_file:
    c51_40 = cPickle.load(input_file)
      # -6 is privacy original and -7 is mspn privacy 
priv50= c50_40[0][0:10]
priv51= c51_40[0][0:10]

print(np.mean(priv50,(0,1)))
print(np.mean(priv50>0,(0,1)))
# [0.066      0.081      0.128      0.122      0.149      0.094
#  0.159      0.15133333 1.40376111]
# [0.066 0.081 0.128 0.122 0.149 0.179 0.492 0.48  0.997]
print(np.mean(priv51,(0,1)))
print(np.mean(priv51>0,(0,1)))
# [0.067      0.081      0.128      0.122      0.149      0.094
#  0.159      0.15133333 1.40376111]
# [0.067 0.081 0.128 0.122 0.149 0.179 0.492 0.48  0.997]
######################################################################################################################################
###############################     mixed          ###################################################################
######################################################################################################################################
with open(r"originalH0mixedn100050priv_or_runs__50repetitions", "rb") as input_file:
    m30_40 = cPickle.load(input_file)
with open(r"originalH1mixedn100050priv_or_runs__50repetitions", "rb") as input_file:
    m31_40 = cPickle.load(input_file)
priv30= m30_40[0]
priv31= m31_40[0]

print(np.mean(priv30,(0,1)))
print(np.mean(priv30>0,(0,1)))
# [0. 0. 0. 0. 0. 0. 0. 0. 0.]
# [0. 0. 0. 0. 0. 0. 0. 0. 0.]
print(np.mean(priv31,(0,1)))
print(np.mean(priv31>0,(0,1)))
# [0. 0. 0. 0. 0. 0. 0. 0. 0.]
# [0. 0. 0. 0. 0. 0. 0. 0. 0.]
######################################################################################################################################
with open(r"originalH0mixedn1000050priv_or_runs__50repetitions", "rb") as input_file:
    m40_40 = cPickle.load(input_file)
with open(r"originalH1mixedn1000050priv_or_runs__50repetitions", "rb") as input_file:
    m41_40 = cPickle.load(input_file)
priv40= m40_40[0]
priv41= m41_40[0]

print(np.mean(priv40,(0,1)))
print(np.mean(priv40>0,(0,1)))
# [0. 0. 0. 0. 0. 0. 0. 0. 0.]
# [0. 0. 0. 0. 0. 0. 0. 0. 0.]
print(np.mean(priv41,(0,1)))
print(np.mean(priv41>0,(0,1)))
# [0. 0. 0. 0. 0. 0. 0. 0. 0.]
# [0. 0. 0. 0. 0. 0. 0. 0. 0.]
######################################################################################################################################
with open(r"originalH0mixedn10000050priv_or_runs__50repetitions", "rb") as input_file:
    m50_40 = cPickle.load(input_file)
with open(r"originalH1mixedn10000050priv_or_runs__50repetitions", "rb") as input_file:
    m51_40 = cPickle.load(input_file)
priv50= m50_40[0][0:10]
priv51= m51_40[0][0:10]

print(np.mean(priv50,(0,1)))
print(np.mean(priv50>0,(0,1)))
# [0. 0. 0. 0. 0. 0. 0. 0. 0.]
# [0. 0. 0. 0. 0. 0. 0. 0. 0.]
print(np.mean(priv51,(0,1)))
print(np.mean(priv51>0,(0,1)))
# [0. 0. 0. 0. 0. 0. 0. 0. 0.]
# [0. 0. 0. 0. 0. 0. 0. 0. 0.]