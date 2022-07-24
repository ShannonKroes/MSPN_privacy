# -*- coding: utf-8 -*-
"""
In this file we extract the privacy results 
"""
import os
import pickle as cPickle
import numpy as np
os.chdir(r"C:/Users/Shannon/Documents/Sanquin/Project 4/MSPN_privacy/results/privacy/mspn")
######################################################################################################################################
###############################     normal          ###################################################################
######################################################################################################################################
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
# [0.93233059 0.93062728 0.93253777 0.9334774  0.93177445 0.93790078
#  0.93096036 0.93473591 2.45983858]
# [1. 1. 1. 1. 1. 1. 1. 1. 1.]

######################################################################################################################################
with open(r"H0normaln100000mis99999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse4000_clusters=no_tests=1000_50priv_an_runs__50repetitions", "rb") as input_file:
    n50_50 = cPickle.load(input_file)
with open(r"H1normaln100000mis99999kmeanscol_testrdct-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]no_clusters=4000_standardizeTrueecdfFalse50repetitionssave_inter", "rb") as input_file:
    n51_50 = cPickle.load(input_file)
      # -6 is privacy original and -7 is mspn privacy 
priv50= n50_50[-6][0:10]
priv51= n51_50[-7][0:10]

print(np.mean(priv50,(0,1)))
print(np.mean(priv50>0,(0,1)))
# [0.98748112 0.97731901 0.98100755 0.97506466 0.97376466 0.97767711
#  0.97174326 0.97413188 2.55036991]
# [1. 1. 1. 1. 1. 1. 1. 1. 1.]

print(np.mean(priv51,(0,1)))
print(np.mean(priv51>0,(0,1)))
# we hebben alleen de eerste drie??
# np.sum(priv50==0, (0,1))
#  array([10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000])

''' H1 normal has been overwritten ''' 
######################################################################################################################################
###############################     poisson (count)          ###################################################################
######################################################################################################################################

with open(r"H0poissonn1000mis999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse40_clusters=no_tests=1000_50priv_an_runs__50repetitions", "rb") as input_file:
    p30_40 = cPickle.load(input_file)
with open(r"H1poissonn1000mis999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse40_clusters=no_tests=1000_50priv_an_runs__50repetitions", "rb") as input_file:
    p31_40 = cPickle.load(input_file)
priv30= p30_40[-6]
priv31= p31_40[-6]

print(np.mean(priv30,(0,1)))
print(np.mean(priv30>0,(0,1)))
# [0.96400571 0.95940484 0.9586433  1.97575974 1.98027897 2.00047297
#  3.17392191 3.18147762 2.52370592]
# [0.99946 0.99974 0.99974 0.99992 0.99994 0.99992 0.99994 0.99998 1.     ]

print(np.mean(priv31,(0,1)))
print(np.mean(priv31>0,(0,1)))
# [0.95861219 0.96195769 0.96060835 1.97812559 1.97843555 1.99721033
#  3.17061977 3.17120658 2.53744737]
# [0.99958 0.99958 0.99964 0.99996 0.99992 0.99996 0.99988 0.99992 1.     ]
######################################################################################################################################
with open(r"H0poissonn10000mis9999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse400_clusters=no_tests=1000_50priv_an_runs__50repetitions", "rb") as input_file:
    p40_40 = cPickle.load(input_file)
with open(r"H1poissonn10000mis9999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse400_clusters=no_tests=1000_50priv_an_runs__50repetitions", "rb") as input_file:
    p41_40 = cPickle.load(input_file)
priv40= p40_40[-6]
priv41= p41_40[-6]

print(np.mean(priv40,(0,1)))
print(np.mean(priv40>0,(0,1)))
# [0.93496683 0.93286576 0.93344575 1.9346604  1.94189158 1.94645601
#  3.1111881  3.11236823 2.48110007]
# [0.99778 0.99796 0.99828 0.9999  0.99992 0.9999  0.99994 0.99996 1.     ]
print(np.mean(priv41,(0,1)))
print(np.mean(priv41>0,(0,1)))
# [0.93017458 0.93344656 0.93496515 1.93478792 1.94595205 1.95187142
#  3.11261402 3.11532313 2.49423626]
# [0.99822 0.99784 0.9982  0.99996 0.9999  0.99994 0.99988 0.99994 1.     ]
######################################################################################################################################
with open(r"H0poissonn100000mis99999kmeanscol_testrdct-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]no_clusters=4000_standardizeTrueecdfFalse50repetitionssave_inter", "rb") as input_file:
    p50_50 = cPickle.load(input_file)
with open(r"H1poissonn100000mis99999kmeanscol_testrdct-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]no_clusters=4000_standardizeTrueecdfFalse50repetitionssave_inter", "rb") as input_file:
    p51_50 = cPickle.load(input_file)
priv50= p50_50[-7][0:10]
priv51= p51_50[-7][0:10]

print(np.mean(priv50,(0,1)))
print(np.mean(priv50>0,(0,1)))
# [0.957969   0.94811741 0.94779625 1.97534226 1.95304158 1.98377116
#  3.15673768 3.15518802 2.55427866]
# [0.9954 0.9944 0.9936 0.9999 0.9999 0.9999 0.9999 0.9999 1.    ]
print(np.mean(priv51,(0,1)))
print(np.mean(priv51>0,(0,1)))
# [0.94683347 0.94693211 0.95797668 1.97568466 1.97079868 1.98759545
#  3.17195802 3.15962922 2.56467904]
# [0.9945 0.9944 0.9937 0.9999 0.9999 0.9999 0.9999 0.9999 1.    ]
######################################################################################################################################
###############################     categorical          ###################################################################
######################################################################################################################################

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
os.chdir(r"C:/Users/Shannon/Documents/Sanquin/Project 4/MSPN_privacy/results/privacy/mspn")
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
''' doen het allebei niet, alleen in andere repository'''
######################################################################################################################################
###############################     mixed          ###################################################################
######################################################################################################################################
with open(r"H0mixedn1000mis999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse40_clusters=no_tests=1000_50priv_an_runs__50repetitions", "rb") as input_file:
    m30_40 = cPickle.load(input_file)
with open(r"H1mixedn1000mis999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse40_clusters=no_tests=1000_50priv_an_runs__50repetitions", "rb") as input_file:
    m31_40 = cPickle.load(input_file)
priv30= m30_40[-6]
priv31= m31_40[-6]

print(np.mean(priv30,(0,1)))
print(np.mean(priv30>0,(0,1)))
# [11.31927141  2.41568975  8.95659229  0.43317037  0.52809     0.50228
#   0.52573429  0.51439     1.82649351]
# [1. 1. 1. 1. 1. 1. 1. 1. 1.]

print(np.mean(priv31,(0,1)))
print(np.mean(priv31>0,(0,1)))
# [11.24470809  2.4044516   8.90765708  0.43209832  0.5264      0.4975
#   0.52048571  0.51419     1.84993007]
# [1. 1. 1. 1. 1. 1. 1. 1. 1.]

######################################################################################################################################
with open(r"H0mixedn10000mis9999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse400_clusters=no_tests=1000_50priv_an_runs__50repetitions", "rb") as input_file:
    m40_40 = cPickle.load(input_file)
with open(r"H1mixedn10000mis9999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse400_clusters=no_tests=1000_50priv_an_runs__50repetitions", "rb") as input_file:
    m41_40 = cPickle.load(input_file)
priv40= m40_40[-6]
priv41= m41_40[-6]

print(np.mean(priv40,(0,1)))
print(np.mean(priv40>0,(0,1)))
# [10.49463895  2.23730384  5.9430075   0.34983068  0.52717     0.37218667
#   0.37879143  0.51718     1.67423091]
# [1. 1. 1. 1. 1. 1. 1. 1. 1.]

print(np.mean(priv41,(0,1)))
print(np.mean(priv41>0,(0,1)))
# [10.41180032  2.22901678  5.88760909  0.34945483  0.52446     0.3714
#   0.37414571  0.51759     1.68171094]
# [1. 1. 1. 1. 1. 1. 1. 1. 1.]
######################################################################################################################################
with open(r"H0mixedn100000mis99999kmeanscol_testrdct-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]no_clusters=4000_standardizeTrueecdfFalse50repetitionssave_inter", "rb") as input_file:
    m50_50 = cPickle.load(input_file)
with open(r"H1mixedn100000mis99999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse4000_clusters=no_tests=1000_50priv_an_runs__50repetitions", "rb") as input_file:
    m51_50 = cPickle.load(input_file)
priv50= m50_50[-7][0:10]
priv51= m51_50[-6][0:10]

print(np.mean(priv50,(0,1)))
print(np.mean(priv50>0,(0,1)))
# [10.64783192  2.32014168  4.31985309  0.29737125  0.53465     0.3689
#   0.22068571  0.5198      1.70164038]
# [1. 1. 1. 1. 1. 1. 1. 1. 1.]
print(np.mean(priv51,(0,1)))
print(np.mean(priv51>0,(0,1)))
# [10.56220771  2.31568686  4.26171476  0.29748889  0.52995     0.36966667
#   0.21708571  0.5194      1.70364557]
# [1. 1. 1. 1. 1. 1. 1. 1. 1.]
