"""
In this file we extract the privacy results for data with count-valued predictors. 
"""
import os
import pickle as cPickle
os.chdir(r"C:\Users\Shannon\Documents\Sanquin\Project 4\Results\Final privacy results")
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

print(np.mean(priv41,(0,1)))
print(np.mean(priv41>0,(0,1)))
# [0.93017458 0.93344656 0.93496515 1.93478792 1.94595205 1.95187142
#  3.11261402 3.11532313 2.49423626]
# [0.99822 0.99784 0.9982  0.99996 0.9999  0.99994 0.99988 0.99994 1.     ]
######################################################################################################################################
with open(r"H0poissonn100000mis99999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse4000_clusters=no_tests=1000_50priv_an_runs__50repetitions", "rb") as input_file:
    p50_50 = cPickle.load(input_file)
with open(r"H1poissonn100000mis99999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse4000_clusters=no_tests=1000_50priv_an_runs__50repetitions", "rb") as input_file:
    p51_50 = cPickle.load(input_file)
priv50= p50_50[-7][0:10]
priv51= p51_50[-7][0:10]

print(np.mean(priv50,(0,1)))
print(np.mean(priv50>0,(0,1)))


print(np.mean(priv51,(0,1)))
print(np.mean(priv51>0,(0,1)))


