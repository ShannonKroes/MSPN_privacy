"""
In this file we extract the privacy results for mixed data.
"""
import os
import pickle as cPickle
os.chdir(r"C:\Users\Shannon\Documents\Sanquin\Project 4\Results\Final privacy results")

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
with open(r"H0mixedn100000mis99999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse4000_clusters=no_tests=1000_50priv_an_runs__50repetitions", "rb") as input_file:
    m50_50 = cPickle.load(input_file)
with open(r"H1mixedn100000mis99999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse4000_clusters=no_tests=1000_50priv_an_runs__50repetitions", "rb") as input_file:
    m51_50 = cPickle.load(input_file)
priv50= m50_50[-7][0:10]
priv51= m51_50[-6][0:10]

print(np.mean(priv50,(0,1)))
print(np.mean(priv50>0,(0,1)))

print(np.mean(priv51,(0,1)))
print(np.mean(priv51>0,(0,1)))
# [10.56220771  2.31568686  4.26171476  0.29748889  0.52995     0.36966667
#   0.21708571  0.5194      1.70364557]
# [1. 1. 1. 1. 1. 1. 1. 1. 1.]
