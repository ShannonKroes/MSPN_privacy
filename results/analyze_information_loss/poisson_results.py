# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 00:08:47 2021

@author: Shannon
"""
import os
os.chdir(r"C:\Users\Shannon\Documents\Sanquin\Project 4")
from all_functions import *
from scipy import stats
from statsmodels.stats.power import tt_solve_power
import pickle as cPickle
os.chdir(r"C:\Users\Shannon\Documents\Sanquin\Project 4\Results\Final information loss results")
sim_n=Simulation(distr="poisson")
mean_diff= sim_n.true_param/20

''' final results'''

with open(r"H0poissonn1000mis999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse40_clusters=2500repetitions", "rb") as input_file:
    p30_40 = cPickle.load(input_file)
       

with open(r"H1poissonn1000mis999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse40_clusters=2500repetitions", "rb") as input_file:
    p31_40 = cPickle.load(input_file)

# How many repetitions do we need?
sd_diff=np.std(p30_40[2].reshape(p30_40[2].shape[0])-p30_40[3]) 
n_p30 = tt_solve_power(effect_size=mean_diff/sd_diff, alpha=0.05, power=0.8, alternative='two-sided')
# 82.65689873277464

sd_diff=np.std(p31_40[2].reshape(p31_40[2].shape[0])-p31_40[3]) 
n_p31 = tt_solve_power(effect_size=mean_diff/sd_diff, alpha=0.05, power=0.8, alternative='two-sided')
# 79.467423664519

# so lets do 200 for both
p3=200

tStat, pValue =  stats.ttest_rel(p30_40[2].reshape(p30_40[2].shape[0])[0:p3], p30_40[3][0:p3])
print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) 
#P-Value:1.3231789673913393e-08 T-Statistic:5.929574376505883

tStat, pValue =  stats.ttest_rel(p31_40[2].reshape(p31_40[2].shape[0])[0:p3], p31_40[3][0:p3])
print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) 
# P-Value:0.3043115273860878 T-Statistic:-1.029890485669211

####################################################################################################################################
# adding column name to the respective columns
df_p30_40.columns =['Anonymized data beta=0', 'Original data beta=0', 'Anonymized data beta=0.75', 'Original data beta=0.75']

ax= df_p30_40.plot.density(figsize = (4, 4),
                       linewidth = 2)
  
plt.xlabel("Regression parameter estimate")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

####################################################################################################################################

with open(r"H0poissonn10000mis9999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse400_clusters=150repetitions", "rb") as input_file:
    p40_40 = cPickle.load(input_file)

with open(r"H1poissonn10000mis9999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse400_clusters=150repetitions", "rb") as input_file:
    p41_40 = cPickle.load(input_file)
       
# How many repetitions do we need?
sd_diff=np.std(p40_40[2].reshape(p40_40[2].shape[0])-p40_40[3]) 
n_p40 = tt_solve_power(effect_size=mean_diff/sd_diff, alpha=0.05, power=0.8, alternative='two-sided')
# 7.378020853162903

sd_diff=np.std(p41_40[2].reshape(p41_40[2].shape[0])-p41_40[3]) 
n_p41 = tt_solve_power(effect_size=mean_diff/sd_diff, alpha=0.05, power=0.8, alternative='two-sided')
# 7.454649883859011

# so lets do 50 for both (just because 8 or 10 is so little and could seem like we never had enough power)
p4=50

tStat, pValue =  stats.ttest_rel(p40_40[2].reshape(p40_40[2].shape[0])[0:p4], p40_40[3][0:p4])
print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) 
#P-Value:0.7315106127270954 T-Statistic:0.34508074115905224

tStat, pValue =  stats.ttest_rel(p41_40[2].reshape(p41_40[2].shape[0])[0:p4], p41_40[3][0:p4])
print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) 
#P-Value:0.9434262315310125 T-Statistic:-0.07132903115443476

estimates_p40_40= np.vstack([p40_40[2].reshape(p40_40[2].shape[0]),p40_40[3],p41_40[2].reshape(p41_40[2].shape[0]),p41_40[3]]).T
df_p40_40 = pd.DataFrame(estimates_p40_40)

# adding column name to the respective columns
df_p40_40.columns =['Anonymized data beta=0', 'Original data beta=0', 'Anonymized data beta=0.75', 'Original data beta=0.75']

ax= df_p40_40.plot.density(figsize = (4, 4),
                       linewidth = 2)
  
plt.xlabel("Regression parameter estimate")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
####################################################################################################################################


with open(r"H0poissonn100000mis99999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse4000_clusters=100repetitions", "rb") as input_file:
    p50_40 = cPickle.load(input_file)
       
with open(r"H1poissonn100000mis99999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse4000_clusters=100repetitions", "rb") as input_file:
    p51_40 = cPickle.load(input_file)

# How many repetitions do we need?
sd_diff=np.std(p50_40[2].reshape(p50_40[2].shape[0])-p50_40[3]) 
n_p50 = tt_solve_power(effect_size=mean_diff/sd_diff, alpha=0.05, power=0.8, alternative='two-sided')
# 4.129638478544667

sd_diff=np.std(p51_40[2].reshape(p51_40[2].shape[0])-p51_40[3]) 
n_p51 = tt_solve_power(effect_size=mean_diff/sd_diff, alpha=0.05, power=0.8, alternative='two-sided')
# 4.1107793248203235

# so lets do 10 for both (although for 100 it is still both above .05)
p5=10

tStat, pValue =  stats.ttest_rel(p50_40[2].reshape(p50_40[2].shape[0])[0:p5], p50_40[3][0:p5])
print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) 
#P-Value:0.7273829383145324 T-Statistic:-0.35967816475818437

tStat, pValue =  stats.ttest_rel(p51_40[2].reshape(p51_40[2].shape[0])[0:p5], p51_40[3][0:p5])
print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) 
#P-Value:0.3255333495145463 T-Statistic:1.0398593943928893
