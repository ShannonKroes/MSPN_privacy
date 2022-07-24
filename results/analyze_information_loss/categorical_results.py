# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 12:04:28 2021

@author: Shannon
"""

import os
os.chdir(r"C:\Users\Shannon\Documents\Sanquin\Project 4")
from all_functions import *
from scipy import stats
from statsmodels.stats.power import tt_solve_power
import pickle as cPickle
os.chdir(r"C:\Users\Shannon\Documents\Sanquin\Project 4\Results\Final information loss results")
sim_n=Simulation(distr="categorical")
mean_diff= sim_n.true_param/20



''' final results'''

#########################################################################################################################################################################################################
with open(r"H0categoricaln1000mis999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse40_clusters=150repetitions", "rb") as input_file:
    c30_40 = cPickle.load(input_file)
       

with open(r"H1categoricaln1000mis999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse40_clusters=150repetitions", "rb") as input_file:
    c31_40 = cPickle.load(input_file)
  
#########################################################################################################################################################################################################

# How many repetitions do we need?
sd_diff=np.std(c30_40[2].reshape(c30_40[2].shape[0])-c30_40[3]) 
n_c30 = tt_solve_power(effect_size=mean_diff/sd_diff, alpha=0.05, power=0.8, alternative='two-sided')
# 6.287451437261245

sd_diff=np.std(c31_40[2].reshape(c31_40[2].shape[0])-c31_40[3]) 
n_c31 = tt_solve_power(effect_size=mean_diff/sd_diff, alpha=0.05, power=0.8, alternative='two-sided')
# 6.627191709048656
# so lets do 50 for both
c3=50

tStat, pValue =  stats.ttest_rel(c30_40[2].reshape(c30_40[2].shape[0])[0:c3], c30_40[3][0:c3])
print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) 
#P-Value:0.06301271462621821 T-Statistic:1.9023387450185918

tStat, pValue =  stats.ttest_rel(c31_40[2].reshape(c31_40[2].shape[0])[0:c3], c31_40[3][0:c3])
print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) 
# P-Value:0.012414005876348649 T-Statistic:2.5959664873518165

#########################################################################################################################################################################################################

estimates_c30_40= np.vstack([c30_40[2].reshape(c30_40[2].shape[0]),c30_40[3],c31_40[2].reshape(c31_40[2].shape[0]),c31_40[3]]).T
df_c30_40 = pd.DataFrame(estimates_c30_40)


# adding column name to the respective columns
df_c30_40.columns =['Anonymized data beta=0', 'Original data beta=0', 'Anonymized data beta=0.75', 'Original data beta=0.75']

ax= df_c30_40.plot.density(figsize = (4, 4),
                       linewidth = 2)
  
plt.xlabel("Regression parameter estimate")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


#########################################################################################################################################################################################################

with open(r"H0categoricaln10000mis9999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse400_clusters=35repetitions", "rb") as input_file:
    c40_40 = cPickle.load(input_file)
       

with open(r"H1categoricaln10000mis9999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse400_clusters=35repetitions", "rb") as input_file:
    c41_40 = cPickle.load(input_file)
       
sd_diff=np.std(c40_40[2].reshape(c40_40[2].shape[0])-c40_40[3]) 
n_c40 = tt_solve_power(effect_size=mean_diff/sd_diff, alpha=0.05, power=0.8, alternative='two-sided')
# 2.777014014286742

sd_diff=np.std(c41_40[2].reshape(c41_40[2].shape[0])-c41_40[3]) 
n_c41 = tt_solve_power(effect_size=mean_diff/sd_diff, alpha=0.05, power=0.8, alternative='two-sided')
# 2.7240318049949894

# so lets do 50 for both
c4=10

tStat, pValue =  stats.ttest_rel(c40_40[2].reshape(c40_40[2].shape[0])[0:c4], c40_40[3][0:c4])
print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) 
#P-Value:0.9029575921248068 T-Statistic:0.12540749213326213

tStat, pValue =  stats.ttest_rel(c41_40[2].reshape(c41_40[2].shape[0])[0:c4], c41_40[3][0:c4])
print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) 
#P-Value:0.8053232945567665 T-Statistic:-0.2538398045419506

# for 35:
# P-Value:0.6428546097800429 T-Statistic:0.46788164970130525
# P-Value:0.8313671056202028 T-Statistic:0.2145923034681648

#########################################################################################################################################################################################################

estimates_c40_40= np.vstack([c40_40[2].reshape(c40_40[2].shape[0]),c40_40[3],c41_40[2].reshape(c41_40[2].shape[0]),c41_40[3]]).T
df_c40_40 = pd.DataFrame(estimates_c40_40)


# adding column name to the respective columns
df_c40_40.columns =['Anonymized data beta=0', 'Original data beta=0', 'Anonymized data beta=0.75', 'Original data beta=0.75']

ax= df_c40_40.plot.density(figsize = (4, 4),
                       linewidth = 2)
  
plt.xlabel("Regression parameter estimate")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#########################################################################################################################################################################################################
#########################################################################################################################################################################################################

with open(r"H0categoricaln100000mis99999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse4000_clusters=150repetitions", "rb") as input_file:
    c50_40 = cPickle.load(input_file)
       

with open(r"H1categoricaln100000mis99999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse4000_clusters=150repetitions", "rb") as input_file:
    c51_40 = cPickle.load(input_file)
       
       
sd_diff=np.std(c50_40[2].reshape(c50_40[2].shape[0])-c50_40[3]) 
n_c50 = tt_solve_power(effect_size=mean_diff/sd_diff, alpha=0.05, power=0.8, alternative='two-sided')
# 2.753170641252558
sd_diff=np.std(c51_40[2].reshape(c51_40[2].shape[0])-c51_40[3]) 
n_c51 = tt_solve_power(effect_size=mean_diff/sd_diff, alpha=0.05, power=0.8, alternative='two-sided')
#  2.636155058456351

# so lets do 10 for both
c5=10

tStat, pValue =  stats.ttest_rel(c50_40[2].reshape(c50_40[2].shape[0])[0:c5], c50_40[3][0:c5])
print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) 
#P-Value:0.8581681716903832 T-Statistic:0.18390053202017467

tStat, pValue =  stats.ttest_rel(c51_40[2].reshape(c51_40[2].shape[0])[0:c5], c51_40[3][0:c5])
print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) 
# P-Value:0.5905655316432963 T-Statistic:-0.5578276322790837

# for 150
# P-Value:0.9143936637169521 T-Statistic:-0.10768092285999688
# P-Value:0.20862427839697575 T-Statistic:-1.2628263722803825
#########################################################################################################################################################################################################

estimates_c50_40= np.vstack([c50_40[2].reshape(c50_40[2].shape[0]),c50_40[3],c51_40[2].reshape(c51_40[2].shape[0]),c51_40[3]]).T
df_c50_40 = pd.DataFrame(estimates_c50_40)


# adding column name to the respective columns
df_c50_40.columns =['Anonymized data beta=0', 'Original data beta=0', 'Anonymized data beta=0.75', 'Original data beta=0.75']

ax= df_c50_40.plot.density(figsize = (4, 4),linewidth = 2, legend=False)


  
plt.xlabel("Regression parameter estimate")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#########################################################################################################################################################################################################


            


