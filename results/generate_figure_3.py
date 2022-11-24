# -*- coding: utf-8 -*-
"""
Code to generate Figure 3.
"""
import os
import pickle as cPickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# set working directory
os.chdir(r"results/information_loss")
# load the results
with open(r"H0normaln1000mis999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse40_clusters=5500repetitions", "rb") as input_file:
    n30_40 = cPickle.load(input_file)
with open(r"H1normaln1000mis999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse40_clusters=5500repetitions", "rb") as input_file:
    n31_40 = cPickle.load(input_file)
with open(r"H0normaln10000mis9999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse400_clusters=no_tests=100_0priv_an_runs__100repetitions", "rb") as input_file:
    n40_40 = cPickle.load(input_file)
with open(r"H1normaln10000mis9999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse400_clusters=no_tests=100_0priv_an_runs__100repetitions", "rb") as input_file:
    n41_40 = cPickle.load(input_file)
with open(r"H0normaln100000mis99999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse4000_clusters=75repetitions", "rb") as input_file:
    n50_40 = cPickle.load(input_file)
with open(r"H1normaln100000mis99999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse4000_clusters=75repetitions", "rb") as input_file:
    n51_40 = cPickle.load(input_file)
with open(r"H0poissonn1000mis999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse40_clusters=2500repetitions", "rb") as input_file:
    p30_40 = cPickle.load(input_file)
with open(r"H1poissonn1000mis999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse40_clusters=2500repetitions", "rb") as input_file:
    p31_40 = cPickle.load(input_file)
with open(r"H0poissonn10000mis9999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse400_clusters=150repetitions", "rb") as input_file:
    p40_40 = cPickle.load(input_file)
with open(r"H1poissonn10000mis9999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse400_clusters=150repetitions", "rb") as input_file:
    p41_40 = cPickle.load(input_file)
with open(r"H0poissonn100000mis99999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse4000_clusters=100repetitions", "rb") as input_file:
    p50_40 = cPickle.load(input_file)
with open(r"H1poissonn100000mis99999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse4000_clusters=100repetitions", "rb") as input_file:
    p51_40 = cPickle.load(input_file)
with open(r"H0categoricaln1000mis999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse40_clusters=150repetitions", "rb") as input_file:
    c30_40 = cPickle.load(input_file)
with open(r"H1categoricaln1000mis999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse40_clusters=150repetitions", "rb") as input_file:
    c31_40 = cPickle.load(input_file)
with open(r"H0categoricaln10000mis9999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse400_clusters=35repetitions", "rb") as input_file:
    c40_40 = cPickle.load(input_file)
with open(r"H1categoricaln10000mis9999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse400_clusters=35repetitions", "rb") as input_file:
    c41_40 = cPickle.load(input_file)
with open(r"H0categoricaln100000mis99999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse4000_clusters=150repetitions", "rb") as input_file:
    c50_40 = cPickle.load(input_file)
with open(r"H1categoricaln100000mis99999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse4000_clusters=150repetitions", "rb") as input_file:
    c51_40 = cPickle.load(input_file)
with open(r"H0mixedn1000mis999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse40_clusters=4000repetitions", "rb") as input_file:
    m30_40 = cPickle.load(input_file)
with open(r"H1mixedn1000mis999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse40_clusters=4000repetitions", "rb") as input_file:
    m31_40 = cPickle.load(input_file)
with open(r"H0mixedn10000mis9999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse400_clusters=150repetitions", "rb") as input_file:
    m40_40 = cPickle.load(input_file)
with open(r"H1mixedn10000mis9999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse400_clusters=150repetitions", "rb") as input_file:
    m41_40 = cPickle.load(input_file)
with open(r"H0mixedn100000mis99999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse4000_clusters=150repetitions", "rb") as input_file:
    m50_40 = cPickle.load(input_file)
with open(r"H1mixedn100000mis99999kmeanst-1pois_discretecol_testrdchist_sourcenumpyan_sample_size[100000]_standardizeTrueecdfFalse4000_clusters=150repetitions", "rb") as input_file:
    m51_40 = cPickle.load(input_file)
#########################################################################################
# function to put results into data frame
def get_df(res0, res1):
    estimates = np.vstack([res0[2].reshape(
        res0[2].shape[0]), res0[3], res1[2].reshape(res1[2].shape[0]), res1[3]]).T
    df_res = pd.DataFrame(estimates)
    df_res.columns = ['Anonymized data beta=0', 'Original data beta=0',
                      'Anonymized data beta!=0', 'Original data beta!=0']
    return df_res
# create data frames
df_n30_40 = get_df(n30_40, n31_40)
df_n40_40 = get_df(n40_40, n41_40)
df_n50_40 = get_df(n50_40, n51_40)

df_m30_40 = get_df(m30_40, m31_40)
df_m40_40 = get_df(m40_40, m41_40)
df_m50_40 = get_df(m50_40, m51_40)

df_p30_40 = get_df(p30_40, p31_40)
df_p40_40 = get_df(p40_40, p41_40)
df_p50_40 = get_df(p50_40, p51_40)

df_c30_40 = get_df(c30_40, c31_40)
df_c40_40 = get_df(c40_40, c41_40)
df_c50_40 = get_df(c50_40, c51_40)
#########################################################################################
# create plot
fig, axes = plt.subplots(4, 3, figsize=(8.27, 11.69))
df_n30_40.plot.density(linewidth=1, legend=False, ax=axes[0, 0], style=['-','--',':', '-.'])
df_n40_40.plot.density(linewidth=1, legend=False, ax=axes[0, 1], style=['-','--',':', '-.'])
df_n50_40.plot.density(linewidth=1, legend=False, ax=axes[0, 2], style=['-','--',':', '-.'])
axes[0, 0].set_ylabel("Density normal data")
axes[0, 2].set_ylabel("")
axes[0, 1].set_ylabel("")

df_p30_40.plot.density(linewidth=1, legend=False, ax=axes[1, 0], style=['-','--',':', '-.'])
df_p40_40.plot.density(linewidth=1, legend=False, ax=axes[1, 1], style=['-','--',':', '-.'])
df_p50_40.plot.density(linewidth=1, legend=False, ax=axes[1, 2], style=['-','--',':', '-.'])
axes[1, 0].set_ylabel("Density count data")
axes[1, 1].set_ylabel("")
axes[1, 2].set_ylabel("")

df_c30_40.plot.density(linewidth=1, legend=False, ax=axes[2, 0], style=['-','--',':', '-.'])
df_c40_40.plot.density(linewidth=1, legend=False, ax=axes[2, 1], style=['-','--',':', '-.'])
df_c50_40.plot.density(linewidth=1, legend=False, ax=axes[2, 2], style=['-','--',':', '-.'])
axes[2, 0].set_ylabel("Density categorical data")
axes[2, 1].set_ylabel("")
axes[2, 2].set_ylabel("")

df_m30_40.plot.density(linewidth=1, legend=False, ax=axes[3, 0], style=['-','--',':', '-.'])
df_m40_40.plot.density(linewidth=1, legend=False, ax=axes[3, 1], style=['-','--',':', '-.'])
df_m50_40.plot.density(linewidth=1, legend=False, ax=axes[3, 2], style=['-','--',':', '-.'])
axes[3, 0].set_ylabel("Density mixed data")
axes[3, 1].set_ylabel("")
axes[3, 2].set_ylabel("")
axes[0, 0].set_title("n=1,000")
axes[0, 1].set_title("n=10,000")
axes[0, 2].set_title("n=100,000")

lines, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(lines, labels, loc = 'lower center', ncol=2)
plt.show()