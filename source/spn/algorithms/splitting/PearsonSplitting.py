# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:16:23 2021

@author: Shannon
"""

import numpy as np
from spn.algorithms.splitting.Base import split_data_by_clusters
from networkx import from_numpy_matrix, connected_components



def clusters_by_p_value(adm, threshold, n_features):
    adm[adm > threshold] = 0

    adm[adm > 0] = 1

    result = np.zeros(n_features)
    for i, c in enumerate(connected_components(from_numpy_matrix(adm))):
        result[list(c)] = i + 1

    return result

def get_split_cols_linear(threshold=.05):
    def split_cols_linear(local_data, ds_context, scope):
        d=local_data.shape[1]
        from scipy.stats.stats import pearsonr
        pvalues_cor=np.zeros((d,d), dtype=float)
        for i in range(d):
            pvalues_cor_i=np.zeros((d), dtype=float)
            for j in range(d):
                pvalues_cor_i[j]= pearsonr(np.transpose(local_data)[i], np.transpose(local_data)[j])[1]
            pvalues_cor[i]=pvalues_cor_i[j]
        print(pvalues_cor)    
        clusters = clusters_by_p_value(pvalues_cor, threshold, local_data.shape[1])
            
        return split_data_by_clusters(local_data, clusters, scope, rows=False)
    return split_cols_linear



