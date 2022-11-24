# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 16:44:32 2022

@author: Shannon
"""

def extract_levels(data):
    levels=np.zeros(data.shape[1])
    
    for i in range(data.shape[1]):
        levels[i]= len(np.unique(data.T[i]))
    return levels

def save_object(obj, filename):
    with open(filename, 'wb') as output: 
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
def cor_matrix(data):
    d=data.shape[1]
    cors=np.zeros((d,d), dtype=float)
    for i in range(d):
        cor_i=np.zeros((d), dtype=float)
        for j in range(d):
            cor_i[j]= pearsonr(np.transpose(data)[i], np.transpose(data)[j])[0]
        cors[i]=cor_i
        print(cors)
        
    return cors

def get_ks_p_values(data, AN):
    d=data.shape[1]
    ks_p_values= np.zeros((d), dtype=float)
    for i in range(d):
        ks_p_values[i]=ks_2samp(np.transpose(AN)[i], np.transpose(data)[i]).pvalue
    return ks_p_values

def compute_freqs(variable):
    df= pd.DataFrame(variable, columns = ['variable'])
    freq= np.array(df['variable'].value_counts())
    return freq

def get_chi_p_values(data, AN):
    d= data.shape[1]
    chi_p_values= np.zeros((d), dtype=float)
    AN=np.round(AN)
    data= np.round(data)
       
    for i in range(d):
        unique_AN= np.unique(AN.T[i])
        unique_data= np.unique(data.T[i])
        levels= np.union1d(unique_AN, unique_data)
        no_levels= levels.shape[0]
        an_ind= np.zeros((no_levels))
        or_ind= np.zeros((no_levels))
        
        for l in range(levels.shape[0]):
            an_ind[l]= np.any(levels[l]==unique_AN)
            or_ind[l]= np.any(levels[l]==unique_data)

        AN_freq_i= compute_freqs(AN.T[i])
        
        AN_freq_full= np.zeros((no_levels))
        AN_freq_full[an_ind==1]= AN_freq_i
        data_freq_i= compute_freqs(data.T[i])
        data_freq_full= np.zeros((no_levels))
        data_freq_full[or_ind==1]= data_freq_i

        chi_p_values[i]=chi2_contingency(np.array([AN_freq_full,data_freq_full]))[1]
        
    return chi_p_values

def compute_CI_or(result):
    no_reps= result[3].shape[0]
    lower_bound= np.mean(result[3])-2*np.std(result[3])/np.sqrt(no_reps)
    upper_bound= np.mean(result[3])+2*np.std(result[3])/np.sqrt(no_reps)
    return [lower_bound, upper_bound]