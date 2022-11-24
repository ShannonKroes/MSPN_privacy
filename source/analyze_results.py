# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 16:47:10 2022

@author: Shannon
"""

def compute_CI(result):
    no_reps= result[2].shape[0]
    lower_bound= result[4]-2*np.std(result[2])/np.sqrt(no_reps)
    upper_bound= result[4]+2*np.std(result[2])/np.sqrt(no_reps)
    
    return [lower_bound, upper_bound]

def compute_CI_ans(bias, estimates, no_reps):
    
    lower_bound= bias-2*np.std(estimates)/np.sqrt(no_reps)
    upper_bound=bias+2*np.std(estimates)/np.sqrt(no_reps)
    
def paste_results(result):
    
    name=result[-1]
    information= "\nRMSE_an=" + str(result[0]) +"\n"+ "RMSE_or="+ str(result[1])  +"\n"+"bias_an="+str(result[4])+"\n"+"bias_or="+ str(result[5])+"\nempirical_SE_an="+str(np.std(result[2])) +"\nempirical_SE_or="+str( np.std(result[3]))+"\nmean_SE_an=" + str(result[6]) +"\n"+ "mean_SE_or="+ str(result[7]) +"\nmean_raw_diff="+str(result[10])+"\nabs_cor_diff="+str(result[11])+"\nuniv_prop_same="+str(result[12])+"\n"           
    mean_privacy_an = np.mean(result[13],(0,1))
    mean_privacy_or = np.mean(result[14],(0,1))
    prop_privacy_an = np.mean(result[13]>0,(0,1))
    prop_privacy_or = np.mean(result[14]>0,(0,1))
    privacy= "\nprivacy_or="+str( mean_privacy_or)+ "\nprivacy_an="+str( mean_privacy_an)+'\nprop_privacy_an'+str(prop_privacy_an)+'\nprop_privacy_or'+str(prop_privacy_or)
    result= name+" \ninformation loss\n"+information+privacy+"\n" +"\n" +"\n" +"\n" +"\n" +"\n" +"\n" +"\n" 

    return result  
      
def paste_results_original(result):
    
    name=result[-1]
    information= "\n"+ "RMSE_or="+ str(result[0])  +"\n"+"bias_or="+ str(result[2])+"\nempirical_SE_or="+str( np.std(result[1]))+"\n"+ "mean_SE_or="+ str(result[3]) +"\n"           
    mean_PPP_or= np.mean(result[5])
    mean_PPP_per_var_or=  np.mean(result[5], axis=(0,1))
    #See if ppp is above threshold 0
    PPP_p_or=  np.mean(result[5]>0, axis=(0,1))
    p_PPP_per_var_or=  np.mean(result[5]>0, axis=(0,1))
    mean_proximity_per_var_or=  np.mean(result[6], axis=(0,1))
    privacy=  "\np_PPP_per_var_or="+ str(p_PPP_per_var_or)+       "\nmean_proximity_per_var_or="+ str(mean_proximity_per_var_or)+       "\nmean_PPP_or="+str( mean_PPP_or)+      "\nPPP_p_or="+str( PPP_p_or)  
    result= name+" \ninformation loss\n"+information+"\nprivacy\n"+privacy+"\n" +"\n" +"\n" +"\n" +"\n" +"\n" +"\n" +"\n" 
    return result      
  
    
    
    
    
    
    
    