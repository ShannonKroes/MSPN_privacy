# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 16:07:06 2022

@author: kroessks
"""
import numpy as np
which = lambda lst:list(np.where(lst)[0])
from mpyc.runtime import mpc  


def vertical_kmeans(data, k, reps = 2, tol = .01, inits = 1, random_state = 190194):
    # ! to do: tol is 5% instead of 5 (could call this tol to be in line with kmeans)
    # repeat this with multiple seeds
    '''
    Function to perform k-means on vertically partitioned data using Multi-Party Computation with mpyc. 

    Parameters
    ----------
    data : list
        List of vertically partitioned data. For example, if we have three parties (p1, p2 and p2) and
        party 1 has variables X1 and X2, party 2 has variable X3 and party 3 has variable X4, we express a toy data set
        with 3 records (e.g. patients/people) as follows:
            
        p1 p1 p2 p3
        -----------
        X1 X2 X3 X4
        1  2  3  4
        2  4  6  8
        0  1  1  0
        
        In this case, the data list should have the form: [np.array([1,2,0], [2,4,1]),np.array( [3,6,1]),np.array( [4,8,0])].
        
    k : int
        Desired number of clusters.
    reps : int
        Number of iterations.
    tol : float
        Percentage of people that should change cluster in order to continue k-means algorithm (re-assigning people to different clusters).
    inits : int
        Number of times the algorithm is run with different initial centroids. 
    Returns
    -------
    clusters : np.array(int)
        np.array with one integer for every individual that indicates to which of the k clusters they have been assigned. 

    '''
    # First every party chooses the centroid value pertaining to their variables.
    # In the current these centroids are chosen at random.
    # then every party computes the distance from these chosen centroids
    no_parties = len(data)
    no_feat_party = np.zeros(no_parties, dtype=int)
    initial_centroids = []
    n = data[0].shape[0]
    for s in range(inits):
        np.random.seed(int(random_state+s))
        for i, dat in enumerate(data):
            no_feat_party[i] = dat.shape[1]        
    
            if no_feat_party[i]>1:
                centroids_party = np.zeros((no_feat_party[i], k))
                for j in range(no_feat_party[i]): 
                    centroids_party[j] = np.random.choice(dat.T[j], k, replace = False)
            else: 
                centroids_party = np.random.choice(dat.reshape(-1), k, replace = False)
            initial_centroids.append(centroids_party)
        clusters_prev = np.zeros(n)
        clusters = np.ones(n)
        # then loop according to number of repetitions.  
        for r in range(reps):
            sum_diff = (n-np.sum(clusters==clusters_prev))/n # percentage of people that changed cluster
            if sum_diff>tol:
                # we compute the distance for every person
                clusters_prev = clusters
                if r == 0:  
                    centroids = initial_centroids 
                distances = distance_from_centroids(data, centroids, no_feat_party, k, n)  
                clusters_raw =  mpc.run(mpc_distance_combine(distances))
                
                # when constructing the number of clusters, it is possible that some clusters lose all its participants
                # or never even gain any. This results in fewer than k clusters. For now we accept this:
                print(np.unique(clusters_raw, return_counts=True))
                unique_clusters = np.unique(clusters_raw)
                k = len(unique_clusters)
                dict_clusters = {}
                for A, B in zip( unique_clusters, np.arange(k)):
                    dict_clusters[A] = B
                clusters = np.zeros(n)    
                for i in range(n):
                    clusters[i] = dict_clusters[clusters_raw[i]]
                print(np.unique(clusters, return_counts=True))
                
                if r < (reps):
                # recompute the centroids based on the current clustering  
                    centroids = []
                    for p, dat in enumerate(data): 
                        if no_feat_party[p]>1:
                            centroids_p = np.zeros((no_feat_party[p], k))
                            for j in range(no_feat_party[p]): 
                                for c in range(k):
                                    centroids_p[j][c] = np.mean(dat.T[j][clusters==c])
                        if no_feat_party[p]==1:
                            centroids_p = np.zeros(k)
                            for c in range(k):
                                centroids_p[c] = np.mean(dat[clusters==c])
                        centroids.append(centroids_p)
    return clusters 


def distance_from_centroids_list(data, centroids, no_feat_party, k): 
    distances = []# all the distances for all the variables for all parties for all k 
    for c in range(k): 
        distances_i = [] # all the distances for all the variables for all parties
        for i, dat in enumerate(data):
            distances_j = [] # all the distances for all the variables for one party
            if no_feat_party[i]>1:
                for j, dat_j in enumerate(dat):
                        distances_j.append(np.absolute(dat_j - centroids[i][j][c]))
            if no_feat_party[i]==1:
                distances_j.append(np.absolute(dat - centroids[i][0][c]))
            distances_i.append(distances_j)
        distances.append(distances_i)   
        
    # compute distances per party
    
    return distances

def distance_from_centroids(data, centroids, no_feat_party, k, n):  
    distances = np.zeros((k, len(data), n))  
    for c in range(k): 
        for p, dat in enumerate(data): 
            distances_p = np.zeros((no_feat_party[p], n))             
            if no_feat_party[p]>1:  
                for j, dat_j in enumerate(dat.T):  
                    distances_p[j] = np.absolute(dat_j - centroids[p][j][c])  
            distances[c,p,:] = np.sum(distances_p,0)  
            if no_feat_party[p]==1:  
                distances[c,p,:] = np.absolute(dat - centroids[p][c]).reshape(-1) 
      
    return distances  
                    
 
async def mpc_distance_combine(distances): 
    k = len(distances)  
    no_parties, n = distances[0].shape  
    clusters = np.zeros(n)  
    distances_int = np.round(distances*100,0) 
    await mpc.start() 
    for i in range(n):  
        sums = np.zeros(k)  
        for c in range(k):  
            # for every person and for every centroid  
    
            secret_scores = []  
            for p in range(no_parties):  
                secret_scores.append(mpc.SecInt(16)(int(distances_int[c][p][i])))      
            all_scores = mpc.input(secret_scores, senders = int(0))   
            total_score = sum(all_scores)  
            sums[c] = await mpc.output(total_score)  
 
        clusters[i] = which(sums== np.min(sums))[0]  
    await mpc.shutdown() 

    return clusters    


        
        
        
        
        
# vertical_kmeans(data, k=5)
# np.absolute(np.sum(clusters-clusters_prev))
# also this??
# packages\IPython\core\compilerop.py:101: RuntimeWarning: coroutine 'mpc_distance_combine' was never awaited
#   return compile(source, filename, symbol, self.flags | PyCF_ONLY_AST, 1)




