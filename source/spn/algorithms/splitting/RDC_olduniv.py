"""
Created on March 20, 2018

@author: Alejandro Molina
"""

import numpy as np
from sklearn.cluster import KMeans

from spn.algorithms.splitting.Base import split_data_by_clusters, clusters_by_adjacency_matrix, preproc
import logging
from spn.algorithms.splitting.Clustering import get_split_rows_KMeans
from scipy.stats.stats import pearsonr

logger = logging.getLogger(__name__)
_rpy_initialized = False


def init_rpy():
    global _rpy_initialized
    if _rpy_initialized:
        return
    _rpy_initialized = True

    from rpy2 import robjects
    from rpy2.robjects import numpy2ri
    import os

    path = os.path.dirname(__file__)

    robjects.r("options(warn=-1)")

    with open(path + "/rdc.R", "r") as rfile:
        code = "".join(rfile.readlines())
        robjects.r(code)

    numpy2ri.activate()


def get_RDC_transform(data, meta_types, ohe=False, k=10, s=1 / 6):
    from rpy2 import robjects

    init_rpy()

    assert data.shape[1] == len(meta_types), "invalid parameters"

    r_meta_types = [mt.name.lower() for mt in meta_types]

    try:
        df = robjects.r["as.data.frame"](data)
        out = robjects.r["transformRDC"](df, ohe, r_meta_types, k, s)
        out = np.asarray(out)
    except Exception as e:
        np.savetxt("/tmp/errordata.txt", data)
        logger.info(e)
        raise e

    return out


def get_RDC_adjacency_matrix(data, meta_types, ohe=False, linear=True, test="rdc"):
    from rpy2 import robjects

    init_rpy()

    assert data.shape[1] == len(meta_types), "invalid parameters"

    r_meta_types = [mt.name.lower() for mt in meta_types]

    try:
        df = robjects.r["as.data.frame"](data)
        out = robjects.r["testRDC"](df, ohe, r_meta_types, linear, test)
        out = np.asarray(out)
    except Exception as e:
        np.savetxt("/tmp/errordata.txt", data)
        logger.info(e)
        raise e
    return out


def get_split_cols_RDC(threshold=0.3, ohe=True, linear=True):
    def split_cols_RDC(local_data, ds_context, scope):
        adjm = get_RDC_adjacency_matrix(local_data, ds_context.get_meta_types_by_scope(scope), ohe, linear)

        clusters = clusters_by_adjacency_matrix(adjm, threshold, local_data.shape[1])

        return split_data_by_clusters(local_data, clusters, scope, rows=False)

    return split_cols_RDC


def get_split_rows_RDC(n_clusters=2, k=10, s=1 / 6, ohe=True, seed=17):
    def split_rows_RDC(local_data, ds_context, scope):
        data = get_RDC_transform(local_data, ds_context.get_meta_types_by_scope(scope), ohe, k=k, s=s)

        clusters = KMeans(n_clusters=n_clusters, random_state=seed, n_jobs=1).fit_predict(data)

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_RDC


############################################################################################
#
# Python version
#
import itertools

from networkx.algorithms.components.connected import connected_components
from networkx.convert_matrix import from_numpy_matrix
import scipy.stats

from sklearn.cross_decomposition import CCA
from spn.structure.StatisticalTypes import MetaType

CCA_MAX_ITER = 100


def ecdf(X):
    """
    Empirical cumulative distribution function
    for data X (one dimensional, if not it is linearized first)
    """
    # return scipy.stats.rankdata(X, method='max') / len(X)

    mv_ids = np.isnan(X)

    N = X.shape[0]
    X = X[~mv_ids]
    R = scipy.stats.rankdata(X, method="max") / len(X)
    X_r = np.zeros(N)
    X_r[~mv_ids] = R
    return X_r


def empirical_copula_transformation(data):
    ones_column = np.ones((data.shape[0], 1))
    data = np.concatenate((np.apply_along_axis(ecdf, 0, data), ones_column), axis=1)
    return data


def make_matrix(data):
    """
    Ensures data to be 2-dimensional
    """
    if data.ndim == 1:
        data = data[:, np.newaxis]
    else:
        assert data.ndim == 2, "Data must be 2 dimensional {}".format(data.shape)

    return data


def ohe_data(data, domain):
    dataenc = np.zeros((data.shape[0], len(domain)))

    dataenc[data[:, None] == domain[None, :]] = 1

    #
    # this control fails when having missing data as nans
    if not np.any(np.isnan(data)):
        assert np.all((np.nansum(dataenc, axis=1) == 1)), "one hot encoding bug {} {} {}".format(
            domain, data, np.nansum(dataenc, axis=1)
        )

    return dataenc


def rdc_transformer(
    local_data,
    meta_types,
    domains,
    k=None,
    s=1.0 / 6.0,
    non_linearity=np.sin,
    return_matrix=False,
    ohe=True,
    rand_gen=None,
):
    # logger.info('rdc transformer', k, s, non_linearity)
    """
    Given a data_slice,
    return a transformation of the features data in it according to the rdc
    pipeline:
    1 - empirical copula transformation
    2 - random projection into a k-dimensional gaussian space
    3 - pointwise  non-linear transform
    """

    N, D = local_data.shape

    if rand_gen is None:
        rand_gen = np.random.RandomState(17)

    #
    # precomputing transformations to reduce time complexity
    #

    #
    # FORCING ohe on all discrete features
    features = []
    for f in range(D):
        if meta_types[f] == MetaType.DISCRETE:
            features.append(ohe_data(local_data[:, f], domains[f]))
        else:
            features.append(local_data[:, f].reshape(-1, 1))
    # else:
    #     features = [data_slice.getFeatureData(f) for f in range(D)]

    #
    # NOTE: here we are setting a global k for ALL features
    # to be able to precompute gaussians
    if k is None:
        feature_shapes = [f.shape[1] if len(f.shape) > 1 else 1 for f in features]
        k = max(feature_shapes) + 1

    #
    # forcing two columness
    features = [make_matrix(f) for f in features]

    #
    # transform through the empirical copula
    features = [empirical_copula_transformation(f) for f in features]

    #
    # substituting nans with zero (the above step should have taken care of that)
    features = [np.nan_to_num(f) for f in features]

    #
    # random projection through a gaussian
    random_gaussians = [rand_gen.normal(size=(f.shape[1], k)) for f in features]

    rand_proj_features = [s / f.shape[1] * np.dot(f, N) for f, N in zip(features, random_gaussians)]

    nl_rand_proj_features = [non_linearity(f) for f in rand_proj_features]

    #
    # apply non-linearity
    if return_matrix:
        return np.concatenate(nl_rand_proj_features, axis=1)

    else:
        return [np.concatenate((f, np.ones((f.shape[0], 1))), axis=1) for f in nl_rand_proj_features]


def rdc_cca(indexes):
    
    i, j, rdc_features = indexes
    cca = CCA(n_components=1, max_iter=CCA_MAX_ITER)
    X_cca, Y_cca = cca.fit_transform(rdc_features[i], rdc_features[j])
    rdc = np.corrcoef(X_cca.T, Y_cca.T)[0, 1]

    #p_value = pearsonr(np.array(X_cca), np.array(Y_cca))[1] 
    
    # logger.info(i, j, rdc)
    #return [rdc, p_value]
    return rdc


def rdc_test(local_data, meta_types, domains, k=None, s=1.0 / 6.0, non_linearity=np.sin, n_jobs=-1, rand_gen=None):
    n_features = local_data.shape[1]

    rdc_features = rdc_transformer(
        local_data, meta_types, domains, k=k, s=s, non_linearity=non_linearity, return_matrix=False, rand_gen=rand_gen
    )

    pairwise_comparisons = list(itertools.combinations(np.arange(n_features), 2))

    from joblib import Parallel, delayed

    rdc_vals = Parallel(n_jobs=n_jobs, max_nbytes=1024, backend="threading")(
        delayed(rdc_cca)((i, j, rdc_features)) for i, j in pairwise_comparisons
    )

    
    rdc_adjacency_matrix = np.zeros((n_features, n_features))

    p_values = np.zeros((n_features, n_features))
    # this is jus tfilling the matrix with the values that have already been produced
    for (i, j), rdc in zip(pairwise_comparisons, rdc_vals):

        rdc_adjacency_matrix[i, j] = rdc
        rdc_adjacency_matrix[j, i] = rdc
    #
    # setting diagonal to 1
    rdc_adjacency_matrix[np.diag_indices_from(rdc_adjacency_matrix)] = 1

    return [rdc_adjacency_matrix, p_values]


def getIndependentRDCGroups_py(
    local_data, threshold, meta_types, domains, k=None, s=1.0 / 6.0, non_linearity=np.sin, n_jobs=-2, rand_gen=None, test="rdc"
):
    rdc_adjacency_matrix, p_values = rdc_test(
        local_data, meta_types, domains, k=k, s=s, non_linearity=non_linearity, n_jobs=n_jobs, rand_gen=rand_gen
    )

    #
    # Why is this necessary?
    #
    
    # we print to see nans

          
    # where do these nans come from?
    rdc_adjacency_matrix[np.isnan(rdc_adjacency_matrix)] = 0
    # added by kroes et al.:
    rdc_adjacency_matrix[np.corrcoef(local_data.T)>rdc_adjacency_matrix]=np.corrcoef(local_data.T)[np.corrcoef(local_data.T)>rdc_adjacency_matrix]
    
    n_features = local_data.shape[1]
 

    
    #
    
    # thresholding 
    if test== "rdc":
        rdc_adjacency_matrix[rdc_adjacency_matrix < threshold] = 0
    # if p_value, then we deem groups to be independent with a p value over .05
    if test=="p_value":
        rdc_adjacency_matrix[p_values > threshold] = 0
        rdc_adjacency_matrix[np.diag_indices_from(rdc_adjacency_matrix)] = 1

    # logger.info("thresholding %s", rdc_adjacency_matrix)

    #
    # getting connected components
    

    
    result = np.zeros(n_features)
    for i, c in enumerate(connected_components(from_numpy_matrix(rdc_adjacency_matrix))):
        result[list(c)] = i + 1

    
    
    return result


def get_split_cols_RDC_py(threshold=0.3, ohe=True, k=10, s=1 / 6, non_linearity=np.sin, n_jobs=-2, rand_gen=None, test="rdc"):
    def split_cols_RDC_py(local_data, ds_context, scope):
        meta_types = ds_context.get_meta_types_by_scope(scope)
        domains = ds_context.get_domains_by_scope(scope)

        clusters = getIndependentRDCGroups_py(
            local_data,
            threshold,
            meta_types,
            domains,
            k=k,
            s=s,
            #ohe=True,
            non_linearity=non_linearity,
            n_jobs=n_jobs,
            rand_gen=rand_gen,
            test=test,
        )

        return split_data_by_clusters(local_data, clusters, scope, rows=False)

    return split_cols_RDC_py


def get_split_rows_RDC_py(n_clusters=2, ohe=True, k=10, s=1 / 6, non_linearity=np.sin, n_jobs=-2, rand_gen=None):
    def split_rows_RDC_py(local_data, ds_context, scope):
        meta_types = ds_context.get_meta_types_by_scope(scope)
        domains = ds_context.get_domains_by_scope(scope)

        rdc_data = rdc_transformer(
            local_data,
            meta_types,
            domains,
            k=k,
            s=s,
            non_linearity=non_linearity,
            return_matrix=True,
            rand_gen=rand_gen,
        )

        clusters = KMeans(n_clusters=n_clusters, random_state=rand_gen, n_jobs=n_jobs).fit_predict(rdc_data)

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_RDC_py


def get_split_rows_RDC_plus(n_clusters=2, ohe=True, k=10, s=1 / 6, non_linearity=np.sin, n_jobs=-2, rand_gen=None, pre_proc=None):
    
    def split_rows_RDC_plus(local_data, ds_context, scope):
        # this function contains first running the RDC split
        # then checking whether it succeeds in finding a split that explains variance
        # if not, evaluating with "score" how much variance can be explained for one variable by splitting only on that variable
        # and then using this score to weigh the ecdf values to cluster these data
        
        meta_types = ds_context.get_meta_types_by_scope(scope)
        domains = ds_context.get_domains_by_scope(scope)

        rdc_data = rdc_transformer(
            local_data,
            meta_types,
            domains,
            k=k,
            s=s,
            non_linearity=non_linearity,
            return_matrix=True,
            rand_gen=rand_gen,
        )

        RDC_res = KMeans(n_clusters=n_clusters, random_state=rand_gen, n_jobs=n_jobs).fit(rdc_data)
        
        # with inertia we extract the within cluster sum of squared distances
        
        SS_within_RDC= RDC_res.inertia_
        SS_between_RDC= np.sum((rdc_data-np.mean(rdc_data,0))**2)-SS_within_RDC
        score_RDC=SS_within_RDC /SS_between_RDC
        #print(local_data.shape)
    
        n,d= local_data.shape
        score=np.zeros(d)
        variables=np.transpose(local_data)
        
        for i in range(d):
            res = KMeans(n_clusters=n_clusters, random_state=rand_gen).fit(variables[i].reshape(n,1))
            SS_within= res.inertia_
            SS_between= np.sum((variables[i]-np.mean(variables[i],0))**2)-SS_within
            # create a score that indicates the univariate gain of k-means
            score[i]=SS_within/SS_between
        
        score[score>1]=1


        weights= np.array([1-score]*n).reshape((n,d))
        
        ecdf_data=np.apply_along_axis(ecdf, 0, local_data)
        weighted_data= weights*ecdf_data
        
        RDC_plus_res = KMeans(n_clusters=n_clusters, random_state=rand_gen).fit(weighted_data)
        SS_within_RDC_plus= RDC_plus_res.inertia_
        SS_between_RDC_plus= np.sum((weighted_data-np.mean(weighted_data,0))**2)-SS_within_RDC_plus
        score_RDC_plus=SS_within_RDC_plus/SS_between_RDC_plus
        
        
        if score_RDC_plus<score_RDC:
            clusters= RDC_plus_res.fit_predict(weighted_data)
        else:
            clusters=RDC_res.fit_predict(rdc_data)
                

        return split_data_by_clusters(local_data, clusters, scope)

    return split_rows_RDC_plus








def get_split_rows_RDC_plus_kmeans(n_clusters=2, k=10, s=1 / 6, non_linearity=np.sin, n_jobs=-2, rand_gen=None, pre_proc=None, seed=17):

    def split_rows_RDC_plus_kmeans(local_data, ds_context, scope):
        
        d=len(ds_context.meta_types)

        
        
        #print("GOING TO CHOOSE NOW")
        #print(scope)
        #print(ds_context.meta_types)
        #print(np.array(ds_context.meta_types)[scope])
             
        
        
        
        if np.all(np.array(ds_context.meta_types)[scope]==np.repeat(MetaType.REAL,len(scope))):

            #print("CHOSE KMEANS")
            ohe=False
            data = preproc(local_data, ds_context, pre_proc, ohe)
            clusters = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(data)
            #print(local_data.shape)
            # print this to see whether it is only applied to mixed data 
            #print("KMEANS meta types")
            #print(np.array(ds_context.meta_types)[scope])
            return split_data_by_clusters(local_data, clusters, scope, rows=True)
        else:
            #print("CHOSE RDC")
            ohe=True
            # this is the RDC function
            meta_types = ds_context.get_meta_types_by_scope(scope)
            domains = ds_context.get_domains_by_scope(scope)
            rdc_data = rdc_transformer(
                local_data,
                meta_types,
                domains,
                k=k,
                s=s,
                non_linearity=non_linearity,
                return_matrix=True,
                rand_gen=rand_gen,
            )
    
            RDC_res = KMeans(n_clusters=n_clusters, random_state=rand_gen, n_jobs=n_jobs).fit(rdc_data)
            
            # with inertia we extract the within cluster sum of squared distances
            
            SS_within_RDC= RDC_res.inertia_
            SS_between_RDC= np.sum((rdc_data-np.mean(rdc_data,0))**2)-SS_within_RDC
            score_RDC=SS_within_RDC /SS_between_RDC
            #print(local_data.shape)
        
            n,d= local_data.shape
            score=np.zeros(d)
            variables=np.transpose(local_data)
            
            for i in range(d):
                res = KMeans(n_clusters=n_clusters, random_state=rand_gen).fit(variables[i].reshape(n,1))
                SS_within= res.inertia_
                SS_between= np.sum((variables[i]-np.mean(variables[i],0))**2)-SS_within
                # create a score that indicates the univariate gain of k-means
                score[i]=SS_within/SS_between
            
            score[score>1]=1

            #print(local_data.shape)
            # print this to see whether it is only applied to mixed data 
            #print("rdc meta types")
            #print(np.array(ds_context.meta_types)[scope])
    
            weights= np.array([1-score]*n).reshape((n,d))
            
            ecdf_data=np.apply_along_axis(ecdf, 0, local_data)
            weighted_data= weights*ecdf_data
            
            RDC_plus_res = KMeans(n_clusters=n_clusters, random_state=rand_gen).fit(weighted_data)
            SS_within_RDC_plus= RDC_plus_res.inertia_
            SS_between_RDC_plus= np.sum((weighted_data-np.mean(weighted_data,0))**2)-SS_within_RDC_plus
            score_RDC_plus=SS_within_RDC_plus/SS_between_RDC_plus
            
            
            if score_RDC_plus<score_RDC:
                clusters= RDC_plus_res.fit_predict(weighted_data)
            else:
                clusters=RDC_res.fit_predict(rdc_data)
                        
        
            return split_data_by_clusters(local_data, clusters, scope)
    return split_rows_RDC_plus_kmeans





def get_split_rows_univ_weights(n_clusters=2, ohe=True, k=10, s=1 / 6, non_linearity=np.sin, n_jobs=-2, rand_gen=None, pre_proc=None):
    
    def split_rows_univ_weights(local_data, ds_context, scope):

        
        meta_types = ds_context.get_meta_types_by_scope(scope)
        domains = ds_context.get_domains_by_scope(scope)

        rdc_data = rdc_transformer(
            local_data,
            meta_types,
            domains,
            k=k,
            s=s,
            non_linearity=non_linearity,
            return_matrix=True,
            rand_gen=rand_gen,
        )


        n,d= local_data.shape
        score=np.zeros(d)
        variables=np.transpose(local_data)
        
        for i in range(d):
            res = KMeans(n_clusters=n_clusters, random_state=rand_gen).fit(variables[i].reshape(n,1))
            SS_within= res.inertia_
            SS_between= np.sum((variables[i]-np.mean(variables[i],0))**2)-SS_within
            # create a score that indicates the univariate gain of k-means
            score[i]=SS_within/SS_between
        
        score[score>1]=1

        weights= np.array([1-score]*n).reshape((n,d))
        
        ecdf_data=np.apply_along_axis(ecdf, 0, local_data)
        weighted_data= weights*ecdf_data
        
        RDC_plus_res = KMeans(n_clusters=n_clusters, random_state=rand_gen).fit(weighted_data)

        
        
        clusters= RDC_plus_res.fit_predict(weighted_data)
               

        return split_data_by_clusters(local_data, clusters, scope)

    return split_rows_univ_weights



def get_adjacency_matrix_rdc_py(
    local_data, meta_types, domains, k=None, s=1.0 / 6.0, non_linearity=np.sin, n_jobs=-2, rand_gen=None
):
    rdc_adjacency_matrix, p_values = rdc_test(
        local_data, meta_types, domains, k=k, s=s, non_linearity=non_linearity, n_jobs=n_jobs, rand_gen=rand_gen
    )

    #
    # Why is this necessary?
    #
    
       
    
    return rdc_adjacency_matrix


# def get_split_rows_univ(n_clusters=2, ohe=True, k=10, s=1 / 6, non_linearity=np.sin, n_jobs=-2, rand_gen=None, pre_proc=None, seed=17):
    
#     def split_rows_univ(local_data, ds_context, scope):
#         # with this function we choose the split based on which variable is most highly correlated with the rest of the variables
        
#         n,d=local_data.shape
#         meta_types = ds_context.get_meta_types_by_scope(scope)
#         domains = ds_context.get_domains_by_scope(scope)
#         rdc_adjacency_matrix= get_adjacency_matrix_rdc_py( local_data, meta_types, domains, k=None, s=1.0 / 6.0, non_linearity=np.sin, n_jobs=-2, rand_gen=None)

#         cor_per_var= np.mean(rdc_adjacency_matrix,1)+np.mean(rdc_adjacency_matrix,0)
#         # we pick the variable with the maximum correlation as the splitting variable
#         discrete_vars= (meta_types!=MetaType.REAL)

#         if np.sum(meta_types==np.array([MetaType.REAL]))<=(d/2):
#             # if there are more discrete variables than continuous variables (or equal), pick a discrete variable to split
#             cor_per_var_discr= cor_per_var[discrete_vars]
#             local_data_discr= local_data.T[discrete_vars]

#             splitting_var= local_data_discr[cor_per_var_discr==np.max(cor_per_var_discr)]

#         else:
#             # we choose the first variable, so that it remains a univariate split
#             splitting_var= local_data.T[cor_per_var==np.max(cor_per_var)][0]
#             print(splitting_var)

#             print(cor_per_var)
#             print(splitting_var.shape)
#             print(cor_per_var==np.max(cor_per_var))
            
#         clusters = KMean s(n_clusters=n_clusters, random_state=seed).fit_predict(splitting_var.T)
#         return split_data_by_clusters(local_data, clusters, scope)

#     return split_rows_univ



def get_split_rows_univ(n_clusters=2, ohe=True, k=10, s=1 / 6, non_linearity=np.sin, n_jobs=-2, rand_gen=None, pre_proc=None, seed=17):
    
    def split_rows_univ(local_data, ds_context, scope):
        # with this function we choose the split based on which variable is most highly correlated with the rest of the variables
        # we weight these correlations by the additional explained variance of the univariate splits
        # this will make it more likely that discrete variables with few levels are chosen.
        n,d=local_data.shape
        meta_types = ds_context.get_meta_types_by_scope(scope)
        domains = ds_context.get_domains_by_scope(scope)
        rdc_adjacency_matrix= get_adjacency_matrix_rdc_py( local_data, meta_types, domains, k=None, s=1.0 / 6.0, non_linearity=np.sin, n_jobs=-2, rand_gen=None)
        cor_per_var= np.mean(rdc_adjacency_matrix,1)+np.mean(rdc_adjacency_matrix,0)
        # we pick the variable with the maximum correlation as the splitting variable
      
        univ_gain=np.zeros(d)
        variables=np.transpose(local_data)
        
        for i in range(d):
            res = KMeans(n_clusters=n_clusters, random_state=rand_gen).fit(variables[i].reshape(n,1))
            SS_within= res.inertia_
            SS_between= np.sum((variables[i]-np.mean(variables[i],0))**2)-SS_within
            # create a score that indicates the univariate gain of k-means
            univ_gain[i]=SS_within/SS_between
        
        univ_gain[univ_gain>1]=1
        score= cor_per_var*(1-univ_gain)

        # if two variables have the exact same score, we arbritrarily pick the first one

        splitting_var= local_data.T[score==np.nanmax(score)][0]

        clusters = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(splitting_var.reshape(-1,1))
       
        return split_data_by_clusters(local_data, clusters, scope)


                
    return split_rows_univ
