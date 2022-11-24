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
import matplotlib.pyplot as plot
from scipy.stats import chi2_contingency

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

def get_no_bins_chi(n):
    if n<80:
        no_bins= 3
    elif n<120:
        no_bins=4
    elif n<250:
        no_bins=5
    elif n<350:
        no_bins=6
    elif n<500:
        no_bins=7
    elif n<1000:
        no_bins=8
    elif n<2000:
        no_bins=9
    elif n<5000:
        no_bins=10
    else:
        no_bins= np.round(np.sqrt(n/100),0)
    return no_bins


    

# first we try correlation, since we are using this as a golden standard

def chi_square_cor(x,y):
    # no_bins= get_no_bins_chi(x.shape[0])
    # percentiles= np.linspace(0,no_bins, no_bins+1)
    # percentiles= percentiles*(100/no_bins)

    # percentiles[0]=0
    # percentiles[-1]=100
    
    
    # x_bins= np.percentile(x,percentiles)
    # y_bins= np.percentile(y,percentiles)
    # hist= plot.hist2d(x,y, bins=np.array([x_bins, y_bins]))
    # counts= hist[0]
    
    # chi_stat, p_value, dof, expected= chi2_contingency(counts) 
    
    rho,p_value= pearsonr(x,y)
    
    return p_value, rho
    # return p_value, chi_stat


def chi_cca_p(indexes):
    i,j,data= indexes
    x= data.T[i]
    y=data.T[j]
    p= chi_square_cor(x,y)[0]
    return p
    
    
def chi_cca_chi(indexes):
    i,j,data= indexes
    x= data.T[i]
    y=data.T[j]
    chi_stat= chi_square_cor(x,y)[1]
    return chi_stat

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
    # this is just filling the matrix with the values that have already been produced
    for (i, j), rdc in zip(pairwise_comparisons, rdc_vals):

        rdc_adjacency_matrix[i, j] = rdc
        rdc_adjacency_matrix[j, i] = rdc
    #
    # setting diagonal to 1
    rdc_adjacency_matrix[np.diag_indices_from(rdc_adjacency_matrix)] = 1

    return [rdc_adjacency_matrix, p_values]
def rdc_test(local_data, meta_types, domains, k=None, s=1.0 / 6.0, non_linearity=np.sin, n_jobs=-1, rand_gen=None, test="rdc"):
    n,n_features = local_data.shape


    from joblib import Parallel, delayed

    if test=="chi":
        pairwise_comparisons = list(itertools.combinations(np.arange(n_features), 2))

        rdc_vals = Parallel(n_jobs=n_jobs, max_nbytes=1024, backend="threading")(
            delayed(chi_cca_chi)((i, j, local_data)) for i, j in pairwise_comparisons
        )

        p_vals= Parallel(n_jobs=n_jobs, max_nbytes=1024, backend="threading")(
            delayed(chi_cca_p)((i, j, local_data)) for i, j in pairwise_comparisons
        )

    if test=="rdc":
        rdc_features = rdc_transformer(
            local_data, meta_types, domains, k=k, s=s, non_linearity=non_linearity, return_matrix=False, rand_gen=rand_gen
        )
    
        pairwise_comparisons = list(itertools.combinations(np.arange(n_features), 2))
            
        rdc_vals = Parallel(n_jobs=n_jobs, max_nbytes=1024, backend="threading")(
            delayed(rdc_cca)((i, j, rdc_features)) for i, j in pairwise_comparisons
        )

    
    rdc_adjacency_matrix = np.zeros((n_features, n_features))

    p_values = np.zeros((n_features, n_features))
    # this is just filling the matrix with the values that have already been produced
    for (i, j), rdc in zip(pairwise_comparisons, rdc_vals):

        rdc_adjacency_matrix[i, j] = rdc
        rdc_adjacency_matrix[j, i] = rdc
    
    if test=="chi":        
        for (i, j), p_value in zip(pairwise_comparisons, p_vals):
    
            p_values[i, j] = p_value
            p_values[j, i] = p_value
            
            
    #
    # setting diagonal to 1
    
    if test=="chi":
        #!!!!!!!!!!!!!!!!!! voor cor
        rdc_adjacency_matrix[np.diag_indices_from(rdc_adjacency_matrix)] = 0
        p_values[np.diag_indices_from(p_values)]=1

    else:
        rdc_adjacency_matrix[np.diag_indices_from(rdc_adjacency_matrix)] = 1

    return [rdc_adjacency_matrix, p_values]


def getIndependentRDCGroups_py(
    local_data, threshold, meta_types, domains, k=None, s=1.0 / 6.0, non_linearity=np.sin, n_jobs=-2, rand_gen=None, test="rdc"
):
    rdc_adjacency_matrix, p_values = rdc_test(
        local_data, meta_types, domains, k=k, s=s, non_linearity=non_linearity, n_jobs=n_jobs, rand_gen=rand_gen, test=test
    )

    #
    # Why is this necessary?
    #
    
    # we print to see nans

          
    # where do these nans come from?
    rdc_adjacency_matrix[np.isnan(rdc_adjacency_matrix)] = 0
    # added by kroes et al.:
    if test!="chi":
        rdc_adjacency_matrix[np.corrcoef(local_data.T)>rdc_adjacency_matrix]=np.corrcoef(local_data.T)[np.corrcoef(local_data.T)>rdc_adjacency_matrix]
    
    n_features = local_data.shape[1]
 
    # thresholding 
    if test== "rdc":
        rdc_adjacency_matrix[rdc_adjacency_matrix < threshold] = 0
    # if p_value, then we deem groups to be independent with a p value over .05
    if test=="p_value":
        rdc_adjacency_matrix[p_values > threshold] = 0
        rdc_adjacency_matrix[np.diag_indices_from(rdc_adjacency_matrix)] = 1
    if test=="chi":
        
        rdc_adjacency_matrix[p_values > .05] = 0
        #we also ignore low significant correlations
        
        np.fill_diagonal(rdc_adjacency_matrix, 1)       
        rdc_adjacency_matrix[np.diag_indices_from(rdc_adjacency_matrix)] = 1 
        rdc_adjacency_matrix[rdc_adjacency_matrix<.05] = 0
        #print("adjacencymatrix after thresholding", rdc_adjacency_matrix)
        
    if test== "rdc":
        rdc_adjacency_matrix[rdc_adjacency_matrix < .05] = 0
    # logger.info("thresholding %s", rdc_adjacency_matrix)

    #
    # getting connected components
    

    
    result = np.zeros(n_features)
    for i, c in enumerate(connected_components(from_numpy_matrix(rdc_adjacency_matrix))):
        result[list(c)] = i + 1
    #("rdc network result")
    #print(result)
    
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
        # we also weigh what percentage of the domain has been split
        # to push the algorithm towards more equally spanned domains
        # we only perform this algorithm for (partially) discrete data
        n,d=local_data.shape
        meta_types = ds_context.get_meta_types_by_scope(scope)
        domains = ds_context.get_domains_by_scope(scope)
 
        rdc_adjacency_matrix, p_values = rdc_test(
            local_data, meta_types, domains, k=k, s=s, non_linearity=non_linearity, n_jobs=n_jobs, rand_gen=rand_gen
        )

        #rdc_adjacency_matrix= get_adjacency_matrix_rdc_py( local_data, meta_types, domains, k=None, s=1.0 / 6.0, non_linearity=np.sin, n_jobs=-2, rand_gen=None)
        rdc_adjacency_matrix[np.corrcoef(local_data.T)>rdc_adjacency_matrix]=np.corrcoef(local_data.T)[np.corrcoef(local_data.T)>rdc_adjacency_matrix]

        cor_per_var= np.mean(rdc_adjacency_matrix,1)+np.mean(rdc_adjacency_matrix,0)
        # we pick the variable with the maximum correlation as the splitting variable
        
        original_domains = np.array(ds_context.get_domains_by_scope(scope))

        # we get the local domains:
        local_domains= np.array([np.min(local_data,0),np.max(local_data,0)]).T
        domain_lengths= original_domains.T[1]-original_domains.T[0]
        local_domain_lengths= local_domains.T[1]-local_domains.T[0]

        # save proportion of domains included in this cluster:
        prop_domains= local_domain_lengths/domain_lengths

                   
        univ_gain=np.zeros(d)
        variables=np.transpose(local_data)
               
        
        for i in range(d):
            res = KMeans(n_clusters=n_clusters, random_state=rand_gen).fit(variables[i].reshape(n,1))
            SS_within= res.inertia_
 
            SS_between= np.sum((variables[i]-np.mean(variables[i],0))**2)-SS_within
 
            # create a score that indicates the univariate gain of k-means
            univ_gain[i]=SS_within/SS_between

            
        
        univ_gain[univ_gain>1]=1
        score= cor_per_var*(univ_gain)*prop_domains
    
        splitting_var= local_data.T[score==np.nanmax(score)]
       
        clusters = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(splitting_var.reshape(-1,1))
       
        return split_data_by_clusters(local_data, clusters, scope)


                
    return split_rows_univ







def best_split_through_center(data, slopes= "tan"):
# try different slopes 
    n,d= data.shape
    if slopes== "tan":
        angles= np.linspace(0,180,180)
        slopes= np.tan(angle_to_rad(angles))
    

    cors= np.zeros((len(slopes),2))
    middle= np.median(data,0)
    
    for i in range(len(slopes)):
        
        # we set the intercept so that the line passes through the middle point
        
        intercept= middle[1]-slopes[i]*middle[0]             
        cluster1= data[(data.T[0]*slopes[i]+intercept)<data.T[1]]
        cluster2= data[(data.T[0]*slopes[i]+intercept)>=data.T[1]]
    
        cors[i]= np.array([np.corrcoef(cluster1.T)[0][1],np.corrcoef(cluster2.T)[0][1]])
        
    mean_abs_cors= np.mean(np.absolute(cors),1)
    best_slope=slopes[mean_abs_cors==np.min(mean_abs_cors)][0]
    print("minimum cor=")
    print(np.min(mean_abs_cors))
    clusters=np.zeros(n)
    intercept= middle[1]-best_slope*middle[0]           

    clusters[(data.T[0]*best_slope+intercept)<data.T[1]]=1
    
    return clusters






def get_split_rows_cor2d(n_clusters=2, ohe=True, k=10, s=1 / 6, non_linearity=np.sin, n_jobs=-2, rand_gen=None, pre_proc=None, seed=17):
    
    def split_rows_cor2d(local_data, ds_context, scope):

        # splitting function only for 2d data
        # that evaluates different splits through the middle
        # and chooses the split with minimum inter-cluster correlation
        clusters = best_split_through_center(local_data)
        
        return split_data_by_clusters(local_data, clusters, scope)


                
    return split_rows_cor2d


def get_split_rows_univkmeans(n_clusters=2, ohe=True, k=10, s=1 / 6, non_linearity=np.sin, n_jobs=-2, rand_gen=None, pre_proc=None, seed=17):
    
    def split_rows_univ(local_data, ds_context, scope):
        # with this function we choose the split based on which variable is most highly correlated with the rest of the variables
        # we weight these correlations by the additional explained variance of the univariate splits
        # this will make it more likely that discrete variables with few levels are chosen.
        # we also weigh what percentage of the domain has been split
        # to push the algorithm towards more equally spanned domains
        # we only perform this algorithm for (partially) discrete data
        n,d=local_data.shape
        meta_types = ds_context.get_meta_types_by_scope(scope)
        domains = ds_context.get_domains_by_scope(scope)
        if np.all(np.array(ds_context.meta_types)[scope]==np.repeat(MetaType.REAL,len(scope))):
  
              # we use kmeans for purely continuous data
              ohe=False
              data = preproc(local_data, ds_context, pre_proc, ohe)
              clusters = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(data)

              return split_data_by_clusters(local_data, clusters, scope, rows=True)
        else:        
            rdc_adjacency_matrix, p_values = rdc_test(
                local_data, meta_types, domains, k=k, s=s, non_linearity=non_linearity, n_jobs=n_jobs, rand_gen=rand_gen
            )
    
            #rdc_adjacency_matrix= get_adjacency_matrix_rdc_py( local_data, meta_types, domains, k=None, s=1.0 / 6.0, non_linearity=np.sin, n_jobs=-2, rand_gen=None)
            rdc_adjacency_matrix[np.corrcoef(local_data.T)>rdc_adjacency_matrix]=np.corrcoef(local_data.T)[np.corrcoef(local_data.T)>rdc_adjacency_matrix]
    
            cor_per_var= np.mean(rdc_adjacency_matrix,1)+np.mean(rdc_adjacency_matrix,0)
            # we pick the variable with the maximum correlation as the splitting variable
            
            original_domains = np.array(ds_context.get_domains_by_scope(scope))
    
            # we get the local domains:
            local_domains= np.array([np.min(local_data,0),np.max(local_data,0)]).T
            domain_lengths= original_domains.T[1]-original_domains.T[0]
            local_domain_lengths= local_domains.T[1]-local_domains.T[0]
    
            # save proportion of domains included in this cluster:
            prop_domains= local_domain_lengths/domain_lengths
    
                       
            univ_gain=np.zeros(d)
            variables=np.transpose(local_data)
                   
            
            for i in range(d):
                res = KMeans(n_clusters=n_clusters, random_state=rand_gen).fit(variables[i].reshape(n,1))
                SS_within= res.inertia_
     
                SS_between= np.sum((variables[i]-np.mean(variables[i],0))**2)-SS_within
     
                # create a score that indicates the univariate gain of k-means
                univ_gain[i]=SS_within/SS_between
    
                
            
            univ_gain[univ_gain>1]=1
            score= cor_per_var*(univ_gain)*prop_domains
        
            splitting_var= local_data.T[score==np.nanmax(score)]
           
            clusters = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(splitting_var.reshape(-1,1))
           
            return split_data_by_clusters(local_data, clusters, scope)


                
    return split_rows_univ






















#############################################

# sub aanpak

def effect_of_slope_on_cor_2d_tan(data, angle):
       
    # we set the intercept so that the line passes through the middle point
    middle= np.median(data,0)

    slope=np.tan(angle_to_rad(angle))
    intercept= middle[1]-slope*middle[0]             
    cluster1= data[(data.T[0]*slope+intercept)<data.T[1]]
    cluster2= data[(data.T[0]*slope+intercept)>=data.T[1]]

    cor= np.array([np.corrcoef(cluster1.T)[0][1],np.corrcoef(cluster2.T)[0][1]])
    
    mean_abs_cors= np.mean(np.absolute(cor))

    return mean_abs_cors

def optimize_split_through_center(data):
    n,d= data.shape
    # we create a partial function that is for this specific data set
    f_data= partial(effect_of_slope_on_cor_2d_tan, data)
           
    space = hp.uniform('x', 0, 360)
    # Create the algorithm
    tpe_algo = tpe.suggest
    
    # Create a trials object
    tpe_trials = Trials()
    
    # Run 2000 evals with the tpe algorithm
    tpe_best = fmin(fn=f_data, space=space, 
                    algo=tpe_algo, trials=tpe_trials, 
                    max_evals=350)
    best_slope= np.tan(angle_to_rad(tpe_best['x']))
        # we set the intercept so that the line passes through the middle point
    
    middle= np.median(data,0)

    clusters=np.zeros(n)
    intercept= middle[1]-best_slope*middle[0]           
    clusters[(data.T[0]*best_slope+intercept)<data.T[1]]=1
    
    return clusters



def get_split_rows_corsub(n_clusters=2, ohe=True, k=10, s=1 / 6, non_linearity=np.sin, n_jobs=-2, rand_gen=None, pre_proc=None, seed=17):
    
    def split_rows_corsub(local_data, ds_context, scope):

        # splitting function only for 2d data
        # that evaluates different splits through the middle
        # and chooses the split with minimum inter-cluster correlation

        cors= np.absolute(np.corrcoef(local_data.T))
        np.fill_diagonal(cors,0)
        
        # we can just replace the max
        
        # first find the max
        
        # select the variables with the highest correlation
        inds= np.where(cors==np.max(cors))[0]

        # Take the first one and see if it has a first element
        try:
            inds[1]
        except:
            inds= np.where(cors==np.max(cors))
       
        highest=np.zeros((2), dtype=int)
        highest[0]= inds[0]
        highest[1]= inds[1]
        
        # only get the highest bivariate correlation and minimize that
        sub_data= local_data[:,highest]
        
                
        #clusters = best_split_through_center(sub_data)
        clusters = optimize_split_through_center(sub_data)

        return split_data_by_clusters(local_data, clusters, scope)


                
    return split_rows_corsub































































#############################################################################
# optimization of split with hyperplane, minimize within cluster correlation
import numpy as np
#from hyperopt import hp, tpe, fmin, Trials
from functools import partial

def angle_to_rad(angle):
    radius= angle/360*(2*np.pi)
    return radius



def effect_of_slope_on_one_cor_tan(data, *angles_dict, cor_ind=-1):
    
    # so far this function only works for continuous variables 
    # this is the same function as the previous one but then generalized to
    # multiple dimensions
    # for every dimension we fill in an angle
    # because the angles come as a dictionary, we first need to unpack all the selected values
    # it is also the same function as the next one, but we optimize one specific correlation
    
    n,d=data.shape
    xs= data.T[0:(d-1)].T
    y=data.T[-1].T    
    angles_dict= angles_dict[0]   
    
    if type(angles_dict)==tuple:
        angles= np.array([angles_dict])
    else:             
        angles = []
        
        for key in angles_dict.keys() :
            angles.append(angles_dict[key])    
            
    # transform each angle to a radius so we can extract the slope
    angles= np.array(angles)
    rads= angle_to_rad(angles)
    slopes= np.tan(rads)

    middle= np.median(data,0)
    middle_xs= middle[0:(d-1)]

    intercept= middle[-1]-np.sum(slopes*middle_xs)

    cluster1_inds= (np.sum(slopes*xs,1)+intercept)<y  
    cluster1_inds= cluster1_inds.reshape(n)*1
    cluster1= data[cluster1_inds==1]
    cluster2= data[cluster1_inds==0]

    # now that we have the clusters we can quantify the correlation within each cluster
    cor1= np.absolute(np.corrcoef(cluster1.T))
    cor2= np.absolute(np.corrcoef(cluster2.T))

    # we assume correlation measures are only zero of the 
    cor1[np.isnan(cor1)]=0
    cor2[np.isnan(cor2)]=0

    mean_abs_cor= np.mean(np.array([cor1[cor_ind], cor2[cor_ind]]))
    #print(mean_abs_cor)
    
    
    return mean_abs_cor

def effect_of_slope_on_cor_tan(data, *angles_dict):
    
    # so far this function only works for continuous variables 
    # this is the same function as the previous one but then generalized to
    # multiple dimensions
    # for every dimension we fill in an angle
    # because the angles come as a dictionary, we first need to unpack all the selected values
    
    n,d=data.shape
     
    # we define the x values
    # which are all the variables in the data except the variable we use as a reference
    # we have chosen to always use the last variable as a reference (which we refer to as y)
    
    xs= data.T[0:(d-1)].T
    
    
    y=data.T[-1].T    

    angles_dict= angles_dict[0]   

    
    if type(angles_dict)==tuple:
        angles= np.array([angles_dict])
    else:             
        angles = []
        
        for key in angles_dict.keys() :
            angles.append(angles_dict[key])    
            
    # transform each angle to a radius so we can extract the slope
    angles= np.array(angles)
    rads= angle_to_rad(angles)
    slopes= np.tan(rads)


    # we choose the split such that the line passes through the middle
    # here the middle is defined as the median on all variables
    # this means we do not need to estimate the intercept and it will probably lead
    # to groups of similar size
    # we compute the 
    middle= np.median(data,0)
    middle_xs= middle[0:(d-1)]

    intercept= middle[-1]-np.sum(slopes*middle_xs)

    # now that we have the intercept we can compute the clusters
    # save the indices of cluster 1
    
   
    cluster1_inds= (np.sum(slopes*xs,1)+intercept)<y

    
    cluster1_inds= cluster1_inds.reshape(n)*1
    cluster1= data[cluster1_inds==1]
    cluster2= data[cluster1_inds==0]

    # now that we have the clusters we can quantify the correlation within each cluster
    cor1= np.sum(np.absolute(np.corrcoef(cluster1.T)))-d
    cor2= np.sum(np.absolute(np.corrcoef(cluster2.T)))-d

    mean_abs_cor= np.mean(np.array([cor1, cor2]))

    return mean_abs_cor

def get_clusters_for_angles(data, *angles_dict):
    
    # so far this function only works for continuous variables 
    # this is the same function as the previous one but then generalized to
    # multiple dimensions
    # for every dimension we fill in an angle
    # because the angles come as a dictionary, we first need to unpack all the selected values
    
    n,d=data.shape
     
    # we define the x values
    # which are all the variables in the data except the variable we use as a reference
    # we have chosen to always use the last variable as a reference (which we refer to as y)
    
    xs= data.T[0:(d-1)].T
        
    y=data.T[-1].T    

    angles_dict= angles_dict[0]   

    
    if type(angles_dict)==tuple:
        angles= np.array([angles_dict])
    else:             
        angles = []
        
        for key in angles_dict.keys() :
            angles.append(angles_dict[key])    
            
    # transform each angle to a radius so we can extract the slope
    angles= np.array(angles)
    rads= angle_to_rad(angles)
    slopes= np.tan(rads)


    # we choose the split such that the line passes through the middle
    # here the middle is defined as the median on all variables
    # this means we do not need to estimate the intercept and it will probably lead
    # to groups of similar size
    # we compute the 
    middle= np.median(data,0)
    middle_xs= middle[0:(d-1)]

    intercept= middle[-1]-np.sum(slopes*middle_xs)

    # now that we have the intercept we can compute the clusters
    # save the indices of cluster 1
    
  
    cluster1_inds= (np.sum(slopes*xs,1)+intercept)<y

    
    cluster_inds= cluster1_inds.reshape(n)*1


    return cluster_inds




def clusters_optimized_cor(data, max_evals=350):
    f_data= partial(effect_of_slope_on_cor_tan, data)
    n,d= data.shape


    if d==9:
        lin_search_space= {'x1':hp.uniform('x1', 0, 180),
                          'x2':hp.uniform('x2', 0, 180),
                          'x3':hp.uniform('x3', 0, 180),
                          'x4':hp.uniform('x4', 0, 180),
                          'x5':hp.uniform('x5', 0, 180),
                          'x6':hp.uniform('x6', 0, 180),
                          'x7':hp.uniform('x7', 0, 180),
                          'x8':hp.uniform('x8', 0, 180),
                   
          }
        
    if d==8:
        lin_search_space= {'x1':hp.uniform('x1', 0, 180),
                          'x2':hp.uniform('x2', 0, 180),
                          'x3':hp.uniform('x3', 0, 180),
                          'x4':hp.uniform('x4', 0, 180),
                          'x5':hp.uniform('x5', 0, 180),
                          'x6':hp.uniform('x6', 0, 180),
                          'x7':hp.uniform('x7', 0, 180),
                   
          }
    if d==8:
        lin_search_space= {'x1':hp.uniform('x1', 0, 180),
                          'x2':hp.uniform('x2', 0, 180),
                          'x3':hp.uniform('x3', 0, 180),
                          'x4':hp.uniform('x4', 0, 180),
                          'x5':hp.uniform('x5', 0, 180),
                          'x6':hp.uniform('x6', 0, 180),
                          'x7':hp.uniform('x7', 0, 180),
                   
          }
    if d==8:
        lin_search_space= {'x1':hp.uniform('x1', 0, 180),
                          'x2':hp.uniform('x2', 0, 180),
                          'x3':hp.uniform('x3', 0, 180),
                          'x4':hp.uniform('x4', 0, 180),
                          'x5':hp.uniform('x5', 0, 180),
                          'x6':hp.uniform('x6', 0, 180),
                          'x7':hp.uniform('x7', 0, 180),
                   
          }
    if d==7:
        lin_search_space= {'x1':hp.uniform('x1', 0, 180),
                          'x2':hp.uniform('x2', 0, 180),
                          'x3':hp.uniform('x3', 0, 180),
                          'x4':hp.uniform('x4', 0, 180),
                          'x5':hp.uniform('x5', 0, 180),
                          'x6':hp.uniform('x6', 0, 180),
                   
          }
    if d==6:
        lin_search_space= {'x1':hp.uniform('x1', 0, 180),
                          'x2':hp.uniform('x2', 0, 180),
                          'x3':hp.uniform('x3', 0, 180),
                          'x4':hp.uniform('x4', 0, 180),
                          'x5':hp.uniform('x5', 0, 180),

                   
          }
    if d==5:
        lin_search_space= {'x1':hp.uniform('x1', 0, 180),
                          'x2':hp.uniform('x2', 0, 180),
                          'x3':hp.uniform('x3', 0, 180),
                          'x4':hp.uniform('x4', 0, 180),
                  
          }        
    
    if d==4:
        lin_search_space= {'x1':hp.uniform('x1', 0, 180),
                          'x2':hp.uniform('x2', 0, 180),
                          'x3':hp.uniform('x3', 0, 180),
                  
          }   
        
    if d==3:
        lin_search_space= {'x1':hp.uniform('x1', 0, 180),
                          'x2':hp.uniform('x2', 0, 180),
                  
          }   
    if d==2:
        lin_search_space= {'x1':hp.uniform('x1', 0, 180),
                  
          }   
    print("sample size =")
    print(data.shape[0])
    tpe_algo = tpe.suggest
    tpe_trials = Trials()
    
    # Run 2000 evals with the tpe algorithm
    optimization_result = fmin(fn=f_data, space=lin_search_space, 
                    algo=tpe_algo, trials=tpe_trials, 
                    max_evals=max_evals)
    
    
    clusters= get_clusters_for_angles(data, optimization_result)
    
    return clusters

def get_split_rows_mincor(n_clusters=2, ohe=True, k=10, s=1 / 6, non_linearity=np.sin, n_jobs=-2, rand_gen=None, pre_proc=None, seed=17):
    
    def split_rows_mincor(local_data, ds_context, scope):

        clusters = clusters_optimized_cor(local_data)
        
        #we print the correlation within these clusters
        cluster1= local_data[clusters==1]
        cluster2= local_data[clusters==0]
        cor1_matrix= np.corrcoef(cluster1.T)
        cor2_matrix= np.corrcoef(cluster2.T)
        cor_matrix= np.corrcoef(local_data.T)
        # we assume the correlation matrix can only have zeroes
        # if it splits such that the variance is zero
        # which would mean no correlation
        cor2_matrix[np.isnan(cor2_matrix)]= 0
        cor1_matrix[np.isnan(cor1_matrix)]= 0
        cor_matrix[np.isnan(cor_matrix)]= 0

        np.fill_diagonal(cor2_matrix,0)
        np.fill_diagonal(cor1_matrix,0)
        np.fill_diagonal(cor_matrix,0)

        cor1= np.sum(np.absolute(cor1_matrix))
        cor2= np.sum(np.absolute(cor2_matrix))
        cor_total= np.mean(np.sum(np.absolute(cor_matrix)))

        cor= np.mean(np.array([cor1, cor2]))
        print("\n")
        print("\n")
        print("\n")

        print("cluster correlation=")
        print(cor)
        print(cor_total)
        print("cluster correlation lower than local_data=")
        print(cor_total>cor)

        print("%new cor of old cor (should be under a hundred)")
        
        print(cor/cor_total)
        
        print(local_data.shape)
        print("\n")
        print("\n")
        print("\n")

        

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_mincor






































###########################################################################
###########################################################################
###########################################################################
########################################################################### 
import statsmodels.api as sm


# these functions we use to create a plane orthogonal to bivariate
# correlations with variable y
def ortho_univ_split(data):
    n,d=data.shape
    y=data[:,d-1]

    betas= np.zeros((d-1))
    
    for i in range(d-1):
        
        xs=data[:,i]
        result = sm.OLS(y,xs).fit()
        betas[i]= result.params
    
    slopes=-1/betas
    return slopes
    

def get_clusters_with_slope(data, slopes, y_ind=-1):
    
    # so far this function only works for continuous variables (or at least y has to be continuous)
    # this is the same function as the previous one but then generalized to
    # multiple dimensions
    # for every dimension we 
    # in an angle
    # because the angles come as a dictionary, we first need to unpack all the selected values
    
    n,d=data.shape
     
    # we define the x values
    # which are all the variables in the data except the variable we use as a reference
    # we have chosen to always use the last variable as a reference (which we refer to as y)
    
    xs= np.delete(data, y_ind, axis=1)
    y=data.T[y_ind].T    

    middle= np.median(data,0)
    middle_xs=  np.delete(middle, y_ind, axis=0)
    intercept= middle[y_ind]-np.sum(slopes*middle_xs)

    cluster1_inds= (np.sum(slopes*xs,1)+intercept)<y 
    cluster1_inds= cluster1_inds.reshape(n)*1

    return cluster1_inds   


def get_split_rows_ortho_univ(n_clusters=2, ohe=True, k=10, s=1 / 6, non_linearity=np.sin, n_jobs=-2, rand_gen=None, pre_proc=None, seed=17):
    
    def split_rows_ortho_univ(local_data, ds_context, scope):
        # with this function we choose the split based on which variable is most highly correlated with the rest of the variables
        # we weight these correlations by the additional explained variance of the univariate splits
        # this will make it more likely that discrete variables with few levels are chosen.
        # we also weigh what percentage of the domain has been split
        # to push the algorithm towards more equally spanned domains
        # we only perform this algorithm for (partially) discrete data
        slopes= ortho_univ_split(local_data)
        clusters = get_clusters_with_slope(local_data, slopes)

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_ortho_univ



###########################################################################
###########################################################################
###########################################################################
########################################################################### 

# first we try a variant that does multivariable regression


              
def get_split_rows_multi_cor(n_clusters=2, ohe=True, k=10, s=1 / 6, non_linearity=np.sin, n_jobs=-2, rand_gen=None, pre_proc=None, seed=17):
    
    def split_rows_multi_cor(local_data, ds_context, scope):
                       
        y_ind=-1
        xs= np.delete(local_data, y_ind, axis=1)
        y=local_data[:,y_ind]
        result = sm.OLS(y,xs).fit()
        
        betas= result.params
        slopes=-1/betas

        clusters = get_clusters_with_slope(local_data, slopes)


        #we print the correlation within these clusters
        cluster1= local_data[clusters==1]
        cluster2= local_data[clusters==0]
        cor1_matrix= np.corrcoef(cluster1.T)
        cor2_matrix= np.corrcoef(cluster2.T)
        cor_matrix= np.corrcoef(local_data.T)
        # we assume the correlation matrix can only have zeroes
        # if it splits such that the variance is zero
        # which would mean no correlation
        cor2_matrix[np.isnan(cor2_matrix)]= 0
        cor1_matrix[np.isnan(cor1_matrix)]= 0
        cor_matrix[np.isnan(cor_matrix)]= 0

        np.fill_diagonal(cor2_matrix,0)
        np.fill_diagonal(cor1_matrix,0)
        np.fill_diagonal(cor_matrix,0)

        cor1= np.sum(np.absolute(cor1_matrix))
        cor2= np.sum(np.absolute(cor2_matrix))
        cor_total= np.mean(np.sum(np.absolute(cor_matrix)))

        cor= np.mean(np.array([cor1, cor2]))
        print("\n")
        print("\n")
        print("\n")

        print("cluster correlation=")
        print(cor)
        print(cor_total)
        print("cluster correlation lower than local_data=")
        print(cor_total>cor)
        print(local_data.shape)
        print("\n")
        print("\n")
        print("\n")

        print("%new cor of old cor (should be under a hundred)")
        
        print(cor/cor_total)
        
        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_multi_cor



###########################################################################
###########################################################################
###########################################################################
########################################################################### 

# we also want to try another variant that changes the y variable



def get_split_rows_vary_y(n_clusters=2, ohe=True, k=10, s=1 / 6, non_linearity=np.sin, n_jobs=-2, rand_gen=None, pre_proc=None, seed=17):
    
    def split_rows_vary_y(local_data, ds_context, scope):
        
        n,d= local_data.shape
        cors= np.absolute(np.corrcoef(local_data.T))
        np.fill_diagonal(cors,0)
        
        # we can just replace the max
        
        # first find the max
        
        # select the variables with the highest correlation
        inds= np.where(cors==np.max(cors))[0]
        # we know that this value must be bigger than the sum of correlations
        cors= np.repeat(float(d)**2,float(d))


        # lets just loop over the y's because it's not that expensive to try them out
        for i in range(d):
            y_ind=i
            xs= np.delete(local_data, y_ind, axis=1)
            y=local_data[:,y_ind]
            result = sm.OLS(y,xs).fit()
            
            betas= result.params
            slopes=-1/betas
            
            clusters_i = get_clusters_with_slope(local_data, slopes, y_ind=y_ind)

            cluster1= local_data[clusters_i==1]
            cluster2= local_data[clusters_i==0]
        
            # now that we have the clusters we can quantify the correlation within each cluster
            cor1_matrix= np.corrcoef(cluster1.T)
            cor2_matrix= np.corrcoef(cluster2.T)
            
            # we assume the correlation matrix can only have zeroes
            # if it splits such that the variance is zero
            # which would mean no correlation
            
            cor2_matrix[np.isnan(cor2_matrix)]= 0
            cor1_matrix[np.isnan(cor1_matrix)]= 0

           
            np.fill_diagonal(cor2_matrix,0)
            np.fill_diagonal(cor1_matrix,0)

            cor1= np.sum(np.absolute(cor1_matrix))
            cor2= np.sum(np.absolute(cor2_matrix))
            
            
            

            cors[i]= np.mean(np.array([cor1, cor2]))

            if cors[i]==np.min(cors):

                clusters = get_clusters_with_slope(local_data, slopes)

                print("\n")
                print("\n")
                print("\n")

                print("cluster correlation=")
                print(cors[i])
            
        return split_data_by_clusters(local_data, clusters, scope, rows=True)
    
    return split_rows_vary_y


















###########################################################################
########################################################################### 
# here we do multivariate again, but only choose this if it decreases variance
# otherwise do univariate

def compute_cor_with_clusters(data, clusters):
    
    cluster1= data[clusters==1]
    cluster2= data[clusters==0]
    cor1_matrix= np.corrcoef(cluster1.T)
    cor2_matrix= np.corrcoef(cluster2.T)
    
    # we assume the correlation matrix can only have zeroes
    # if it splits such that the variance is zero
    # which would mean no correlation
    cor2_matrix[np.isnan(cor2_matrix)]= 0
    cor1_matrix[np.isnan(cor1_matrix)]= 0

    np.fill_diagonal(cor2_matrix,0)
    np.fill_diagonal(cor1_matrix,0)

    cor1= np.sum(np.absolute(cor1_matrix))
    cor2= np.sum(np.absolute(cor2_matrix))
    
    cor= np.mean(np.array([cor1, cor2]))

    return cor

def get_split_rows_multi_univ_cor(n_clusters=2, ohe=True, k=10, s=1 / 6, non_linearity=np.sin, n_jobs=-2, rand_gen=None, pre_proc=None, seed=17):
    
    def split_rows_multi_univ_cor(local_data, ds_context, scope):
                       
        y_ind=-1
        xs= np.delete(local_data, y_ind, axis=1)
        y=local_data[:,y_ind]
        result = sm.OLS(y,xs).fit()
        
        betas= result.params
        slopes_multi=-1/betas

        clusters_multi = get_clusters_with_slope(local_data, slopes_multi)
        cor_multi = compute_cor_with_clusters(local_data, clusters_multi)
        
        # we also try univ
        
        slopes_univ= ortho_univ_split(local_data)
        clusters_univ = get_clusters_with_slope(local_data, slopes_univ)
        cor_univ = compute_cor_with_clusters(local_data, clusters_univ)


        if cor_multi<=cor_univ:
            chosen_cor= cor_multi
            clusters= clusters_multi
        if cor_univ<cor_multi:
            chosen_cor= cor_univ
            clusters= clusters_univ
            
        cor_matrix= np.corrcoef(local_data.T)
        np.fill_diagonal(cor_matrix,0)
        cor_total= np.mean(np.sum(np.absolute(cor_matrix)))   
        print("\n")
        print("\n")
        print("\n")

        print("cluster correlation lower than local_data=")
        print(cor_total>chosen_cor)
        print(local_data.shape)
        print("\n")
        print("\n")
        print("\n")

        
        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_multi_univ_cor



###########################################################################






# we choose between univ, multi and kmeans
def get_split_rows_muk_cor(n_clusters=2, ohe=True, k=10, s=1 / 6, non_linearity=np.sin, n_jobs=-2, rand_gen=None, pre_proc=None, seed=17):
    
    def split_rows_muk_cor(local_data, ds_context, scope):
                       
        y_ind=-1
        xs= np.delete(local_data, y_ind, axis=1)
        y=local_data[:,y_ind]
        result = sm.OLS(y,xs).fit()
        
        betas= result.params
        slopes_multi=-1/betas

        clusters_multi = get_clusters_with_slope(local_data, slopes_multi)
        cor_multi = compute_cor_with_clusters(local_data, clusters_multi)
        
        # we also try univ
        
        slopes_univ= ortho_univ_split(local_data)
        clusters_univ = get_clusters_with_slope(local_data, slopes_univ)
        cor_univ = compute_cor_with_clusters(local_data, clusters_univ)


        if cor_multi<=cor_univ:
            chosen_cor= cor_multi
            clusters= clusters_multi
        if cor_univ<cor_multi:
            chosen_cor= cor_univ
            clusters= clusters_univ
            
        cor_matrix= np.corrcoef(local_data.T)
        np.fill_diagonal(cor_matrix,0)
        cor_total= np.mean(np.sum(np.absolute(cor_matrix)))   
        print("\n")
        print("\n")
        print("\n")

        print("cluster correlation lower than local_data=")
        print(cor_total>chosen_cor)
        print(local_data.shape)
        print("\n")
        print("\n")
        print("\n")
        if cor_total<=chosen_cor:
            print("chose kmeans")
            print("\n")
            print("\n")
            print("\n")
            clusters = KMeans(n_clusters=n_clusters, random_state=seed, n_jobs=1).fit_predict(local_data)
            print(cor_total>chosen_cor)

        
        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_muk_cor


###########################################################################




              
def get_split_rows_multi_univ_cor_old(n_clusters=2, ohe=True, k=10, s=1 / 6, non_linearity=np.sin, n_jobs=-2, rand_gen=None, pre_proc=None, seed=17):
    
    def split_rows_multi_univ_cor(local_data, ds_context, scope):
                       
        y_ind=-1
        xs= np.delete(local_data, y_ind, axis=1)
        y=local_data[:,y_ind]
        result = sm.OLS(y,xs).fit()
        
        betas= result.params
        slopes=-1/betas

        clusters = get_clusters_with_slope(local_data, slopes)


        #we print the correlation within these clusters
        cluster1= local_data[clusters==1]
        cluster2= local_data[clusters==0]
        cor1_matrix= np.corrcoef(cluster1.T)
        cor2_matrix= np.corrcoef(cluster2.T)
        cor_matrix= np.corrcoef(local_data.T)
        # we assume the correlation matrix can only have zeroes
        # if it splits such that the variance is zero
        # which would mean no correlation
        cor2_matrix[np.isnan(cor2_matrix)]= 0
        cor1_matrix[np.isnan(cor1_matrix)]= 0
        cor_matrix[np.isnan(cor_matrix)]= 0

        np.fill_diagonal(cor2_matrix,0)
        np.fill_diagonal(cor1_matrix,0)
        np.fill_diagonal(cor_matrix,0)

        cor1= np.sum(np.absolute(cor1_matrix))
        cor2= np.sum(np.absolute(cor2_matrix))
        cor_total= np.mean(np.sum(np.absolute(cor_matrix)))

        cor= np.mean(np.array([cor1, cor2]))
        print("\n")
        print("\n")
        print("\n")

        print("cluster correlation=")
        print(cor)
        print(cor_total)
        print("cluster correlation lower than local_data=")
        print(cor_total>cor)
        print(local_data.shape)
        print("\n")
        print("\n")
        print("\n")
        
        print("%new cor of old cor (should be under a hundred)")
        
        print(cor/cor_total)
        if cor>=cor_total:
            slopes= ortho_univ_split(local_data)
            clusters = get_clusters_with_slope(local_data, slopes)
        
        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_multi_univ_cor



###########################################################################
###########################################################################


def get_clusters_for_angles(data, meta_types, *angles_dict):
    # zelfde functie als eerder maar mogelijksgeschikt voor categorische data
    
    # so far this function only works for continuous variables 
    # this is the same function as the previous one but then generalized to
    # multiple dimensions
    # for every dimension we fill in an angle
    # because the angles come as a dictionary, we first need to unpack all the selected values
    
    n,d=data.shape
     
    # we define the x values
    # which are all the variables in the data except the variable we use as a reference
    # we have chosen to always use the last variable as a reference (which we refer to as y)
    
    xs= data.T[0:(d-1)].T
        
    y=data.T[-1].T    

    angles_dict= angles_dict[0]   

    
    if type(angles_dict)==tuple:
        angles= np.array([angles_dict])
    else:             
        angles = []
        
        for key in angles_dict.keys() :
            angles.append(angles_dict[key])    
            
    # transform each angle to a radius so we can extract the slope
    angles= np.array(angles)
    rads= angle_to_rad(angles)
    slopes= np.tan(rads)

    slopes[meta_types==MetaType.DISCRETE]=angles[meta_types==MetaType.DISCRETE]

    # we choose the split such that the line passes through the middle
    # here the middle is defined as the median on all variables
    # this means we do not need to estimate the intercept and it will probably lead
    # to groups of similar size
    # we compute the 
    middle= np.median(data,0)
    middle_xs= middle[0:(d-1)]

    intercept= middle[-1]-np.sum(slopes*middle_xs)

    # now that we have the intercept we can compute the clusters
    # save the indices of cluster 1
    
  
    cluster1_inds= (np.sum(slopes*xs,1)+intercept)<y

    
    cluster_inds= cluster1_inds.reshape(n)*1


    return cluster_inds

def effect_of_slope_on_one_cor_tan(data, meta_types, *angles_dict, cor_ind=-1):
    
    # so far this function only works for continuous variables 
    # this is the same function as the previous one but then generalized to
    # multiple dimensions
    # for every dimension we fill in an angle
    # because the angles come as a dictionary, we first need to unpack all the selected values
    # it is also the same function as the next one, but we optimize one specific correlation
    
    n,d=data.shape
    
    xs= data.T[0:(d-1)].T
    y=data.T[-1].T    

    angles_dict= angles_dict[0]   
    if type(angles_dict)==tuple:
        angles= np.array([angles_dict])
    else:             
        angles = []
        
        for key in angles_dict.keys() :
            angles.append(angles_dict[key])    
            
    # transform each angle to a radius so we can extract the slope
    angles= np.array(angles)
    rads= angle_to_rad(angles)
    slopes= np.tan(rads)

    slopes[meta_types==MetaType.DISCRETE]=angles[meta_types==MetaType.DISCRETE]
    middle= np.median(data,0)
    middle_xs= middle[0:(d-1)]

    intercept= middle[-1]-np.sum(slopes*middle_xs)

    cluster1_inds= (np.sum(slopes*xs,1)+intercept)<y  
    cluster1_inds= cluster1_inds.reshape(n)*1
    cluster1= data[cluster1_inds==1]
    cluster2= data[cluster1_inds==0]

    # now that we have the clusters we can quantify the correlation within each cluster
    cor1= np.absolute(np.corrcoef(cluster1.T))
    cor2= np.absolute(np.corrcoef(cluster2.T))

    # we assume correlation measures are only zero of the 
    cor1[np.isnan(cor1)]=0
    cor2[np.isnan(cor2)]=0

    mean_abs_cor= np.mean(np.array([cor1[cor_ind], cor2[cor_ind]]))
    #print(mean_abs_cor)
    
    
    return mean_abs_cor


def get_search_space(data, meta_types):
    
        
    n,d= data.shape
    search_space = {}
    keys = range(d-1)
    
    for i in range(d-1):
        
            key_i= 'x'+str(i)
            if meta_types[i]== MetaType.DISCRETE:  

                domain=np.unique(data.T[i])
                choices= np.concatenate([domain, np.array(domain[-1]+1).reshape(1)])-.5
                
                search_space[key_i] =hp.choice(str(i),choices)
    
            if meta_types[i]== MetaType.REAL:
                search_space[key_i] = hp.uniform(str(i), 0, 180)
    return search_space
                

# we use another variant
# that finds the optimal plane to minimize the correlation with one specific variable
# that is, the variable with the most correlation 
def clusters_optimized_one_cor(data, meta_types,  max_evals=350, cor_ind=8):
    
    # only get the meta_types that are not the cor_ind
    n,d= data.shape
    meta_types=np.array(meta_types)

    cor_ind_int=np.array(cor_ind, dtype=int).reshape(1)

    ind= np.concatenate([range(0,int(cor_ind_int)), range(int(cor_ind_int)+1,d)])

    meta_types_cor_ind= np.concatenate([meta_types[range(0,int(cor_ind_int))], meta_types[range(int(cor_ind_int)+1,d)]])
    
    f_data= partial(effect_of_slope_on_one_cor_tan,  data, meta_types_cor_ind, cor_ind=cor_ind)
    
    search_space= get_search_space(data, meta_types_cor_ind)

    print("sample size and d =")
    print(data.shape)
    tpe_algo = tpe.suggest
    tpe_trials = Trials()
    
    optimization_result = fmin(fn=f_data, space=search_space, 
                    algo=tpe_algo, trials=tpe_trials, 
                    max_evals=max_evals)
    
    print("optimized cor is:")
    print(f_data(optimization_result))
    clusters= get_clusters_for_angles(data, meta_types_cor_ind, optimization_result)
    
    return clusters



def get_split_rows_oneatatime(n_clusters=2, ohe=True, k=10, s=1 / 6, non_linearity=np.sin, n_jobs=-2, rand_gen=None, pre_proc=None, seed=17):
    
    def split_rows_oneatatime(local_data, ds_context, scope):

        # we compute the correlation matrix and then take the variable that has the largest correlation     
        
        meta_types = ds_context.get_meta_types_by_scope(scope)
        abs_cors= np.absolute(np.corrcoef(local_data.T))
        
        np.fill_diagonal(abs_cors,0)
        cors_per_var= np.sum(abs_cors,0)
        cor_ind= np.where(cors_per_var==np.max(cors_per_var))
        
        if type(cor_ind)!=tuple:
            cor_ind=cor_ind[0]
        
        
        print("cor_ind= ")
        print(cor_ind)
        print(np.max(cors_per_var))       
        print(abs_cors)

        
        clusters = clusters_optimized_one_cor(local_data, meta_types, cor_ind=cor_ind)
        
        #we print the correlation within these clusters
        cluster1= local_data[clusters==1]
        cluster2= local_data[clusters==0]
        cor1_matrix= np.corrcoef(cluster1.T)
        cor2_matrix= np.corrcoef(cluster2.T)
        cor_matrix= np.corrcoef(local_data.T)
        # we assume the correlation matrix can only have zeroes
        # if it splits such that the variance is zero
        # which would mean no correlation
        cor2_matrix[np.isnan(cor2_matrix)]= 0
        cor1_matrix[np.isnan(cor1_matrix)]= 0
        cor_matrix[np.isnan(cor_matrix)]= 0

        np.fill_diagonal(cor2_matrix,0)
        np.fill_diagonal(cor1_matrix,0)
        np.fill_diagonal(cor_matrix,0)

        cor1= np.sum(np.absolute(cor1_matrix))
        cor2= np.sum(np.absolute(cor2_matrix))
        cor_total= np.mean(np.sum(np.absolute(cor_matrix)))

        cor= np.mean(np.array([cor1, cor2]))
        print("\n")
        print("\n")
        print("\n")
        print("cluster correlation=")
        print(cor)
        print(cor_total)
        print("cluster correlation lower than local_data=")
        print(cor_total>cor)
        print(local_data.shape)
        #print(cor1_matrix)
        
        print("%new cor of old cor (should be under a hundred)")
        
        print(cor/cor_total)
        print("\n")
        print("\n")
        print("\n")
        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_oneatatime

