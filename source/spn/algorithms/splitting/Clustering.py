"""
Created on March 25, 2018

@author: Alejandro Molina, edits by Shannon Kroes
"""
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from spn.algorithms.splitting.Base import split_data_by_clusters, preproc
import logging
import os
file_wd = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_wd)
from sklearn.preprocessing import StandardScaler
import scipy

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
    with open(path + "/mixedClustering.R", "r") as rfile:
        code = "".join(rfile.readlines())
        robjects.r(code)

    numpy2ri.activate()


def ECDF(X):
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

def get_split_rows_KMeans(n_clusters=2, pre_proc=None, ohe=False, seed=17, standardize=False, ecdf=False):
    def split_rows_KMeans(local_data, ds_context, scope):
        data = preproc(local_data, ds_context, pre_proc, ohe)
        
        if standardize:
            scaler = StandardScaler().fit(data)
            standardized_data = scaler.transform(data)
            clusters = KMeans(n_clusters=n_clusters, random_state=seed,init="random").fit_predict(standardized_data)
            print("performing random clustering")
        elif ecdf:
            data_ecdf = np.apply_along_axis(ECDF, 0, data)
            clusters = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(data_ecdf)

        else:
            clusters = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(data)
        return split_data_by_clusters(local_data, clusters, scope, rows=True)
    return split_rows_KMeans

def get_split_rows_TSNE(n_clusters=2, pre_proc=None, ohe=False, seed=17, verbose=10, n_jobs=-1):
    # https://github.com/DmitryUlyanov/Multicore-TSNE
    from MulticoreTSNE import MulticoreTSNE as TSNE
    import os

    ncpus = n_jobs
    if n_jobs < 1:
        ncpus = max(os.cpu_count() - 1, 1)

    def split_rows_KMeans(local_data, ds_context, scope):
        data = preproc(local_data, ds_context, pre_proc, ohe)
        kmeans_data = TSNE(n_components=3, verbose=verbose, n_jobs=ncpus, random_state=seed).fit_transform(data)
        clusters = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(kmeans_data)

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_KMeans


def get_split_rows_DBScan(eps=2, min_samples=10, pre_proc=None, ohe=False):
    def split_rows_DBScan(local_data, ds_context, scope):
        data = preproc(local_data, ds_context, pre_proc, ohe)

        clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_DBScan


def get_split_rows_Gower(n_clusters=2, pre_proc=None, seed=17):
    from rpy2 import robjects

    init_rpy()

    def split_rows_Gower(local_data, ds_context, scope):
        data = preproc(local_data, ds_context, pre_proc, False)

        try:
            df = robjects.r["as.data.frame"](data)
            clusters = robjects.r["mixedclustering"](df, ds_context.distribution_family, n_clusters, seed)
            clusters = np.asarray(clusters)
        except Exception as e:
            np.savetxt("/tmp/errordata.txt", local_data)
            logger.info(e)
            raise e

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_Gower


def get_split_rows_GMM(n_clusters=2, pre_proc=None, ohe=False, seed=17, max_iter=100, n_init=2, covariance_type="full"):
    """
    covariance_type can be one of 'spherical', 'diag', 'tied', 'full'
    """

    def split_rows_GMM(local_data, ds_context, scope):
        data = preproc(local_data, ds_context, pre_proc, ohe)

        estimator = GaussianMixture(
            n_components=n_clusters,
            covariance_type=covariance_type,
            max_iter=max_iter,
            n_init=n_init,
            random_state=seed,
        )

        clusters = estimator.fit(data).predict(data)

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_GMM
