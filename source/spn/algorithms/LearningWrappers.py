"""
Created on March 30, 2018

@author: Alejandro Molina
"""

import numpy as np

from spn.algorithms.StructureLearning import get_next_operation, learn_structure
from spn.algorithms.CnetStructureLearning import get_next_operation_cnet, learn_structure_cnet
from spn.algorithms.Validity import is_valid

from spn.structure.Base import Sum, assign_ids

from spn.structure.leaves.histogram.Histograms import create_histogram_leaf
from spn.structure.leaves.parametric.Parametric import create_parametric_leaf
from spn.structure.leaves.piecewise.PiecewiseLinear import create_piecewise_leaf
from spn.structure.leaves.cltree.CLTree import create_cltree_leaf
from spn.algorithms.splitting.Conditioning import (
    get_split_rows_naive_mle_conditioning,
    get_split_rows_random_conditioning,
)
import logging

logger = logging.getLogger(__name__)


def learn_classifier(data, ds_context, spn_learn_wrapper, label_idx, **kwargs):
    spn = Sum()
    for label, count in zip(*np.unique(data[:, label_idx], return_counts=True)):
        branch = spn_learn_wrapper(data[data[:, label_idx] == label, :], ds_context, **kwargs)
        spn.children.append(branch)
        spn.weights.append(count / data.shape[0])

    spn.scope.extend(branch.scope)
    assign_ids(spn)

    valid, err = is_valid(spn)
    assert valid, "invalid spn: " + err

    return spn


def get_splitting_functions(cols, rows, ohe, threshold, rand_gen, n_jobs, col_test="rdc", n_clusters=2, standardize=False, ecdf=False, parties = None):
    from spn.algorithms.splitting.Clustering import get_split_rows_KMeans, get_split_rows_TSNE, get_split_rows_GMM, get_split_rows_KMeans_vertical
    from spn.algorithms.splitting.PoissonStabilityTest import get_split_cols_poisson_py
    from spn.algorithms.splitting.RDC import get_split_cols_RDC_py, get_split_rows_RDC_py, get_split_rows_RDC_plus_kmeans, get_split_rows_RDC_plus, get_split_rows_univ_weights, get_split_rows_univ, get_split_rows_cor2d, get_split_rows_mincor, get_split_rows_corsub, get_split_rows_ortho_univ, get_split_rows_multi_cor, get_split_rows_vary_y, get_split_rows_multi_univ_cor, get_split_rows_muk_cor, get_split_rows_oneatatime
    from spn.algorithms.splitting.PearsonSplitting import get_split_cols_linear

    if isinstance(cols, str):
        if cols == "rdc":
            split_cols = get_split_cols_RDC_py(threshold, rand_gen=rand_gen, ohe=ohe, n_jobs=n_jobs, test=col_test)
        elif cols=="pearson":
            split_cols= get_split_cols_linear(threshold)
        elif cols == "poisson":
            split_cols = get_split_cols_poisson_py(threshold, n_jobs=n_jobs)
        else:
            raise AssertionError("unknown columns splitting strategy type %s" % str(cols))
    else:
        split_cols = cols

    if isinstance(rows, str):
        if rows == "rdc+":
            split_rows = get_split_rows_RDC_plus(rand_gen=rand_gen, ohe=ohe, n_jobs=n_jobs)
        elif rows == "rdc+kmeans":
            split_rows = get_split_rows_RDC_plus_kmeans(rand_gen=rand_gen, n_jobs=n_jobs)            
        elif rows == "rdc":
            split_rows = get_split_rows_RDC_py(n_clusters=n_clusters, rand_gen=rand_gen, ohe=ohe, n_jobs=n_jobs)
        elif rows == "kmeans":
            split_rows = get_split_rows_KMeans(n_clusters=n_clusters, standardize=standardize, ecdf=ecdf)
        elif rows=="univ_weights":
            split_rows= get_split_rows_univ_weights()
        elif rows=="univ":
            split_rows= get_split_rows_univ()
        elif rows=="cor2d":
            split_rows= get_split_rows_cor2d()
        elif rows=="corsub":
            split_rows= get_split_rows_corsub()
        elif rows=="mincor":
             split_rows = get_split_rows_mincor()
        elif rows=="univcor":
             split_rows= get_split_rows_ortho_univ()
        elif rows=="multi_cor":
             split_rows= get_split_rows_multi_cor()
        elif rows=="vary_y":
             split_rows=get_split_rows_vary_y()
        elif rows=="univ_multi":
             split_rows= get_split_rows_multi_univ_cor()
        elif rows=="muk":
             split_rows=get_split_rows_muk_cor()
        elif rows=="oneatatime":
             split_rows= get_split_rows_oneatatime()
        elif rows == "tsne":
            split_rows = get_split_rows_TSNE()
        elif rows == "gmm":
            split_rows = get_split_rows_GMM()
        elif rows == "vertical_kmeans":
            split_rows = get_split_rows_KMeans_vertical(parties = parties, n_clusters=n_clusters, rand_gen = rand_gen)
        else:
            raise AssertionError("unknown rows splitting strategy type %s" % str(rows))
    else:
        split_rows = rows
    return split_cols, split_rows


def learn_mspn_with_missing(
    data,
    ds_context,
    cols="rdc",
    rows="kmeans",
    min_instances_slice=200,
    threshold=0.3,
    linear=False,
    ohe=False,
    leaves=None,
    memory=None,
    rand_gen=None,
    cpus=-1,
):
    if leaves is None:
        # leaves = create_histogram_leaf
        leaves = create_piecewise_leaf

    if rand_gen is None:
        rand_gen = np.random.RandomState(17)

    def l_mspn_missing(data, ds_context, cols, rows, min_instances_slice, threshold, linear, ohe):
        split_cols, split_rows = get_splitting_functions(cols, rows, ohe, threshold, rand_gen, cpus)

        nextop = get_next_operation(min_instances_slice)

        return learn_structure(data, ds_context, split_rows, split_cols, leaves, nextop)

    if memory:
        l_mspn_missing = memory.cache(l_mspn_missing)

    return l_mspn_missing(data, ds_context, cols, rows, min_instances_slice, threshold, linear, ohe)


def learn_mspn(
    data,
    ds_context,
    cols="rdc",
    rows="kmeans",
    min_instances_slice=200,
    threshold=0.3,
    ohe=False,
    leaves=None,
    memory=None,
    rand_gen=None,
    cpus=-1,
    col_test="rdc",
    #cluster_firstcluster_first=False,
    hist_source="numpy",
    #bins_np="auto"
    alpha=0,
    n_clusters=2,
    standardize=False,
    ecdf=False,
    parties = None,
):
    
    
    n,d=data.shape
    
    if leaves is None:
        leaves = create_histogram_leaf

    if rand_gen is None:
        rand_gen = np.random.RandomState(17)

    def l_mspn(data, ds_context, cols, rows, min_instances_slice, threshold, ohe, hist_source, total_n):
        split_cols, split_rows = get_splitting_functions(cols, rows, ohe, threshold, rand_gen, cpus, col_test, n_clusters=n_clusters, standardize=standardize, ecdf=ecdf, parties = parties)
        nextop = get_next_operation(min_instances_slice)
        return learn_structure(data, ds_context, split_rows, split_cols, leaves, nextop,hist_source=hist_source, total_n=total_n, alpha=alpha)

    if memory:
        l_mspn = memory.cache(l_mspn)

    return l_mspn(data, ds_context, cols, rows, min_instances_slice, threshold, ohe, hist_source, n)


def learn_parametric(
    data,
    ds_context,
    cols="rdc",
    rows="kmeans",
    min_instances_slice=200,
    min_features_slice=1,
    multivariate_leaf=False,
    cluster_univariate=False,
    threshold=0.3,
    ohe=False,
    leaves=None,
    memory=None,
    rand_gen=None,
    cpus=-1,
):
    if leaves is None:
        leaves = create_parametric_leaf

    if rand_gen is None:
        rand_gen = np.random.RandomState(17)

    def learn_param(data, ds_context, cols, rows, min_instances_slice, threshold, ohe):
        split_cols, split_rows = get_splitting_functions(cols, rows, ohe, threshold, rand_gen, cpus)

        nextop = get_next_operation(min_instances_slice, min_features_slice, multivariate_leaf, cluster_univariate)

        return learn_structure(data, ds_context, split_rows, split_cols, leaves, nextop)

    if memory:
        learn_param = memory.cache(learn_param)

    return learn_param(data, ds_context, cols, rows, min_instances_slice, threshold, ohe)


def learn_cnet(
    data,
    ds_context,
    cond="naive_mle",
    min_instances_slice=200,
    min_features_slice=1,
    memory=None,
    rand_gen=None,
    cpus=-1,
):
    leaves = create_cltree_leaf

    if cond == "naive_mle":
        conditioning = get_split_rows_naive_mle_conditioning()
    elif cond == "random":
        conditioning = get_split_rows_random_conditioning()

    if rand_gen is None:
        rand_gen = np.random.RandomState(17)

    def learn_param(data, ds_context, conditioning, min_instances_slice):
        nextop = get_next_operation_cnet(min_instances_slice, min_features_slice)
        return learn_structure_cnet(data, ds_context, conditioning, leaves, nextop)

    if memory:
        learn_param = memory.cache(learn_param)

    return learn_param(data, ds_context, conditioning, min_instances_slice)


# added from/for abda:

# from spn.algorithms.splitting.Random import get_split_cols_binary_random_partition, get_split_rows_binary_random_partition, create_random_unconstrained_type_mixture_leaf

# def learn_rand_spn(data, ds_context,
#                    min_instances_slice=200,
#                    row_a=2, row_b=5,
#                    col_a=4, col_b=5,
#                    col_threshold=0.6,
#                    memory=None, rand_gen=None):

#     def learn(data, ds_context, min_instances_slice, rand_gen):

#         if rand_gen is None:
#             rand_gen = np.random.RandomState(17)

#         ds_context.rand_gen = rand_gen

#         split_cols = get_split_cols_binary_random_partition(threshold=col_threshold,
#                                                             beta_a=col_a, beta_b=col_b)
#         splot_rows = get_split_rows_binary_random_partition(beta_a=row_a, beta_b=row_b)

#         # leaves = create_random_parametric_leaf
#         leaves = create_random_unconstrained_type_mixture_leaf

#         nextop = get_next_operation(min_instances_slice)

#         return learn_structure(data, ds_context, splot_rows, split_cols, leaves, nextop)

#     if memory:
#         learn = memory.cache(learn)

#     return learn(data, ds_context, min_instances_slice,  rand_gen)

# from spn.algorithms.splitting.Clustering import get_split_rows_KMeans, get_split_rows_TSNE, get_split_rows_GMM
# from spn.algorithms.splitting.PoissonStabilityTest import get_split_cols_poisson_py
# from spn.algorithms.splitting.RDC import get_split_cols_RDC_py, get_split_rows_RDC_py, get_split_rows_RDC_plus_kmeans, get_split_rows_RDC_plus, get_split_rows_univ_weights, get_split_rows_univ, get_split_rows_cor2d, get_split_rows_mincor, get_split_rows_corsub, get_split_rows_ortho_univ, get_split_rows_multi_cor, get_split_rows_vary_y, get_split_rows_multi_univ_cor, get_split_rows_muk_cor
# from spn.algorithms.splitting.PearsonSplitting import get_split_cols_linear
# from spn.structure.leaves.histogram.Histograms import create_histogram_leaf
# from spn.structure.leaves.piecewise.PiecewiseLinear import create_piecewise_leaf
# from spn.structure.leaves.typedleaves.TypedLeaves import create_type_leaf

# def learn_hspn(data,
#                ds_context,
#                min_instances_slice=200,
#                threshold=0.3,
#                linear=False,
#                # ohe=True,
#                memory=None,
#                rand_gen=None):

#     def learn(data, ds_context, min_instances_slice, threshold, linear, ohe, rand_gen=None):

#         if rand_gen is None:
#             rand_gen = np.random.RandomState(17)

#         ds_context.rand_gen = rand_gen

#         #
#         # FIXME: adopt the python version of RDC, allowing to deal with missing values
#         # split_cols = get_split_cols_RDC(threshold, ohe, linear)
#         split_cols = get_split_cols_RDC_py(threshold, ohe=True, k=10, s=1 / 6,
#                                            non_linearity=np.sin, n_jobs=1,
#                                            rand_gen=rand_gen)
#         split_rows = get_split_rows_RDC_py(n_clusters=2, ohe=True, k=10, s=1 / 6,
#                                            non_linearity=np.sin, n_jobs=1,
#                                            rand_gen=rand_gen)
#         # get_split_rows_RDC(n_clusters=2, k=10, s=1 / 6, ohe=True, seed=rand_gen)

#         leaves = create_type_leaf



#         nextop = get_next_operation(min_instances_slice)

#         return learn_structure(data, ds_context, split_rows, split_cols, leaves, nextop)

#     if memory:
#         learn = memory.cache(learn)

#     return learn(data, ds_context,  min_instances_slice, threshold, linear,
#                  ohe=True, rand_gen=rand_gen)