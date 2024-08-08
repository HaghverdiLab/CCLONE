from scipy.stats import binom
import scipy.optimize as op
import numpy as np
from joblib import Parallel, delayed
from itertools import permutations


def get_wNMF_matrices(adata, mode=""):
    adata = get_varcall(adata)
    adata = get_weights(adata, mode=mode)
    return adata


def get_weights(adata, mode=""):
    # set weight matrix
    weights = np.zeros(adata.X.shape)
    if mode == "10X":
        # Set S to probability of missing ALT when randomly sampling cov times
        # note: this is not adapted to SmartSeq2 data
        for i in np.unique(adata.X):
            p = 1 - binom.pmf(k=0, n=i, p=.5)  # prob of sampling i times the reference allele
            weights[np.where(adata.X == i)] = p
        weights[adata.layers["M"] == 0.5] = 1  # if we see both allels, the prob of having missed any is 0
        weights[np.where(adata.X < 2)] = 0  # only count from 2+ reads
    elif mode == "no_weight":
        weights = np.ones(adata.X.shape)
    elif mode == "binary":
        weights = np.ones(adata.X.shape)
        weights[np.where(adata.X < 2)] = 0
    else:
        weights[np.where(adata.X >= 2)] = 0.5
        weights[adata.layers["M"] == 0.5] = 1
    adata.layers["weights"] = weights
    return adata


def get_varcall(adata):
    # set X matrix
    M = np.zeros(adata.X.shape)
    whr = np.where(adata.X >= 2)
    # M contains VAF=.5 if we see ref and alt, VAF=1 if we see only ALT, VAF=0 else
    M[whr] = ((adata.layers["ALT"][whr] >= 1) * (adata.layers["REF"][whr] >= 1) * 0.5) + \
             ((adata.layers["ALT"][whr] >= 1) * (adata.layers["REF"][whr] == 0) * 1)
    adata.layers["M"] = M
    return adata


def NMF_weighted(adata, k=2, max_cycles=1000,
                 parallel=True, n_jobs=-1):
    """
    Function to calculate weighted NMF given variant call matrix X and weight matrix (of equal shape as X).
    We try to learn C (cells*k) and V (k*variants) that minimise the cost function:
        ((X-CV)**2)°W     with ° the hadamart product
    There is no closed solution of this equation, instead we alternatively minimize C and V.
    In this equation the rows of C and columns of V are independent. Because of this, we can mimize them in parallel
    for speedup.

    Parameters
    ----------
    X: `float` np.array [n_vars * n_cells]  # todo make it be n_cells * n_vars
        Variant call matrix. Contains VAF of variant for that cell / position.
    weights: `float` np.array [n_cells * n_vars]
        Weight matrix. Should reflect the confidence of our variant calls.
    k: `int` (default: 2)
        Number of clusters
    max_cycles: `int` (default: 1000)
        Number of cycles to fit C and V before break.
    parallel: `bool` (default: True)
        Whether to fit rows of C and columns of V in parallel for runtime speed-up.
    n_jobs: `int` (default: None)
        How many cores to run on in parallel. Per default we set it to the number of available CPUs.

    Returns
    -------
    C: `float` np.array [n_cells * k]
        Final cell weights for weighted NMF.
    V: `float` np.array [k * n_vars]
        Final variant weights for weighted NMF.
    """
    M, weights = adata.layers["M"], adata.layers["weights"]
    # init
    bounds = np.repeat([[0, 1]], k, 0)  # bounds of C and V matrices
    v_entries, c_entries = M.shape[1], M.shape[0]
    # initialize C and V with random values btw 0 and 1:
    V = np.random.rand(k, v_entries)
    C = np.random.rand(c_entries, k)
    if parallel:
        # we send blocks of size 100 to each core.
        # Note that too small blocks will impede runtime because of send & fetch delays.
        n_parts_V = np.ceil(v_entries / 100)
        n_parts_C = np.ceil(c_entries / 100)
    import time
    start = time.time()

    for i in range(max_cycles):
        if i % 10 == 0:
            print(i)
            end = time.time()
            hours, rem = divmod(end - start, 3600)
            minutes, seconds = divmod(rem, 60)
            print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
            start = time.time()

        #########
        # fit V #
        #########
        def fun_v(v, h):  # helper function for minimize call
            return np.sum(((np.dot(C, v) - M[:, h]) ** 2) * weights[:, h])

        if parallel:
            V = parallel_run(V, v_entries, n_parts_V, fun_v, bounds, n_jobs=n_jobs, axis=1).T
        else:
            for j in np.arange(0, M.shape[1]).tolist():
                V[:, j] = op.minimize(fun_v, V[:, j], bounds=bounds, args=j).x

        #########
        # fit C #
        #########
        def fun_c(c, h):  # helper function for minimize call
            return np.sum(((np.dot(c, V) - M[h]) ** 2) * weights[h])

        if parallel:
            C = parallel_run(C, c_entries, n_parts_C, fun_c, bounds, n_jobs=n_jobs, axis=0)
        else:
            for j in np.arange(0, M.shape[0]).tolist():
                C[j] = op.minimize(fun_c, C[j], bounds=bounds, args=j).x

    return C, V


def parallel_run(in_arr, n_entries, n_parts, fun, bounds, n_jobs, axis=0):
    """
    Helper function to run minimize in parallel. This is needed to format, send and fetch the running block to the CPU.
    """
    # prepare input
    pars = []
    for j in range(n_entries):
        pars.append((in_arr[j] if axis == 0 else in_arr[:, j], j))

    # run fitting
    def fit_helper(pars_):
        out = []
        for p in pars_:
            out.append(op.minimize(fun, p[0], bounds=bounds, args=p[1]).x)
        return out

    n_ = int(np.ceil(n_entries / n_parts))
    result = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(fit_helper)(pars[i:i + n_]) for i in range(0, n_entries, n_))

    # format result
    r = []
    for i in result:
        r.extend(i)
    return np.array(r)


def orth_score(C):
    k = C.shape[1]
    sc = 0
    n = 0
    for i in range(k):
        for j in np.arange(i + 1, k):
            n += 1
            sc += np.dot(C[:, i], C[:, j]) / (np.linalg.norm(C[:, i]) * np.linalg.norm(C[:, j]))
    return sc / n


def bootstrap_wNMF(adata, k, n_bootstrap=50, bootstrap_percent=.9,
                   max_cycles=100, parallel=True, n_jobs=-1, mode=""):
    """
    Bootstrap function for the wNMF. We are not guaranteed to find a global minimum for the wNMF and the output
    will change over multiple iteration. To get an estimation of the robustnes we run the wNMF multiple times using
    a subset of the variants as input. We then get n_bootstrap C and V matrices, align these and return the mean of
    the runs as cell weights. We also return the std of the runs to get an estimation of the robustness of these
    assignments.

    Parameters
    ----------
    adata: Anndata object
        Containing REF and ALT matrices in adata.uns
    k: `int` (default: 2)
        Number of clusters
    n_bootstrap: `int` (default: 10)
        Number of boostrap runs.
    bootstrap_percent: `float` (default: 0.9)
        Percentage of variants kept in each of the bootstrap iterations.
    max_cycles: `int` (default: 1000)
        Number of cycles to fit C and V before break.
    parallel: `bool` (default: True)
        Whether to fit rows of C and columns of V in parallel for runtime speed-up.
    n_jobs: `int` (default: None)
        How many cores to run on in parallel. Per default we set it to the number of available CPUs.
    mode: `str` in ['', '10X'] (default: '')
        Whether to use the SmartSeq2 model or the 10X one (0 - 0.5 - 1).
   Returns
    -------

    """
    n_vars = adata.shape[1]
    C_all, V_all = [], []
    if np.any([i not in list(adata.uns.keys()) for i in ["M", "weights"]]):
        adata = get_wNMF_matrices(adata, mode=mode)

    ######################
    # run bootstrap wNMF #
    ######################
    for i in range(n_bootstrap):
        # subset to bootstrap_percent% of variants
        idx = np.random.choice(np.arange(n_vars), int(n_vars * bootstrap_percent), replace=False)
        adata_sub = adata[:, idx]
        # run wNMF
        C, V = NMF_weighted(adata_sub, k=k, max_cycles=max_cycles, parallel=parallel, n_jobs=n_jobs)
        V_ = np.zeros(shape=(k, n_vars)) * np.nan  # weights of excluded variants set to nan
        V_[:, idx] = V
        C_all.append(C), V_all.append(V_)
    C_all, V_all = np.array(C_all), np.array(V_all)

    #######################
    # align the wNMF runs #
    #######################
    # We align all runs to maximize simlarity to the first one. # todo maybe compare all to all?
    # Note: this could cause issues if the first run is the only one that didn't work. However, assuming the wNMF works
    # reasonably well for that patient it should be ok. And if the results are untrustworthy we assume the method fails.
    clus1 = np.argmax(C_all[0], axis=1)
    perm = list(permutations(np.arange(0, k)))
    for i in np.arange(0, n_bootstrap):
        d = np.zeros((k, k))
        clus2 = np.argmax(C_all[i], axis=1)
        for j in range(k):
            for h in range(k):
                d[j, h] = (np.sum((clus1 == j) & (clus2 == h)))
        p = [perm[np.argmax(np.sum(d[np.arange(0, k), perm], axis=1))]]
        C_all[i] = C_all[i].T[p][0].T
        V_all[i] = V_all[i][p][0]  # this part does not work give the same output in older versions of numpy
        # that's why we have numpy~=1.25.0 in the requirements

    ######################
    #     get output     #
    ######################
    if n_bootstrap > 1:
        C, C_std = np.mean(np.array(C_all), axis=0), np.nanstd(C_all, axis=0)
        V, V_std = np.nanmean(V_all, axis=0), np.nanstd(V_all, axis=0)
        adata.varm["V"], adata.varm["V_std"] = V.T, V_std.T
        adata.obsm["C"], adata.obsm["C_std"] = C, C_std

    else:
        C, C_std = C_all[0], None
        V, V_std = V_all[0], None
        adata.varm["V"], adata.obsm["C"] = V.T, C

    adata.uns["k"] = k
    adata.uns["orth_score"] = orth_score(C) if k > 1 else np.nan
    return adata
