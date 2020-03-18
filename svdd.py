from collections import namedtuple
from sklearn.cluster import KMeans
import numpy as np
from numpy.random import choice
from sklearn import svm
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import matplotlib.font_manager
import time
from tqdm import tqdm

def _compute_radius_center(clf, method=1):
    sv, coef = clf.support_vectors_, clf.dual_coef_
    sv_pos = np.where((coef < 1)[0, ...])[0]
    coef.shape = (coef.shape[1], )
    coef = coef/np.sum(coef)
    center = np.dot(coef, sv)
    #method 1 is a fast approximation of the radius which is good enough for our purpose
    if method == 0:
        m = rbf_kernel(sv, sv, gamma=clf.gamma)
        radius = 1 - 2 * np.dot(m[sv_pos[0], ...], coef) + np.dot(coef, np.dot(m, coef))
    else:
        v = sv[sv_pos[0], ...].reshape(1, sv.shape[1])
        m = rbf_kernel(v, sv, gamma=clf.gamma)
        radius = 1 - np.dot(m, coef)
    return radius, center

def _do_one_class_svm_sample(gamma, nu, x_train, sample_indices, compute_rc=True):
    x_train_sample = x_train[sample_indices, ...]
    nsample = x_train_sample.shape[0]
    nu_1 = nu  if nu * nsample > 1 else 1/nsample
    clf = svm.OneClassSVM(gamma=gamma, nu=nu_1)
    clf.fit(x_train_sample)
    if compute_rc:
        radius, center = _compute_radius_center(clf)
        #print('radius = \n', radius)
        #print('center = \n', center)
        return sample_indices[clf.support_], radius, center
    else:
        return sample_indices[clf.support_]
    
# draw a random sample from the original data and peform svdd on it
def _do_one_class_svm_random(gamma, nu, x_train, sample_size, compute_rc=True):
    sample = choice(x_train.shape[0], sample_size)
    return _do_one_class_svm_sample(gamma, nu, x_train, sample, compute_rc=compute_rc)

# the sampling svdd implementation, see the __main__ section for an example
def sample_svdd(x_train,
                outlier_fraction=0.0001,
                kernel_s=1.3,
                maxiter=5000,
                sample_size=10,
                resample_n=3,
                stop_tol=1e-6,
                n_iter=30,
                iter_history=True,
                seed=2513646):
    #sanity checks
    if maxiter <= 0:
        print("ERROR: maxiter must be positive integer")
        raise ValueError
        
    nobs = x_train.shape[0]
    if nobs <= sample_size:
        print("ERROR: sample size must be strictly smaller than number of observations in input data")
        raise ValueError

    gamma, nu = 0.5 / (kernel_s * kernel_s), outlier_fraction

    if np.isfinite(gamma) != True or np.isfinite(nu) != True or (nu < 0) or (nu > 1):
        print("ERROR: Invalid kernel_s or outlier_fraction input")
        raise ValueError

    #if negative seed is provided use a system chosen seed
    np.random.seed(seed=seed if seed >= 0 else None)

    if iter_history:
        radius_history, center_history = np.empty(maxiter+1), list()

    clf = None

    sv_ind_prev, radius_prev, center_prev = _do_one_class_svm_random(gamma, nu, x_train, sample_size)

    if iter_history:
        radius_history[0] = radius_prev
        center_history.append(center_prev)

    i, converged, iter_n = 0, 0, 0
    while i < maxiter:
        if converged: break
        
        sv_ind_local = _do_one_class_svm_random(gamma, nu, x_train, sample_size, compute_rc=False)
        for dummy1 in range(resample_n-1):
            sv_ind_locals = _do_one_class_svm_random(gamma, nu, x_train, sample_size, compute_rc=False)
            sv_ind_local = np.union1d(sv_ind_locals, sv_ind_local)
        sv_ind_merge = np.union1d(sv_ind_local, sv_ind_prev)
        sv_ind_master, radius_master, center_master = _do_one_class_svm_sample(gamma, nu, x_train, sv_ind_merge)


        if iter_history:
            radius_history[i+1] = radius_master
            center_history.append(center_master)

        iter_n = iter_n + 1 if np.fabs(radius_master - radius_prev) <= stop_tol * np.fabs(radius_prev) else 0
        if iter_n >= n_iter:
            converged = 1
        else:
            sv_ind_prev, center_prev, radius_prev = sv_ind_master, center_master, radius_master
        i += 1

    if iter_history:
        radius_history = radius_history[0:i+1]
    niter = i + 1

    SampleSVDDRes      = namedtuple("SampleSVDDRes", "Params  IterHist OneClassSVM")
    SampleSVDDParams   = namedtuple("SampleSVDDParams", "sv_ center_ radius_")
    SampleSVDDIterHist = namedtuple("SampleSVDDIterHist", "niter_ radius_history_ center_history_ converged_")

    params = SampleSVDDParams(sv_ind_master, center_master, radius_master)
    
    print('center of each SVDD : ',center_master)
    print('best radius of each SVDD : ',radius_master)
    iterhist = None
    if iter_history:
        iterhist = SampleSVDDIterHist(niter, radius_history, center_history, converged)

    nsv = sv_ind_master.shape[0]
    print('number support vector: ', nsv)
    clf = svm.OneClassSVM(gamma=gamma, nu=nu if nu * nsv > 1 else 1./nsv)
    clf.fit(x_train[sv_ind_master, ...])

    return SampleSVDDRes(params, iterhist, clf), center_master, radius_master

def train_svdd(data_train, outlier_fraction=0.001, kernel_s=5.0):
    # Parameter
    seed = 24215125
    np.random.seed(seed)
    sample_size, resample_n, n_iter = 10, 1, 10
    stop_tol, maxiter = 1e-4, 5000 
    
    result, center, R = sample_svdd(data_train,
                                 outlier_fraction=outlier_fraction, 
                                 kernel_s = kernel_s, 
                                 resample_n = resample_n, 
                                 maxiter = maxiter, 
                                 sample_size = sample_size, 
                                 stop_tol = stop_tol, 
                                 n_iter = n_iter, 
                                 iter_history = True, 
                                 seed=seed)
    clf = result.OneClassSVM
    
    return clf

