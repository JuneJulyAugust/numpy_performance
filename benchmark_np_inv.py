import time
import random
import scipy.linalg as scipy_linalg
import ad.linalg as ad_linalg
import numpy as np
import matplotlib.pyplot as plt

def lu_inv(A):
    L = np.linalg.cholesky(A)                                                                                                              
    L_inv = np.linalg.inv(L)
    A_inv_lu = np.dot(L_inv.T, L_inv)
    return A_inv_lu

def bench_lu_inv(m, niter):
    ts_list = []

    for i in xrange(niter):
        tic = time.clock()
        m_inv = lu_inv(m)
        toc = time.clock() 
        ts_list.append(toc - tic)
    
    ts_list = np.array(ts_list)
    print("{}, mean: {}, max: {}, std: {}".format("lu inverse", np.mean(ts_list), np.max(ts_list), np.std(ts_list)))
     
    plt.plot(ts_list, label="lu inverse")

    return np.array(ts_list)



def bench_ad_inv(m, niter):
    ts_list = []

    for i in xrange(niter):
        tic = time.clock()
        m_inv = ad_linalg.inv(m)
        toc = time.clock() 
        ts_list.append(toc - tic)
    
    ts_list = np.array(ts_list)
    print("{}, mean: {}, max: {}, std: {}".format("ad inverse", np.mean(ts_list), np.max(ts_list), np.std(ts_list)))
     
    plt.plot(ts_list, label="ad inverse")

    return np.array(ts_list)

def bench_numpy_pinv(m, niter):
    ts_list = []

    for i in xrange(niter):
        tic = time.clock()
        m_inv = np.linalg.pinv(m)
        toc = time.clock() 
        ts_list.append(toc - tic)
    
    ts_list = np.array(ts_list)
    print("{}, mean: {}, max: {}, std: {}".format("numpy pinverse", np.mean(ts_list), np.max(ts_list), np.std(ts_list)))
     
    plt.plot(ts_list, label="numpy pinverse")

    return np.array(ts_list)

def bench_scipy_inv(m, niter):
    ts_list = []

    for i in xrange(niter):
        tic = time.clock()
        m_inv = scipy_linalg.inv(m)
        toc = time.clock() 
        ts_list.append(toc - tic)

    ts_list = np.array(ts_list)
    print("{}, mean: {}, max: {}, std: {}".format("scipy inverse", np.mean(ts_list), np.max(ts_list), np.std(ts_list)))
     
    plt.plot(ts_list, label="scipy inverse")
    return np.array(ts_list)

def bench_scipy_pinv(m, niter):
    ts_list = []

    for i in xrange(niter):
        tic = time.clock()
        m_inv = scipy_linalg.pinv(m)
        toc = time.clock() 
        ts_list.append(toc - tic)

    ts_list = np.array(ts_list)
    print("{}, mean: {}, max: {}, std: {}".format("scipy pinverse", np.mean(ts_list), np.max(ts_list), np.std(ts_list)))
     
    plt.plot(ts_list, label="scipy pinverse")
    return np.array(ts_list)

def bench_scipy_pinv2(m, niter):
    ts_list = []

    for i in xrange(niter):
        tic = time.clock()
        m_inv = scipy_linalg.pinv2(m)
        toc = time.clock() 
        ts_list.append(toc - tic)

    ts_list = np.array(ts_list)
    print("{}, mean: {}, max: {}, std: {}".format("scipy pinverse2", np.mean(ts_list), np.max(ts_list), np.std(ts_list)))
     
    plt.plot(ts_list, label="scipy pinverse2")
    return np.array(ts_list)

def bench_numpy_inv(m, niter):
    ts_list = []

    for i in xrange(niter):
        tic = time.clock()
        m_inv = np.linalg.inv(m)
        toc = time.clock() 
        ts_list.append(toc - tic)
    
    ts_list = np.array(ts_list)
    print("{}, mean: {}, max: {}, std: {}".format("numpy inverse", np.mean(ts_list), np.max(ts_list), np.std(ts_list)))
     
    plt.plot(ts_list, label="numpy inverse")

    return np.array(ts_list)

if __name__ == "__main__":
    sk = np.load("./sk.dat") 
    print(sk)
    niters = 200000
    bench_numpy_inv(sk, niters)
    bench_numpy_pinv(sk, niters)
    #bench_scipy_inv(sk, niters)
    #bench_scipy_pinv(sk, niters)
    #bench_scipy_pinv2(sk, niters)
    #bench_lu_inv(sk, niters)
    #bench_ad_inv(sk, niters)

    legend = plt.legend(loc="upper right", shadow=True, fontsize="x-large");
    plt.grid()
    plt.show()

