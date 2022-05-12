#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" @author: jiaguo """

# import Modules
import os
import random
import pandas as pd
import numpy as np
from numpy import linalg as la
from collections import Counter
from gensim.models import Word2Vec

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""
    def __init__(self, sents_list, random):
        self.sents_list = sents_list
        self.random = random

    def __iter__(self):
        for i, s in enumerate(self.sents_list):
            if self.random:
                s = random.sample(s, k=len(s)) # reorder
            yield(s)

def get_01(x, p):
    q = np.quantile(x, p)
    y = [1 if i<q else 0 for i in x]
    return(y)

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

def nearestPD(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    B = (A + A.T) / 2
    _, s, V = la.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if isPD(A3):
        return A3
    spacing = np.spacing(la.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
    return A3

task_id = int(os.getenv("SGE_TASK_ID"))
print(task_id, flush=True)

""" parameters """
n_signal = 10
n_noise = 140
n_C_total = n_signal+n_noise
pred_c = ["p_"+str(x) for x in range(1, n_C_total+1)]
otcm_c = "response"
rate_1 = 0.5

np.random.seed(task_id)
random.seed(task_id)

""" correlated X """
fea_mean = np.repeat(0, n_C_total)

r0 = 0.6
r1 = 0.05
m11 = np.full((n_signal, n_signal), r0)
m12 = np.full((n_signal, n_noise), r1)

m21 = np.full((n_noise, n_signal), r1)
m22 = np.full((n_noise, n_noise), r1)

m1 = np.concatenate([m11, m12], axis=1)
m2 = np.concatenate([m21, m22], axis=1)

fea_cov = np.concatenate([m1, m2])
np.fill_diagonal(fea_cov, 1)

""" X and y """
n_all = 10000*2
all_ID = ["subj_"+str(i) for i in range(n_all)]
fea_cov = nearestPD(fea_cov)
data_X = np.random.multivariate_normal(fea_mean, fea_cov, n_all)
for i in range(data_X.shape[1]):
    data_X[:,i] = get_01(data_X[:,i], rate_1)
corr_01 = np.corrcoef(data_X, rowvar=False)

""" get cosine """
beta_list = list(np.arange(1,11)/10)
beta_list = [-i for i in beta_list[::-1]] + [0] + beta_list
output = pd.DataFrame()
for beta_signal in beta_list:
    beta_all = [beta_signal]*n_signal + [0.0]*(n_C_total-n_signal)
    intercept = -beta_signal*rate_1*n_signal
    y = np.matmul(
        np.concatenate([np.ones((n_all,1)), data_X], axis=1),
        np.array([intercept]+beta_all)
        )
    y = 1 / (1 + np.exp(-y))
    y = np.array([np.random.binomial(1, p) for p in y])
    Counter(y)

    data_01_Xy = np.concatenate([np.reshape(y, (-1,1)), data_X], axis=1)
    corr_m = np.corrcoef(data_01_Xy, rowvar=False)
    print(
        "beta: {} x-y: {:.3f} other-y: {:.3f} other-x: {:.3f}".format(
        beta_signal, corr_m[0,1], corr_m[0,2], corr_m[1,2])
        )

    """ seq """
    data_01_df = pd.DataFrame(data_01_Xy)
    data_01_df.columns = [otcm_c] + pred_c
    data_01_df["event"] = data_01_df.apply(
        lambda x:[c for c in list(data_01_df.columns) if x[c]==1],
        axis=1
    )
    data_01_df["subj_ID"] = all_ID
    data_01_df = data_01_df[["subj_ID", "event"]].copy()

    """ bootstrap """
    res = pd.DataFrame()
    for i in range(5):
        idx = random.sample(list(range(data_01_df.shape[0])), 200)
        data_tran_seq = data_01_df.iloc[idx,:].copy()
        data_tran_seq["seq"] = data_tran_seq.apply(lambda x: random.sample(x.event, len(x.event)), axis=1)

        """ train word2vec """
        sentences = MyCorpus(data_tran_seq.seq, random = True)
        vec_size = 100
        n_epoch = 205
        win_size = 500
        negative = 1
        model = Word2Vec(
            sentences=sentences, seed=2021, workers=8, alpha=0.025,
            vector_size=vec_size, window=win_size, min_count=1,
            sg=0, hs=0, negative=negative, epochs=n_epoch
            )

        corr = {x[0]:x[1] for x in model.wv.most_similar("response", topn=len(model.wv.key_to_index))}
        corr = pd.DataFrame(corr.items(), columns=["ICD", "corr"])
        res = pd.concat([res, corr])

    """ results """
    k = n_signal+1
    def add_group(x):
        return(
            np.select(
                [x in ["p_"+str(i) for i in range(1, k)],
                x in ["p_"+str(i) for i in range(k, n_C_total+1)]],
                ["signal", "noise"])
        )
    res["group"] = res["ICD"].apply(add_group)
    res["beta"] = beta_signal
    output = pd.concat([output, res])


""" save result for each task """
filname = "../../Data/simu_cosine_task_{}".format(
    task_id
)
output.to_csv(filname+".csv", index=False)