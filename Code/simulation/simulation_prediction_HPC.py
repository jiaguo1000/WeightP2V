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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

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

def cos_similarity(V_p, V_d):
    nmrt = sum(V_p * V_d)
    dnmt = np.sqrt(sum(V_p**2)) * np.sqrt(sum(V_d**2))
    return(nmrt / dnmt)

def get_pvec(event_list, model, t, corr, beta, gamma):
    nmrt = np.array(0)
    dnmt = np.array(0)
    for i, event in enumerate(event_list):
        vect = model.wv[event].astype("float64")
        parm = (abs(corr[i])**gamma) * np.exp(-beta*t[i])
        nmrt = nmrt + vect*parm
        dnmt = dnmt + parm
    return(nmrt, dnmt)

def cv_each_fold(cv_i, n, case_p, data_01_df, data_seq_df, vec_size, win_size, n_epoch, negative):
    n_tran_case = int(n * case_p)
    n_tran_ctrl = int(n * (1-case_p))
    n_test_case = int(n * case_p)
    n_test_ctrl = int(n * (1-case_p))

    case_subj = random.sample(data_01_df.loc[data_01_df.response==1,"subj_ID"].tolist(), n_tran_case)
    ctrl_subj = random.sample(data_01_df.loc[data_01_df.response==0,"subj_ID"].tolist(), n_tran_ctrl)
    tran_ID = random.sample(case_subj+ctrl_subj, len(case_subj+ctrl_subj))

    case_subj = random.sample(
        [x for x in data_01_df.loc[data_01_df.response==1,"subj_ID"] if x not in tran_ID],
        n_test_case
        )
    ctrl_subj = random.sample(
        [x for x in data_01_df.loc[data_01_df.response==0,"subj_ID"] if x not in tran_ID],
        n_test_ctrl
        )
    test_ID = random.sample(case_subj+ctrl_subj, len(case_subj+ctrl_subj))

    # seq data
    data_tran_seq = pd.DataFrame({"subj_ID":tran_ID})
    data_tran_seq = pd.merge(data_tran_seq, data_seq_df)

    data_test_seq = pd.DataFrame({"subj_ID":test_ID})
    data_test_seq = pd.merge(data_test_seq, data_seq_df)

    # 01 data
    data_tran_01 = pd.DataFrame({"subj_ID":tran_ID})
    data_tran_01 = pd.merge(data_tran_01, data_01_df)
    y_tran = data_tran_01["response"].tolist()
    data_tran_01 = data_tran_01.drop(columns=["subj_ID", "response"])

    data_test_01 = pd.DataFrame({"subj_ID":test_ID})
    data_test_01 = pd.merge(data_test_01, data_01_df)
    y_test = data_test_01["response"].tolist()
    data_test_01 = data_test_01.drop(columns=["subj_ID", "response"])

    # train word2vec
    data_tran_seq["seq"] = data_tran_seq.apply(lambda x: random.sample(x.event, len(x.event)), axis=1)
    sentences = MyCorpus(data_tran_seq.seq, random = True)

    model = Word2Vec(
        sentences=sentences, seed=2021, workers=8, alpha=0.025,
        vector_size=vec_size, window=win_size, min_count=1,
        sg=0, hs=0, negative=negative, epochs=n_epoch
        )
    V_d = model.wv["response"].astype("float64")

    # seq predict
    data_test_seq["time"] = data_test_seq.apply(
        lambda x: np.repeat(1, len(x.event)),
        axis=1
        )
    data_test_seq["correlation"] = data_test_seq.apply(
        lambda x: [model.wv.similarity(j, "response").astype("float64") for j in x["event"]],
        axis=1
        )
    PV_AUC = []
    for g in [0,2]:
        # test data
        data_test_seq["temp"] = data_test_seq.apply(
            lambda x: get_pvec(
                [k for k in x["event"] if k!="response"], model, t=x["time"], corr=x["correlation"],
                beta=0, gamma=g
                ),
            axis=1
            )
        data_test_seq["nmrt"] = [x[0] for x in data_test_seq["temp"]]
        data_test_seq["dnmt"] = [x[1] for x in data_test_seq["temp"]]
        data_test_seq["pvec"] = data_test_seq.nmrt/data_test_seq.dnmt
        data_test_seq["pred"] = data_test_seq.apply(
            lambda x: cos_similarity(x.pvec, V_d),
            axis=1
        )
        PV_AUC.append(roc_auc_score(y_test, data_test_seq["pred"]))

    # RF and LR
    LG_model = LogisticRegression(random_state=0, penalty="l1", solver="liblinear").fit(data_tran_01, y_tran)
    LG_AUC = roc_auc_score(y_test, LG_model.predict_proba(data_test_01)[:,1])
    RF_model = RandomForestClassifier(random_state=0, n_estimators=500).fit(data_tran_01, y_tran)
    RF_AUC = roc_auc_score(y_test, RF_model.predict_proba(data_test_01)[:,1])
    
    res = {"PV_0": [PV_AUC[0]], "PV_2": [PV_AUC[1]]}
    res = {**{"LG":[LG_AUC], "RF_500":[RF_AUC]}, **res}
    return(res)

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
rate_1 = 0.5
n_all = 10000*2
all_ID = ["subj_"+str(i) for i in range(n_all)]
fea_cov = nearestPD(fea_cov)
data_X = np.random.multivariate_normal(fea_mean, fea_cov, n_all)
for i in range(data_X.shape[1]):
    data_X[:,i] = get_01(data_X[:,i], rate_1)
corr_01 = np.corrcoef(data_X, rowvar=False)

data_X = data_X[np.sum(data_X, 1)!=0,:]

beta_list = list(np.arange(2,11)/10)
case_p_list = [0.5, 0.3, 0.1, 0.05]

""" iter """
output = pd.DataFrame()
for beta_signal in beta_list:
    beta_all = [beta_signal]*n_signal + [0.0]*(n_C_total-n_signal)
    intercept = -beta_signal*rate_1*n_signal

    """ population outcome """
    y = np.matmul(
        np.concatenate([np.ones((n_all,1)), data_X], axis=1),
        np.array([intercept]+beta_all)
        )
    y = 1 / (1 + np.exp(-y))
    y = np.array([np.random.binomial(1, p) for p in y])
    Counter(y)
    data_01_Xy = np.concatenate([np.reshape(y, (-1,1)), data_X], axis=1)
    corr_m = np.corrcoef(data_01_Xy, rowvar=False)

    """ seq """
    data_01_df = pd.DataFrame(data_01_Xy)
    data_01_df.columns = [otcm_c] + pred_c
    data_01_df["subj_ID"] = all_ID[:len(data_X)]

    data_seq_df = data_01_df.copy()
    data_seq_df["event"] = data_seq_df.apply(
        lambda x:[c for c in list(data_seq_df.columns) if x[c]==1],
        axis=1
    )
    data_seq_df = data_seq_df[["subj_ID", "event"]]

    """ cv result """
    res_AUC = pd.DataFrame()
    n = 200
    vec_size = 100
    win_size = 500
    n_epoch = 205
    negative = 1
    for case_p in case_p_list:
        for cv_i in range(5):
            prll_res = cv_each_fold(
                cv_i=cv_i, n=n, case_p=case_p, data_01_df=data_01_df, data_seq_df=data_seq_df,
                vec_size=vec_size, win_size=win_size, n_epoch=n_epoch, negative=negative
                )
            temp = pd.DataFrame(prll_res)
            temp["beta"] = beta_signal
            temp["case_p"] = case_p
            res_AUC = pd.concat([res_AUC, temp])

    output = pd.concat([output, res_AUC])


""" save result for each task """
filname = "../../Data/simu_prediction_task_{}".format(
    task_id
)
output.to_csv(filname+".csv", index=False)