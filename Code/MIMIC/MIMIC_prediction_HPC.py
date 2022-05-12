#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" @author: jiaguo """

# import Modules
import os
import compress_pickle
import random
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

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

# with time, with corr
def get_corr_each_D(d, work_data, model, beta, gamma):
    work_data_temp = work_data.copy().loc[:,["SUBJECT_ID", "event", "time"]]
    work_data_temp["correlation"] = work_data_temp.apply(
        lambda x: [model.wv.similarity(j, d).astype("float64") for j in x["event"]],
        axis=1
        )

    work_data_temp["temp"] = work_data_temp.apply(
        lambda x: get_pvec(x["event"], model, t=x["time"], corr=x["correlation"], beta=beta, gamma=gamma),
        axis=1
        )
    work_data_temp["nmrt"] = [x[0] for x in work_data_temp["temp"]]
    work_data_temp["dnmt"] = [x[1] for x in work_data_temp["temp"]]

    temp = work_data_temp.groupby("SUBJECT_ID").nmrt.apply(lambda x: sum(x)).reset_index().rename(columns={"nmrt":"nmrt_all"})
    work_data_temp = pd.merge(work_data_temp, temp, how="left")

    temp = work_data_temp.groupby("SUBJECT_ID").dnmt.apply(lambda x: sum(x)).reset_index().rename(columns={"dnmt":"dnmt_all"})
    work_data_temp = pd.merge(work_data_temp, temp, how="left")

    work_data_temp["with_time_with_corr"] = work_data_temp.nmrt_all/work_data_temp.dnmt_all
    work_data_temp = work_data_temp.loc[:,["SUBJECT_ID", "with_time_with_corr"]]
    return(work_data_temp)

def cv_each_D(
    cv_i, case_ID, ctrl_ID, case_tran_test, ctrl_tran_test, count_data, seq_data, seq_data_all
    ):
    """ tran test ID """
    case_tran_ID = [case_ID[i] for i in case_tran_test[cv_i][0]]
    case_test_ID = [case_ID[i] for i in case_tran_test[cv_i][1]]

    ctrl_tran_ID = [ctrl_ID[i] for i in ctrl_tran_test[cv_i][0]]
    ctrl_test_ID = [ctrl_ID[i] for i in ctrl_tran_test[cv_i][1]]

    """ LR and RF """
    X_tran = pd.DataFrame({"SUBJECT_ID":case_tran_ID+ctrl_tran_ID})
    X_tran = pd.merge(X_tran, count_data, how="left").drop(columns=["SUBJECT_ID"])
    X_test = pd.DataFrame({"SUBJECT_ID":case_test_ID+ctrl_test_ID})
    X_test = pd.merge(X_test, count_data, how="left").drop(columns=["SUBJECT_ID"])
    assert X_tran.shape[0]+X_test.shape[0] == count_data.shape[0]

    y_tran = [1]*len(case_tran_ID) + [0]*len(ctrl_tran_ID)
    y_test = [1]*len(case_test_ID) + [0]*len(ctrl_test_ID)

    LG_model = LogisticRegression(random_state=0, penalty="l1", solver="liblinear").fit(X_tran, y_tran)
    LG_AUC = roc_auc_score(y_test, LG_model.predict_proba(X_test)[:,1])
    RF_model = RandomForestClassifier(random_state=0, n_estimators=500).fit(X_tran, y_tran)
    RF_AUC = roc_auc_score(y_test, RF_model.predict_proba(X_test)[:,1])

    """ seq prediction """
    """ train word2vec """
    seq_tran = pd.DataFrame({"SUBJECT_ID":case_tran_ID+ctrl_tran_ID})
    seq_tran = pd.merge(seq_tran, seq_data, how="left")
    seq_test = pd.DataFrame({"SUBJECT_ID":case_test_ID+ctrl_test_ID})
    seq_test = pd.merge(seq_test, seq_data, how="left")
    assert seq_tran.shape[0]+seq_test.shape[0] == seq_data.shape[0]
    
    seq_test_word2vec = seq_test.copy()
    seq_test_word2vec["event"] = seq_test_word2vec["event"].apply(lambda x: random.sample(x, k=len(x)))
    seq_test_word2vec = seq_test.groupby("SUBJECT_ID").event.agg(sum).reset_index().rename(columns={"event":"seq"})

    train_lib = seq_data_all.loc[~seq_data_all["SUBJECT_ID"].isin(seq_test["SUBJECT_ID"]),]
    train_lib = train_lib.seq.tolist() + seq_test_word2vec.seq.tolist()
    sentences = MyCorpus(train_lib, random=False)
    win_size = 500
    vec_size = 200
    n_epoch = 1
    wv_model = Word2Vec(
        sentences=sentences, seed=2021, workers=8, alpha=0.025,
        vector_size=vec_size, window=win_size, min_count=1, 
        sg=0, hs=0, negative=5, epochs=n_epoch
        )

    """ PV, with/without time, no corr """
    seq_test["correlation"] = seq_test.event.apply(lambda x: np.repeat(1, len(x)))
    seq_test["time"] = seq_test.apply(
        lambda x: np.repeat(x.diff_time/24/3600/365, len(x.event)),
        axis=1)
    for time_i in range(2):
        beta = 0 if time_i==0 else 1
        seq_test["temp"] = seq_test.apply(
            lambda x: get_pvec(
                event_list=x["event"], model=wv_model, t=x["time"], corr=x["correlation"],
                beta=beta, gamma=2
                ),
            axis=1
            )
        seq_test["nmrt"] = [x[0] for x in seq_test["temp"]]
        seq_test["dnmt"] = [x[1] for x in seq_test["temp"]]

        temp = seq_test.groupby("SUBJECT_ID").nmrt.apply(lambda x: sum(x)).reset_index().rename(columns={"nmrt":"nmrt_all"})
        seq_test = pd.merge(seq_test, temp, how="left")
        temp = seq_test.groupby("SUBJECT_ID").dnmt.apply(lambda x: sum(x)).reset_index().rename(columns={"dnmt":"dnmt_all"})
        seq_test = pd.merge(seq_test, temp, how="left")

        if time_i==0:
            seq_test["seq_no_time_no_corr"] = seq_test.nmrt_all/seq_test.dnmt_all
        if time_i==1:
            seq_test["seq_with_time_no_corr"] = seq_test.nmrt_all/seq_test.dnmt_all
        seq_test = seq_test.drop(columns=["temp", "nmrt", "dnmt", "nmrt_all", "dnmt_all"])

    """ PV, with time, with corr """
    seq_test = seq_test.drop(columns=["correlation"])
    temp = get_corr_each_D(d=disease, work_data=seq_test, model=wv_model, beta=1, gamma=2)
    assert sum(seq_test.SUBJECT_ID == temp.SUBJECT_ID)==seq_test.shape[0]
    seq_test["with_time_with_corr"] = temp["with_time_with_corr"]

    """ prediction using PV """
    test_PV = seq_test.drop(columns=["event", "time_rank", "diff_time", "time"])
    test_PV = test_PV.groupby("SUBJECT_ID", sort=False).first().reset_index()
    test_PV["y_true"] = test_PV["SUBJECT_ID"].apply(lambda x: 1 if x in case_test_ID else 0)
    assert sum(test_PV["y_true"] == y_test)==len(y_test)

    V_d = wv_model.wv[disease].astype("float64")
    test_PV["temp1"] = test_PV.apply(
        lambda x: cos_similarity(x.seq_no_time_no_corr, V_d),
        axis=1)
    test_PV["temp2"] = test_PV.apply(
        lambda x: cos_similarity(x.seq_with_time_no_corr, V_d),
        axis=1)
    test_PV["temp3"] = test_PV.apply(
        lambda x: cos_similarity(x.with_time_with_corr, V_d),
        axis=1)

    """ res AUC """
    res = {
        "LG":LG_AUC,
        "RF_500":RF_AUC,
        "PV_no_time_no_corr": roc_auc_score(y_test, test_PV["temp1"]),
        "PV_with_time_no_corr": roc_auc_score(y_test, test_PV["temp2"]),
        "PV_with_time_with_corr": roc_auc_score(y_test, test_PV["temp3"])
        }
    return(res)

task_id = int(os.getenv("SGE_TASK_ID"))
np.random.seed(task_id)
random.seed(task_id)

""" read data """
seq_data_all = compress_pickle.load("../../Data/MIMIC_seq_data_all.gz")
seq_data_all["event"] = seq_data_all["event"].apply(lambda x: random.sample(x, k=len(x)))
max(seq_data_all["event"].apply(lambda x:len(x))) #476
seq_data_all = seq_data_all.groupby("SUBJECT_ID").event.agg(sum).reset_index().rename(columns={"event":"seq"}) #46520

seq_data = compress_pickle.load("../../Data/MIMIC_predictor_outcome_seq.gz")
count_data = compress_pickle.load("../../Data/MIMIC_predictor_count.gz") #7519

outcome_data = seq_data.loc[:,["SUBJECT_ID", "outcome"]]
outcome_data = outcome_data.groupby("SUBJECT_ID").first().reset_index()
sum(outcome_data["SUBJECT_ID"]==count_data["SUBJECT_ID"])

outcome_list = pd.read_csv("../../Data/MIMIC_outcome_prevalence.csv")
outcome_list = outcome_list.ICD.tolist()

""" each disease """
i = task_id-1
disease = outcome_list[i]
curr_outcome = outcome_data.copy()
curr_outcome["y"] = curr_outcome["outcome"].apply(lambda x: int(disease in x))
assert sum(curr_outcome["SUBJECT_ID"]==count_data["SUBJECT_ID"])==count_data.shape[0], "Error!"

case_ID = curr_outcome.loc[curr_outcome["y"]==1,"SUBJECT_ID"].tolist()
ctrl_ID = curr_outcome.loc[curr_outcome["y"]==0,"SUBJECT_ID"].tolist()
n_case = len(case_ID)
n_ctrl = len(ctrl_ID)
assert n_case+n_ctrl==count_data.shape[0], "Error!"
print("{} | disease = {} | num_case = {} | num_ctrl = {}".format(i, disease, n_case, n_ctrl), flush=True)

""" half/half split 10 times """
i_rep = 10
case_tran_test = []
ctrl_tran_test = []
for seed in range(i_rep):
    kf = KFold(n_splits=2, shuffle=True, random_state=seed)
    case_tran_test.append(list(kf.split(case_ID))[0])
    ctrl_tran_test.append(list(kf.split(ctrl_ID))[0])

""" AUC and then combine """
for cv_i in range(i_rep):
    prll_res = cv_each_D(
        cv_i=cv_i, case_ID=case_ID, ctrl_ID=ctrl_ID,
        case_tran_test=case_tran_test, ctrl_tran_test=ctrl_tran_test,
        count_data=count_data, seq_data=seq_data, seq_data_all=seq_data_all
    )
    temp = pd.DataFrame(prll_res, index=[cv_i])
    res_AUC = temp if cv_i==0 else pd.concat([res_AUC, temp])

""" after combine """
output = res_AUC.copy()
output["ICD"] = disease
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]

""" save result for each task """
filename = "../../Data/MIMIC_task_{}.csv".format(
    task_id
)
output.to_csv(filename, index=False)
