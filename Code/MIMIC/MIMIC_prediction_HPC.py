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

from WeightP2V import *

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
    seq_test = seq_test.drop(columns=["time_rank", "outcome"])
    
    seq_test_word2vec = seq_test.copy()
    seq_test_word2vec["event"] = seq_test_word2vec["event"].apply(lambda x: random.sample(x, k=len(x)))
    seq_test_word2vec = seq_test.groupby("SUBJECT_ID").event.agg(sum).reset_index().rename(columns={"event":"seq"})

    train_lib = seq_data_all.loc[~seq_data_all["SUBJECT_ID"].isin(seq_test["SUBJECT_ID"]),]
    train_lib = train_lib.seq.tolist() + seq_test_word2vec.seq.tolist()
    
    test_PV = WeightP2V(train_lib=train_lib, seq_test=seq_test, disease=disease,
                        win_size=500, vec_size=200, n_epoch=1)
    assert sum(test_PV.SUBJECT_ID==pd.DataFrame({"SUBJECT_ID":case_test_ID+ctrl_test_ID}).SUBJECT_ID)==test_PV.shape[0]
    
    """ res AUC """
    res = {
        "LG":LG_AUC,
        "RF_500":RF_AUC,
        # "PV_no_time_no_corr": roc_auc_score(y_test, test_PV["PV_no_time_no_corr"]),
        "PV_with_time_no_corr": roc_auc_score(y_test, test_PV["PV_with_time_no_corr"]),
        "PV_with_time_with_corr": roc_auc_score(y_test, test_PV["PV_with_time_with_corr"])
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
