#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" @author: jiaguo """

# import Modules
import os
import random
import pandas as pd
import numpy as np
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


def WeightP2V(train_lib, seq_test, disease, win_size=500, vec_size=200, n_epoch=1):
    """
    Returns the cosine similarity between the disease vector and patient vector obtained from WeightP2V

    train_lib: a list of sequences as the training library, e.g., [["ICD_001", "ICD_002"], ["ICD_001", "ICD_002", "ICD_003], ["ICD_005", "ICD_010"]]
    seq_test: a pd.Dataframe of test samples and their all visits before the most recent visit,
              with each row representing a single visit for a patient. Columns are:
                    "SUBJECT_ID": ID of patients
                    "event": a list of medical concepts for each visit, e.g., ["ICD_001", "ICD_002"]
                    "diff_time": a numeric number of time gap (in seconds) between the visit and the most recent visit
    disease: a string of disease, e.g., "ICD_001"
    """
    sentences = MyCorpus(train_lib, random=False)
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
    test_PV = seq_test.drop(columns=["event", "diff_time", "time"])
    test_PV = test_PV.groupby("SUBJECT_ID", sort=False).first().reset_index()

    V_d = wv_model.wv[disease].astype("float64")
    # test_PV["PV_no_time_no_corr"] = test_PV.apply(
    #     lambda x: cos_similarity(x.seq_no_time_no_corr, V_d),
    #     axis=1)
    test_PV["PV_with_time_no_corr"] = test_PV.apply(
        lambda x: cos_similarity(x.seq_with_time_no_corr, V_d),
        axis=1)
    test_PV["PV_with_time_with_corr"] = test_PV.apply(
        lambda x: cos_similarity(x.with_time_with_corr, V_d),
        axis=1)
    test_PV = test_PV[["SUBJECT_ID", "PV_with_time_no_corr", "PV_with_time_with_corr"]]
    return(test_PV)

