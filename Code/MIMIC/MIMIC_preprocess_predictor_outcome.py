#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jiaguo
"""

""" import Modules """
import compress_pickle
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

""" read seq data """
work_data = compress_pickle.load("../../Data/MIMIC_seq_data_all.gz")
subj_count = work_data.groupby('SUBJECT_ID').apply(lambda x: len(x)).reset_index()
subj_count = subj_count.loc[subj_count.iloc[:,1]>1] #7519 unique subjects
work_data = work_data.loc[work_data.SUBJECT_ID.isin(subj_count.SUBJECT_ID)]
work_data.columns

""" most recent visit as outcome """
work_data.ADMITTIME = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in list(work_data.ADMITTIME)]
work_data['time_rank'] = work_data.groupby(['SUBJECT_ID']).ADMITTIME.rank()
work_data = work_data.sort_values(['SUBJECT_ID', 'time_rank']).reset_index(drop=True)

idx = list(work_data.groupby(['SUBJECT_ID']).time_rank.idxmax())
temp = work_data.loc[:,['SUBJECT_ID','ADMITTIME']].iloc[idx,:].rename(columns={"ADMITTIME": "outcome_time"})
work_data = pd.merge(work_data, temp, how='left')

idx = list(work_data.groupby(['SUBJECT_ID']).time_rank.idxmax())
temp = work_data.loc[:,['SUBJECT_ID','event']].iloc[idx,:].rename(columns={"event": "outcome"})
work_data = pd.merge(work_data, temp, how='left') #19950

""" get the time gap """
work_data['diff_time'] = work_data.outcome_time - work_data.ADMITTIME
work_data['diff_time'] = work_data.diff_time.astype('timedelta64[s]').astype('int')
work_data = work_data.loc[work_data['diff_time']!=0,:] #12431
work_data = work_data.drop(columns=['HADM_ID', 'ADMITTIME', 'outcome_time'])
work_data.columns

""" save seq data of predictors and outcomes """
compress_pickle.dump(
    work_data,
    '../../Data/MIMIC_predictor_outcome_seq.gz'
    )


""" count data of predictor, for regression and random forest """
predictor = work_data.groupby('SUBJECT_ID').event.sum().reset_index()
uniq_subj = predictor.SUBJECT_ID #7519
all_predictor = list(set([x for sublist in predictor.event for x in sublist]))

# first test subject
predictor_count = dict.fromkeys(all_predictor, 0)
predictor_count['SUBJECT_ID'] = 0
predictor_count = pd.DataFrame(predictor_count, index=[0])

# for all subject
for i, subj in enumerate(tqdm(uniq_subj)):
    temp = dict.fromkeys(all_predictor, 0)
    x = predictor.loc[predictor.SUBJECT_ID==subj,'event'].tolist()[0]
    temp_dict = {y_i: x.count(y_i) for y_i in set(x)}

    count_all = {k:temp.get(k, 0) + temp_dict.get(k, 0) for k in set(temp)}
    count_all['SUBJECT_ID'] = subj
    count_all = pd.DataFrame(count_all, index=[i+1])
    predictor_count = predictor_count.append(count_all)

# subject ID first column, remove the first test subject
cols = predictor_count.columns.tolist()
cols = cols[-1:] + cols[:-1]
predictor_count = predictor_count[cols]
predictor_count = predictor_count.iloc[1:]
predictor_count.index = predictor_count.index-1

compress_pickle.dump(
    predictor_count,
    '../../Data/MIMIC_predictor_count.gz'
    )


""" below for prevalence of each outcome """
""" outcome count """
outcome_lib = pd.read_csv("../../Data/MIMIC_III/D_ICD_DIAGNOSES.csv").drop(columns=['ROW_ID'])
all_outcome = work_data.groupby('SUBJECT_ID', sort=False).first().reset_index() #7519
all_outcome = [x for sublist in all_outcome.outcome for x in sublist]
all_outcome = [x for x in all_outcome if x[0:3]=='ICD']
all_outcome = {i: all_outcome.count(i) for i in pd.unique(all_outcome)} #1281
all_outcome = pd.DataFrame(all_outcome.items(), columns = ['ICD9_CODE','num_pts_last_visit'])

""" ever in predictor """
uniq_outcome = list(all_outcome.ICD9_CODE)
count_all_before = dict.fromkeys(uniq_outcome, 0)
count_pts_before = dict.fromkeys(uniq_outcome, 0)

uniq_subj = pd.unique(work_data.SUBJECT_ID)
for subj in tqdm(uniq_subj):
    temp = work_data.loc[work_data.SUBJECT_ID==subj, :]
    x = temp.event.sum()
    y = temp.outcome.iloc[0]

    temp_dict = [y_i for y_i in y if y_i in uniq_outcome]
    temp_dict_1 = {y_i: x.count(y_i) for y_i in temp_dict}
    temp_dict_2 = {y_i: int(v_i>0) for (y_i, v_i) in temp_dict_1.items()}

    count_all_before = {k:count_all_before.get(k, 0) + temp_dict_1.get(k, 0) for k in set(count_all_before)}
    count_pts_before = {k:count_pts_before.get(k, 0) + temp_dict_2.get(k, 0) for k in set(count_pts_before)}

count_all = pd.DataFrame(count_all_before.items(), columns = ['ICD9_CODE','if_last_then_num_seq_prev_visits'])
count_pts = pd.DataFrame(count_pts_before.items(), columns = ['ICD9_CODE','if_last_then_num_pts_prev_visits'])

""" output with freq and description """
all_outcome_output = all_outcome.copy()
all_outcome_output = pd.merge(all_outcome_output, count_pts, how='left')
all_outcome_output = pd.merge(all_outcome_output, count_all, how='left')

all_outcome_output['ICD'] = all_outcome_output.ICD9_CODE
all_outcome_output.ICD9_CODE = [x.replace('ICD_', '') for x in all_outcome_output.ICD9_CODE]
all_outcome_output = pd.merge(all_outcome_output, outcome_lib, how='left')
all_outcome_output = all_outcome_output.sort_values(by=['num_pts_last_visit'], ascending=False)
all_outcome_output = all_outcome_output[
    ['ICD', 'num_pts_last_visit', 'if_last_then_num_pts_prev_visits',
    'if_last_then_num_seq_prev_visits', 'SHORT_TITLE', 'LONG_TITLE']
       ]

all_outcome_output.to_csv("../../Data/MIMIC_outcome_prevalence.csv", index=False)
