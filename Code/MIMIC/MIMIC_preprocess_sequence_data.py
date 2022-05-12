#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jiaguo
"""

""" import Modules """
import compress_pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

""" diagnosis and admission """
diag = pd.read_csv("../../Data/MIMIC_III/DIAGNOSES_ICD.csv.gz")
diag = diag.iloc[:,[1,2,4]]
diag.ICD9_CODE = diag.ICD9_CODE.fillna(0)
diag = diag[diag.ICD9_CODE!=0].dropna()

adms = pd.read_csv("../../Data/MIMIC_III/ADMISSIONS.csv.gz")
adms = adms.iloc[:,[1,2,3]]
adms = adms.drop_duplicates()

""" merge """
diag_new = diag.groupby(['SUBJECT_ID', 'HADM_ID']).apply(lambda x: list(x.ICD9_CODE)).reset_index()
diag_new = diag_new.rename(columns={0: 'ICD'})
diag_new = pd.merge(diag_new, adms, on=['SUBJECT_ID', 'HADM_ID'])
diag_id = pd.unique(diag_new.SUBJECT_ID) #46517

diag_count = diag_new.groupby('SUBJECT_ID').apply(lambda x: len(x)).reset_index()
diag_count = diag_count.rename(columns={0: 'diag_count'})
Counter(diag_count.diag_count) # 1:39018, 2:5125, >=3:2374
46517-39018-5125

""" prescription """
pres = pd.read_csv("../../Data/MIMIC_III/PRESCRIPTIONS.csv.gz")
pres = pres.iloc[:,[1,2,10]]
pres.FORMULARY_DRUG_CD = pres.FORMULARY_DRUG_CD.fillna(0)
pres = pres[pres.FORMULARY_DRUG_CD!=0].dropna()

pres_new = pres.groupby(['SUBJECT_ID', 'HADM_ID']).apply(lambda x: list(x.FORMULARY_DRUG_CD)).reset_index()
pres_new = pres_new.rename(columns={0: 'DRUG'})
pres_new = pd.merge(pres_new, adms, on=['SUBJECT_ID', 'HADM_ID'])
pres_id = pd.unique(pres_new.SUBJECT_ID) #39363

pres_count = pres_new.groupby('SUBJECT_ID').apply(lambda x: len(x)).reset_index()
pres_count = pres_count.rename(columns={0: 'pres_count'})
Counter(pres_count.pres_count) # 1:32848, 2:4414, >=3:2101
39363-32848-4414

""" lab tests """
labs = pd.read_csv("../../Data/MIMIC_III/LABEVENTS.csv.gz")
labs = labs.iloc[:,[1,2,3]]
labs.ITEMID = labs.ITEMID.fillna(0)
labs = labs[labs.ITEMID!=0].dropna()

labs_new = labs.groupby(["SUBJECT_ID", "HADM_ID"]).apply(lambda x: list(x.ITEMID)).reset_index()
labs_new = labs_new.rename(columns={0: 'ITEMID'})
labs_new = pd.merge(labs_new, adms, on=['SUBJECT_ID', 'HADM_ID'])
labs_id = pd.unique(labs_new.SUBJECT_ID) #46201

labs_count = labs_new.groupby('SUBJECT_ID').apply(lambda x: len(x)).reset_index()
labs_count = labs_count.rename(columns={0: 'labs_count'})
Counter(labs_count.labs_count) # 1:38956, 2:4971, >=3:2274
46201-38956-4971

""" save list data """
compress_pickle.dump(
    [diag_new, pres_new, labs_new],
    "../../Data/MIMIC_list_data.gz"
    )


""" preprocess the data, unique concept for each visit """
diag, pres, labs = compress_pickle.load("../../Data/MIMIC_list_data.gz")
adms = pd.read_csv("../../Data/MIMIC_III/ADMISSIONS.csv.gz")
adms = adms.iloc[:,[1,2,3]]

diag.ICD = diag.ICD.apply(lambda x: list(set(x)))
pres.DRUG = pres.DRUG.apply(lambda x: list(set(x)))
labs.ITEMID = labs.ITEMID.apply(lambda x: list(set(x)))

uniq_diag = diag.ICD.tolist()
uniq_diag = [x for sublist in uniq_diag for x in sublist]
count_diag = pd.Series(uniq_diag).value_counts() #6984

uniq_pres = pres.DRUG.tolist()
uniq_pres = [x for sublist in uniq_pres for x in sublist]
count_pres = pd.Series(uniq_pres).value_counts() #3267

uniq_labs = labs.ITEMID.tolist()
uniq_labs = [x for sublist in uniq_labs for x in sublist]
count_labs = pd.Series(uniq_labs).value_counts() #710

""" filter freq """
n = 50
count_diag_filter = count_diag[count_diag>=n] #1300
count_pres_filter = count_pres[count_pres>=n] #1348
count_labs_filter = count_labs[count_labs>=n] #490
6984+3267+710-1300-1348-490

uniq_diag = list(count_diag_filter.index)
uniq_pres = list(count_pres_filter.index)
uniq_labs = list(count_labs_filter.index)

diag.ICD = diag.ICD.apply(lambda x: [code for code in x if code in uniq_diag])
pres.DRUG = pres.DRUG.apply(lambda x: [code for code in x if code in uniq_pres])
labs.ITEMID = labs.ITEMID.apply(lambda x: [code for code in x if code in uniq_labs])

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(15, 4))
for i, ax in zip([diag.ICD, pres.DRUG, labs.ITEMID], axes.flatten()):
    plot_data = pd.DataFrame({"num": i.apply(lambda x:len(x))})
    sns.histplot(data=plot_data, x="num", ax=ax)
plt.show()

""" combine three """
work_data = pd.merge(adms, diag.iloc[:,[0,1,2]], how='left')
work_data = pd.merge(work_data, pres.iloc[:,[0,1,2]], how='left')
work_data = pd.merge(work_data, labs.iloc[:,[0,1,2]], how='left')

work_data.ICD = work_data.ICD.fillna('0').apply(lambda x:['ICD_'+str(code) for code in x])
work_data.DRUG = work_data.DRUG.fillna('0').apply(lambda x:['DRUG_'+str(code) for code in x])
work_data.ITEMID = work_data.ITEMID.fillna('0').apply(lambda x:['LAB_'+str(code) for code in x])

work_data['event'] = work_data.ICD+work_data.DRUG+work_data.ITEMID
work_data['event'] = work_data.event.apply(
    lambda x: [code for code in x if (code not in ['ICD_0','DRUG_0','LAB_0'])]
    )
idx = work_data.event.apply(lambda x:len(x)==0)
work_data = work_data.loc[~idx] #58951

""" save seq data for all patients """
work_data_new = work_data[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'event']]
compress_pickle.dump(
    work_data_new,
    '../../Data/MIMIC_seq_data_all.gz'
    )
