# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# This script performs a statistical test of feature enrichment between fig2a and fig1a
# Author: Yiyun

import pickle
import pandas as pd
import sys
import os
from ast import literal_eval
from src.lib.analyses import *
from scipy.stats import chi2_contingency
from scipy import stats
from scipy.stats.contingency import margins
import numpy as np

sys.path.append('../')

# +
### fig1a data source count
# read in data
dm_data = pickle.load(open('../out/20.0216 feat/reg_rf_boruta/dm_data.pkl','rb'))

# get count 
df_counts_1a = pd.DataFrame([{'CERES':dm_data.df_crispr.shape[1],
                           'RNA-seq':dm_data.df_rnaseq.shape[1],
                           'CN':dm_data.df_cn.shape[1],
                           'Mut':dm_data.df_mut.shape[1],
                           'Lineage':dm_data.df_lineage.shape[1]}])

# +
### fig2a TOP10 feature count table
# read in data
dir_in_res = '../out/20.0216 feat/reg_rf_boruta'
dir_in_anlyz = os.path.join(dir_in_res, 'anlyz_filtered')
df_featSummary = pd.read_csv(os.path.join(dir_in_anlyz, 'feat_summary.csv')) #feature summary
df_featSummary['feat_sources'] = df_featSummary['feat_sources'].apply(literal_eval)
df_featSummary['feat_genes'] = df_featSummary['feat_genes'].apply(literal_eval)

# pie chart of feature sources
df_counts_2a = pd.Series([y for x in df_featSummary.feat_sources for y in x]).value_counts()

# +
### Chi-square test for independence: X as feature counts, Y as data source/TOP10 feature source
# H0: The feature counts is not dependent on sources.
# H1: The feature counts is dependent on sources.

# Merge fig2a series into fig1a dataframe
df_counts_combined = df_counts_1a.append(df_counts_2a, \
                                         ignore_index=True).fillna(0).astype('int64') #Fill lineage as 0

# Set significance level
alpha=0.05

# Chi-square test
x2, pval, df, expected_val = chi2_contingency(df_counts_combined)
# -

# Show p value
pval
# # Show Chi-square value at alpha and df
# x2_at_alpha = stats.chi2.ppf(1-alpha,df)
# # Show Chi-square statistics
# x2

# P value < 0.05(<0.001) and Chi-square statistics >> Chi-square value at alpha=0.05 and df=4. Reject the null hypothesis.

# +
### Look at standardized residual of CERES score(no built-in function?)
def stdres(observed, expected):
    n = observed.sum()
    rsum, csum = margins(observed)
    rsum = rsum.astype(np.float64)
    csum = csum.astype(np.float64)
    v = csum * rsum * (n - rsum) * (n - csum) / n**3
    
    return (observed - expected) / np.sqrt(v)

stdres_res = stdres(df_counts_combined.to_numpy(), expected_val)
df_stdres = pd.DataFrame(data=stdres_res[0:,0:],index = ['data source','Top10 feature source'],\
                         columns=df_counts_combined.columns) 
# -

# Show starndardized residual dataframe
df_stdres

# CERES score has the largest standard residue(stdres = 73.069819) at TOP10 feature source, indicating it's highly enriched after feature selection.
