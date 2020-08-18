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

from scipy.stats import chi2_contingency
from scipy import stats
from scipy.stats.contingency import margins
import numpy as np

sys.path.append('../')

# +
#---------------------1. Get data counts--------------------#
### 1. fig1a data source count
# read in data
dm_data = pickle.load(open('../out/20.0216 feat/reg_rf_boruta/dm_data.pkl','rb'))

# get count fig1a
df_count_source = pd.DataFrame([{'CERES':dm_data.df_crispr.shape[1],
                           'RNA-seq':dm_data.df_rnaseq.shape[1],
                           'CN':dm_data.df_cn.shape[1],
                           'Mut':dm_data.df_mut.shape[1],
                           'Lineage':dm_data.df_lineage.shape[1]}])

### 2. fig2a TOP10 feature count table
# read in data
dir_in_res = '../out/20.0216 feat/reg_rf_boruta'
dir_in_anlyz = os.path.join(dir_in_res, 'anlyz_filtered')
df_featSummary = pd.read_csv(os.path.join(dir_in_anlyz, 'feat_summary.csv')) #feature summary
df_featSummary['feat_sources'] = df_featSummary['feat_sources'].apply(literal_eval)
df_featSummary['feat_genes'] = df_featSummary['feat_genes'].apply(literal_eval)

# pie chart of feature sources
df_count_top10 = pd.Series([y for x in df_featSummary.feat_sources for y in x]).value_counts()
df_count_top10 = df_count_top10.to_frame().T


### 3. SupA all siginificant feature count
df_count_allsigfeats = pd.read_csv('%s/anlyz/featSrcCounts/source_counts_allfeatures.csv' % dir_in_res, \
                              header=None, index_col=0, squeeze=True)
df_count_allsigfeats = df_count_allsigfeats.to_frame().T

#--------------------1.End--------------------#

# +
#------------------2. Functions definition--------------------#
### Calcualte standardized residual
# ************************************
# stdres function is a function calculates standardized residual
# This was adapted from a post on Stackoverflow by Warren Weckesser on 12/08/2013, could be found at:
# https://stackoverflow.com/questions/20453729/what-is-the-equivalent-of-r-data-chisqresiduals-in-python
# ************************************
def stdres(observed, expected): 
    n = observed.sum()
    rsum, csum = margins(observed)
    rsum = rsum.astype(np.float64)
    csum = csum.astype(np.float64)
    v = csum * rsum * (n - rsum) * (n - csum) / n**3
    
    return (observed - expected) / np.sqrt(v)

### Chi-squre test for independence: X as feature counts, Y as data source/TOP10 feature source/all significant features
def chi_square_test(df_count1, df_count2, alpha):
    # Merge count1 and count2 into dataframe
    df_counts_combined = pd.concat([df_count1, df_count2], axis=0, ignore_index=True).fillna(0).astype('int64') #Fill lineage as 0
    
    x2, pval, dfree, expected_val = chi2_contingency(df_counts_combined)
    
    # Calculate standard deviation
    stdres_res = stdres(df_counts_combined.to_numpy(), expected_val)
    df_stdres = pd.DataFrame(data=stdres_res[0:,0:],\
                         columns=df_counts_combined.columns) 
    
    
    
    return x2, pval, dfree, df_stdres

#--------------------2. End--------------------#

# +
#--------------------3. Feature Enrichement test--------------------#
### Data source v.s. all significant features
x2_sig_source, pval_sig_source, dfree_sig_source, df_stdres_sig_source = \
chi_square_test(df_count_source, df_count_allsigfeats, 0.05)

# Show p value
print('Data source v.s. all significant features')
print(f'p-value: {pval_sig_source}')
# Show starndardized residual dataframe
df_stdres_sig_source

# +
###Data source v.s. Top10 features
x2_top10_source, pval_top10_source, dfree_top10_source, df_stdres_top10_source = \
chi_square_test(df_count_source, df_count_top10, 0.05)

# Show p value
print('Data source v.s. Top10 features')
print(f'p-value: {pval_top10_source}')
# Show starndardized residual dataframe
df_stdres_top10_source

# +
###Top10 features v.s. significant features
x2_top10_sig, pval_top10_sig, dfree_top10_sig, df_stdres_top10_sig = \
chi_square_test(df_count_allsigfeats, df_count_top10, 0.05)

# # Show p value
print('Top10 features v.s. significant features')
print(f'p-value:{pval_top10_sig}')
# Show starndardized residual dataframe
df_stdres_top10_sig

#--------------------3. End--------------------#
# -

# CERES score always has the largest standardized residuals, indicating it's highly enriched after model building and feature selection.
#
# All P value < 0.05(<0.001) and Chi-square statistics >> Chi-square value at alpha=0.05 and df=4. Reject the null hypothesis.
