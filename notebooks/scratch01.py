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

# Getting processed genomic/ceres datasets

import pickle
import sys
sys.path.append('../')

dm_data = pickle.load(open('../out/20.0216 feat/reg_rf_boruta/dm_data.pkl','rb'))

# +
# dm_data.df_crispr
# dm_data.df_rnaseq
# dm_data.df_cn
# dm_data.df_mut
# dm_data.df_lineage
# -

# Getting primary model results (reg_rf_boruta)

import pandas as pd

# +
# model results
df_res = pd.read_csv('../out/20.0216 feat/reg_rf_boruta/model_results.csv') 

# feature summary, derived from model results (each row unique for feature-target pair)
df_feat = pd.read_csv('../out/20.0216 feat/reg_rf_boruta/anlyz_filtered/feat_summary.csv')

# feature summary additional metrics, derived from model results (each row unique for feature-target pair)
df_feat2 = pd.read_csv('../out/20.0216 feat/reg_rf_boruta/anlyz_filtered/feat_summary.csv')

# aggregated results (each row unique for target)
df_agg = pd.read_csv('../out/20.0216 feat/reg_rf_boruta/anlyz_filtered/agg_summary_filtered.csv')
# -


