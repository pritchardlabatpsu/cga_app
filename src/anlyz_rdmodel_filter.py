#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply filters on model results and analyze
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from src.lib.analyses import *

######################################################################
# Parameters
######################################################################
outdir = './out/20.0216 feat/reg_univariate_rf/'
useGene_dependency = False # whether to use CERES gene dependency (true) or gene effect (false)
thresholds = {'score_rd10': 0.1, # score of reduced model threshold
              'recall_rd10': 0.95} # recall

######################################################################
# Set settings
######################################################################
plt.interactive(False)

# create dir
outdir_sub = '%s/anlyz_filtered/' % outdir
if(not os.path.exists(outdir_sub)): os.mkdir(outdir_sub)

# save settings
f = open('%s/filter_settings.txt' % (outdir_sub), 'w')
for k,v in thresholds.items(): 
    f.write('%s threshold: %.2f\n' % (k,v))
f.close()

######################################################################
# model results filtering and analyses
######################################################################
model_results = pd.read_csv('%s/model_results.csv' % (outdir), header=0)
counts = model_results.groupby('target')['target'].count()
model_results = model_results.loc[model_results.target.isin(counts[counts>1].index),:] #exclude ones with no reduced models
df_res_filtered = model_results.copy()
genes_pass1 = df_res_filtered.loc[(df_res_filtered.model=='top10feat') & (df_res_filtered.score_test>thresholds['score_rd10']),'target']
genes_pass2 = df_res_filtered.loc[(df_res_filtered.model=='top10feat') & (df_res_filtered.corr_test_recall>thresholds['recall_rd10']),'target']
genes_pass = set(genes_pass1).intersection(genes_pass2)
df_res_filtered = df_res_filtered.loc[df_res_filtered.target.isin(genes_pass),:]
anlyz_model_results(df_res_filtered, outdir_sub='%s/stats_score_aggRes/' % outdir_sub, suffix='')

######################################################################
# aggRes filtering and analyses
######################################################################
# read in files
aggRes = pd.read_csv('%s/anlyz/agg_summary.csv' % (outdir), header=0)

# filter based on thresholds
aggRes_filtered = aggRes.copy()
for k,v in thresholds.items():
    aggRes_filtered = aggRes_filtered.loc[aggRes_filtered[k]>v,: ]

# save file
aggRes_filtered.reset_index(inplace=True, drop=True)
aggRes_filtered.to_csv("%s/agg_summary_filtered.csv" % (outdir_sub), index=False)

# analyze
anlyz_aggRes(aggRes_filtered, outdir_sub='%s/stats_score_aggRes/' % outdir_sub, suffix='')

######################################################################
# varExp filtering and analyses
######################################################################
# read in files
varExp = pd.read_csv('%s/anlyz/feat_summary_varExp.csv' % (outdir), header=0)

# filter based on thresholds
varExp_filtered = varExp.loc[varExp.target.isin(aggRes_filtered.target),:].copy()

# save file
varExp_filtered.reset_index(inplace=True, drop=True)
varExp_filtered.to_csv("%s/feat_summary_varExp_filtered.csv" % (outdir_sub), index=False)

# stats
f = open('%s/filter_stats.txt' % outdir_sub, 'w')
f.write('varExp: %d source-target pairs, %d genes\n' % (varExp.shape[0], len(varExp.target.unique())))
f.write('varExp_filtered: %d source-target pairs, %d genes\n' % (varExp_filtered.shape[0], len(varExp_filtered.target.unique())))
f.close()

# analyze
anlyz_varExp(varExp_filtered, outdir_sub='%s/stats_score_feat/' % outdir_sub, suffix='score filtered')
anlyz_varExp_feats(varExp_filtered, outdir_sub=outdir_sub)

if 'dm_data' not in globals():
    from src.lib.data import depmap_data
    dm_data = depmap_data()
    dm_data.dir_datasets = '../datasets/DepMap/19Q3/'
    dm_data.load_data()
    dm_data.preprocess_data()

anlyz_varExp_wSource(varExp_filtered, dm_data, outdir_sub='%s/stats_score_feat/' % outdir_sub, suffix='')
anlyz_scoresGap(varExp_filtered, useGene_dependency, outdir_sub='%s/stats_score_feat/' % outdir_sub)

