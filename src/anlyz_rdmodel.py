#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze model results 
"""

import os
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import glob

from src.lib.utils import getFeatGene, getFeatSource
from src.lib.analyses import *

######################################################################
# Parameters
######################################################################
outdir = './out/20.0216 feat/reg_rf_boruta/'
useGene_dependency = False #whether to use CERES gene dependency (true) or gene effect (false)
metric_eval = 'score_test' #metric in model_results to evaluate, e.g. score_test, score_oob

######################################################################
# Set settings
######################################################################
plt.interactive(False)

# create dir
outdir_sub = '%s/anlyz/' % outdir
if(not os.path.exists(outdir_sub)): os.mkdir(outdir_sub)

# save settings
f = open('%s/anlyz_settings.txt' % (outdir_sub), 'w')
f.write('Evaluation metric used for analysis: %s' % metric_eval)
f.close()

# read in data
model_results = pd.read_csv('%s/model_results.csv' % (outdir), header=0)
counts = model_results.groupby('target')['target'].count()
model_results = model_results.loc[model_results.target.isin(counts[counts>1].index),:] #exclude ones with no reduced models

######################################################################
# High-level stats
######################################################################

df = model_results.loc[model_results.model=='topfeat',]
f = open("%s/model_results_stats.txt" % (outdir_sub), "w")
f.write('There are %d genes out of %d with no feature in reduced data\n' % (sum(df.feature == ''), df.shape[0]))
f.close()

anlyz_model_results(model_results, outdir_sub='%s/stats_score_aggRes/' % outdir_sub, suffix='')

######################################################################
# Aggregate summaries
###################################################################### 
def getAggSummary(x):
    #get variance explained
    df = x.loc[x.model=='all',['target',metric_eval]].copy()
    df.columns = ['target','score_full']
    df.score_full = round(df.score_full,5)
    df['score_rd'] = round(x.loc[x.model=='topfeat', metric_eval].values[0],5) if sum(x.model == 'topfeat')>0 else np.nan
    df['score_rd10'] = round(x.loc[x.model=='top10feat', metric_eval].values[0],5) if sum(x.model == 'top10feat')>0 else np.nan
    df['corr_rd10'] = round(x.loc[x.model == 'top10feat', 'corr_test'].values[0], 5) if sum(x.model == 'top10feat') > 0 else np.nan
    df['recall_rd10'] = round(x.loc[x.model=='top10feat','corr_test_recall'].values[0],5) if sum(x.model == 'top10feat')>0 else np.nan
    df['p19q4_score_rd10'] = round(x.loc[x.model=='top10feat', 'score_p19q4'].values[0],5) if sum(x.model == 'top10feat')>0 else np.nan
    df['p19q4_corr_rd10'] = round(x.loc[x.model == 'top10feat', 'corr_p19q4'].values[0], 5) if sum(x.model == 'top10feat') > 0 else np.nan
    df['p19q4_recall_rd10'] = round(x.loc[x.model=='top10feat', 'corr_p19q4_recall'].values[0],5) if sum(x.model == 'top10feat')>0 else np.nan
    
    return df

aggRes = model_results.groupby('target').apply(getAggSummary)
aggRes.reset_index(inplace=True, drop=True)

# write varExp
aggRes.to_csv("%s/agg_summary.csv" % (outdir_sub), index=False)

anlyz_aggRes(aggRes, outdir_sub='%s/stats_score_aggRes/' % outdir_sub, suffix='')

######################################################################
# Aggregate feature summaries
######################################################################
#-- feature summary, variance ratios etc --
def getVarExp(x):
    #get variance explained
    df = x.loc[x.model == 'univariate', ['feature', 'target', metric_eval]].copy()
    df.columns = ['feature', 'target', 'score_ind']
    df.score_ind = round(df.score_ind,5)
    df['score_rd'] = round(x.loc[x.model == 'top10feat', metric_eval].values[0], 5) if sum(x.model == 'topfeat')>0 else np.nan
    df['score_full'] = round(x.loc[x.model == 'all', metric_eval].values[0], 5)
    df['varExp_ofFull'] = round(df.score_ind / df.score_full,5)
    df['varExp_ofRd'] = round(df.score_ind / df.score_rd,5)
    df['feat_idx'] = list(range(1,df.shape[0]+1))
    return df

varExp = model_results.groupby('target').apply(getVarExp)
varExp.reset_index(inplace=True, drop=True)
varExp['feat_gene'] = varExp['feature'].apply(getFeatGene, firstOnly=True)
varExp['feat_source'] = varExp['feature'].apply(getFeatSource, firstOnly=True)

# write varExp
varExp.to_csv("%s/feat_summary_varExp.csv" % (outdir_sub), index=False)

#-- analyze varExp (related to R2s) (not collapsed - each row = one feat-target pair) --
anlyz_varExp(varExp, outdir_sub='%s/stats_score_feat/' % outdir_sub, suffix='')

if 'dm_data' not in globals():
    from src.lib.data import depmap_data
    dm_data = depmap_data()
    dm_data.load_data(useGene_dependency)
    dm_data.preprocess_data()

anlyz_varExp_wSource(varExp, dm_data, outdir_sub='%s/stats_score_feat/' % outdir_sub, suffix='')

anlyz_scoresGap(varExp, useGene_dependency, outdir_sub='%s/stats_score_feat/' % outdir_sub)

#-- analyze feat summary (related to feat (gene/source)) (collapsed - each row = one target gene) --
anlyz_varExp_feats(varExp, outdir_sub=outdir_sub)


######################################################################
# Gene specific
###################################################################### 
# outdir_sub2 = '%s/gene_specific/' % outdir
# if(not os.path.exists(outdir_sub)): os.mkdir(outdir_sub2)
#
# #generate plot of specific gene
# genBarPlotGene(model_results, 'CDK4', 'score_oob', 0.5, outdir_sub=outdir_sub2)
# genBarPlotGene(model_results, 'KRAS', 'score_oob', 0.5, outdir_sub=outdir_sub2)
# genBarPlotGene(model_results, 'SOX10', 'score_oob', 0.5,  outdir_sub=outdir_sub2)

######################################################################
# Y predictions comparisons
######################################################################
# create dir
outdir_sub2 = '%s/heatmaps/' % outdir_sub
if(not os.path.exists(outdir_sub2)): os.mkdir(outdir_sub2)

def constructYCompr(genes2analyz, compr_pfx):
    # Extract y actual and predicted from pickle file and format into two data frames, respectively; for all given genes
    # compr_pfx specifies the prefix, e.g. tr, te

    df_y_actual = pd.DataFrame()
    df_y_pred = pd.DataFrame()
    for gene2analyz in genes2analyz:
        y_compr = pickle.load(open('%s/model_perf/y_compr_%s.pkl' % (outdir, gene2analyz), "rb"))

        df_y_actual = pd.concat([df_y_actual, pd.DataFrame(y_compr[compr_pfx]['y_actual'].values, columns=[gene2analyz])], axis=1)
        df_y_pred = pd.concat([df_y_pred, pd.DataFrame(y_compr[compr_pfx]['y_pred'].values, columns=[gene2analyz])], axis=1)

    return df_y_actual, df_y_pred

def yComprHeatmap(df_y_actual, df_y_pred, pfx):
    # heatmap
    plt.figure()
    ax = sns.heatmap(df_y_actual, yticklabels=False, xticklabels=False, vmin=-5, vmax=5, cmap='RdBu')
    ax.set(xlabel='Genes', ylabel='Cell lines')
    plt.savefig("%s/%s_heatmap_yactual.png" % (outdir_sub2, pfx))
    plt.close()

    plt.figure()
    ax = sns.heatmap(df_y_pred, yticklabels=False, xticklabels=False, vmin=-5, vmax=5, cmap='RdBu')
    ax.set(xlabel='Genes', ylabel='Cell lines')
    plt.savefig("%s/%s_heatmap_yinferred.png" % (outdir_sub2, pfx))
    plt.close()

genes2analyz = model_results.target.unique()

# for train
df_y_actual, df_y_pred = constructYCompr(genes2analyz, 'tr')
pickle.dump({'actual':df_y_actual, 'predicted':df_y_pred}, open('%s/y_compr_tr.pkl' % outdir_sub,'wb'))
yComprHeatmap(df_y_actual, df_y_pred, 'tr')

# for test
df_y_actual, df_y_pred = constructYCompr(genes2analyz, 'te')
pickle.dump({'actual':df_y_actual, 'predicted':df_y_pred}, open('%s/y_compr_te.pkl' % outdir_sub,'wb'))
yComprHeatmap(df_y_actual, df_y_pred, 'te')

######################################################################
# Concordance
######################################################################
outdir_sub2 = '%s/concordance/' % outdir_sub
if(not os.path.exists(outdir_sub2)): os.mkdir(outdir_sub2)

def getConcordance(df, threshold=-0.6):
    df['concordance'] = 0
    df.loc[(df.y_actual<=threshold) & (df.y_pred<=threshold),'concordance'] = 1
    df.loc[(df.y_actual>threshold) & (df.y_pred>threshold),'concordance'] = 1
    return sum(df.concordance==1)/len(df)

df_conc_tr = pd.DataFrame()
df_conc_te = pd.DataFrame()
for fname in glob.glob('%s/model_perf/y_compr_*.pkl' % outdir):
    f = re.sub('.*_compr_', '', fname)
    gene = re.sub('\.pkl', '', f)
    df = pickle.load(open(fname, 'rb'))

    tmp = pd.DataFrame([{'gene': gene, 'concordance': getConcordance(df['tr'])}])
    df_conc_tr = pd.concat([df_conc_tr, tmp])

    tmp = pd.DataFrame([{'gene': gene, 'concordance': getConcordance(df['te'])}])
    df_conc_te = pd.concat([df_conc_te, tmp])

df_conc_tr.to_csv('%s/concordance_tr.csv' % outdir_sub2, index=False)
df_conc_te.to_csv('%s/concordance_te.csv' % outdir_sub2, index=False)

plt.figure()
ax = sns.distplot(df_conc_tr.concordance)
ax.set(xlim=[0,1.05], xlabel='Concordance',title='Concordance between actual and predicted')
plt.savefig("%s/concordance_tr.pdf" % outdir_sub2)
plt.close()

plt.figure()
ax = sns.distplot(df_conc_te.concordance)
ax.set(xlim=[0,1.05], xlabel='Concordance',title='Concordance between actual and predicted')
plt.savefig("%s/concordance_te.pdf" % outdir_sub2)
plt.close()

######################################################################
# Examine sources
######################################################################
# check for in the selected features
counts = {'CERES': 0,
          'RNA-seq': 0,
          'CN': 0,
          'Mut': 0,
          'Lineage': 0}
df_lineage = pd.DataFrame()
for fname in glob.glob('%s/model_perf/feats_*.csv' % outdir):
    df = pd.read_csv(fname)  # read in the features
    df_feat = pd.DataFrame({'feature': df.feature.str.extract('^(.*)\s')[0],
                            'feature_source': df.feature.str.extract('\[(.*)\]')[0]})
    df_feat['target'] = re.findall('feats_(.*)\.csv', fname)[0]
    df_counts = df.feature.str.extract('\[(.*)\]').groupby(0).size()  # tally the counts
    for n, v in zip(df_counts.index, df_counts.values):
        counts[n] = counts[n] + v  # add to the counts

    if sum(df_feat.feature_source == 'Lineage') > 0:
        df_tmp = df_feat.loc[df_feat.feature_source == 'Lineage', :].copy()
        df_lineage = pd.concat([df_lineage, df_tmp])

df_counts = pd.Series(counts)
df_counts.to_csv('%s/featSrcCounts/source_counts_allfeatures.csv' % outdir_sub, index=True, header=False)
df_lineage.to_csv('%s/featSrcCounts/source_allfeatures_lineage.csv' % outdir_sub)

plotCountsPie(df_counts,
              'Data source summary (all features)',
              'imprank-allfeat_pie',
              '%s/featSrcCounts/' % outdir_sub,
              autopct='%0.2f%%')