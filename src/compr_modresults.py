#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare results across models
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib_venn import venn2

outdir = './out/20.0216 feat/compr_reg-RfBorutaVsUni/'

if(not os.path.exists(outdir)): os.mkdir(outdir)

#inputs
#dir_mod1 = './out/20.0216 feat/class_rfc_boruta/'
#dir_mod2 = './out/20.0216 feat/class_univariate_rfc/'
#mod1_lab = 'class rfc-boruta'
#mod2_lab = 'class univariate-rfc'

dir_mod1 = './out/20.0216 feat/reg_rf_boruta/'
dir_mod2 = './out/20.0216 feat/reg_univariate_rf/'
mod1_lab = 'reg rf-boruta'
mod2_lab = 'reg univariate'

varExp1 = pd.read_csv('%s/anlyz_filtered/feat_summary_varExp_filtered.csv' % (dir_mod1), header=0)
varExp2 = pd.read_csv('%s/anlyz_filtered/feat_summary_varExp_filtered.csv' % (dir_mod2), header=0)

if(not os.path.exists(outdir)): os.mkdir(outdir)

#------------------
# overall
a = set(varExp1.target.unique()).union(set(varExp1.feat_gene.unique()))
b = set(varExp2.target.unique()).union(set(varExp2.feat_gene.unique()))
venn2(subsets = (len(a-b), len(b-a), len(a.intersection(b))), set_labels = (mod1_lab, mod2_lab))
plt.title('Targets and Features')
plt.savefig("%s/venn_overall.pdf" % outdir)
plt.close()

# targets
a = set(varExp1.target.unique())
b = set(varExp2.target.unique())
venn2(subsets = (len(a-b), len(b-a), len(a.intersection(b))), set_labels = (mod1_lab, mod2_lab))
plt.title('Targets')
plt.savefig("%s/venn_targets.pdf" % outdir)
plt.close()

# features
a = set(varExp1.feat_gene.unique())
b = set(varExp2.feat_gene.unique())
venn2(subsets = (len(a-b), len(b-a), len(a.intersection(b))), set_labels = (mod1_lab, mod2_lab))
plt.title('Features')
plt.savefig("%s/venn_feats.pdf" % outdir)
plt.close()

# of the shared target genes
shared_targets = set(varExp1.target.unique()).intersection(set(varExp2.target.unique()))
# percent overlap

def calcOverlap(df):
    a = set(df.feat_gene)
    b = set(varExp2.loc[varExp2.target == df.iloc[0,1],'feat_gene'])
    return pd.Series([len(a.intersection(b)), len(a), len(b)], index=['overlap', 'model1', 'model2'])

df = varExp1.loc[varExp1.target.isin(shared_targets),:].groupby('target').apply(calcOverlap).reset_index()
#df = varExp1.iloc[1:30,:].groupby('target').apply(calcOverlap).reset_index()

plt.figure()
ax = sns.distplot(df.overlap, kde=False)
ax.set(xlabel='Number of features overlap, per target', ylabel='Count', 
       title='Comparison of features for models:\n%s | %s' % (mod1_lab, mod2_lab))
plt.savefig("%s/dist_overlap.pdf" % outdir)
plt.close()


df['min'] = df.loc[:,['model1', 'model2']].apply(min,1)
df['frac'] = df.overlap / df['min']

plt.figure()
ax = sns.distplot(df.frac, kde=False)
ax.set(xlabel='Fraction of features overlap\n(out of the min of total features (model1, model2)', 
       ylabel='Count', 
       title='Comparison of features for models:\n%s | %s' % (mod1_lab, mod2_lab))
plt.savefig("%s/dist_overlap_frac.pdf" % outdir)
plt.close()


# compare the scores
df = varExp1.merge(varExp2, how='inner', on='target', suffixes=('_1','_2'))
df = df.groupby('target').apply(lambda x: x.iloc[0,[3,12]])

df1 = pd.DataFrame({'score_rd':df.score_rd_1.values,'model': mod1_lab})
df2 = pd.DataFrame({'score_rd':df.score_rd_2.values,'model': mod2_lab})
df_c = pd.concat([df1, df2])

plt.figure()
ax = sns.boxplot('model','score_rd', data=df_c)
plt.savefig("%s/score_compr.pdf" % outdir)
plt.close()



# #-------------------
# # adding in just the univariate models
# # note in the end this didn't matter, because the random forest model was on the top 10 features, from univariate
# # and this was used to build a reduced random forest model
# # but this same 10 features were included so the pie chart comparison will be the same
# outdir = './out/20.0216 feat/reg_univariate_rf/'
# indir = '%s/model_perf/' % outdir
#
# # read into model_perf, and retrieve
# varExp = pd.DataFrame()
# for f in os.listdir(indir):
#     df = pd.read_csv('%s/%s' % (indir, f), header=0)
#     df['target'] = re.findall('_(.*)\.csv',f)[0]
#     df = df.iloc[0:10,:]
#     varExp = varExp.append(df, sort=False)
#
# varExp.reset_index(inplace=True, drop=True)
# varExp['feat_gene'] = varExp['feature'].apply(getFeatGene, firstOnly=True)
# varExp['feat_source'] = varExp['feature'].apply(getFeatSource, firstOnly=True)
#
# varExp.to_csv('%s/anlyz/univariate_varExp.csv' % (outdir), index=False)
#
# # read in the filtered varExp from the analyses
# varExp_anlyz = pd.read_csv('%s/anlyz_filtered/feat_summary_varExp_filtered.csv' % (outdir), header=0)
#
# varExp_filtered = varExp.loc[varExp.target.isin(varExp_anlyz.target),:]
# varExp_filtered.to_csv('%s/anlyz_filtered/univariate_varExp_filtered.csv' % (outdir), index=False)
#
