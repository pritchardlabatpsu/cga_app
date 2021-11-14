from os import path, makedirs
import pandas as pd
import numpy as np
import pickle
from ast import literal_eval

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
import matplotlib
from matplotlib_venn import venn2, venn3
from scipy.stats import pearsonr, gaussian_kde
import networkx as nx

from ceres_infer.analyses import *
from ceres_infer.data import stats_Crispr, scale_data

# settings
plt.interactive(False)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
def set_snsfont(s):
    sns.set(font_scale=s)
    sns.set_style("white")
set_snsfont(1.5)

# output directory
dir_out = './manuscript/figures/'
dir_out_srcdata = path.join(dir_out, 'source_data')

if not path.exists(dir_out):
    makedirs(dir_out)
    makedirs(dir_out_srcdata)

# color definitions
src_colors = {'CERES': (214/255, 39/255, 40/255, 1.0), #red
              'RNA-seq': (31/255, 119/255, 180/255, 1.0), #blue
              'CN': (255/255, 127/255, 14/255, 1.0), #orange
              'Mut': (44/255, 160/255, 44/255, 1.0), #green
              'Lineage': (188/255, 189/255, 34/255, 1.0), #yellow
              'nan': (220/255, 220/255, 220/255, 1.0)} #grey

essentiality_colors = {'selective_essential': (205/255, 48/255, 245/255, 1.0), #pink
                      'common_nonessential': (150/255, 140/255, 255/255, 1.0),
                      'common_essential': (81/255, 73/255, 135/255, 1.0)
                      }

#######################################################################
# Figure 1 data source
######################################################################
# read in data
dm_data = pickle.load(open('./out/20.0817 proc_data/gene_effect/dm_data.pkl', 'rb'))

# pie chart
df_counts = pd.DataFrame([{'CERES':dm_data.df_crispr.shape[1],
                           'RNA-seq':dm_data.df_rnaseq.shape[1],
                           'CN':dm_data.df_cn.shape[1],
                           'Mut':dm_data.df_mut.shape[1],
                           'Lineage':dm_data.df_lineage.shape[1]}])
# Fig 1 B
df = df_counts.T[0]
df.name = 'Count'
plotCountsPie(df,
              'Data source',
              'fig1_datasrc',
              dir_out,
              autopct='%0.2f%%',
              colors=[src_colors[s] for s in df_counts.T.index])
df.to_csv(path.join(dir_out_srcdata, 'fig_1b.csv'), index=True)

#------------- Supp -------------
df_crispr_stats = stats_Crispr(dm_data)

# scatter mean vs sd
# Supp Fig S1 A
plt.figure()
ax = sns.scatterplot(x='avg', y='std', data=df_crispr_stats, s=90)
ax.set(xlabel='mean (CERES)', ylabel='SD (CERES)')
plt.tight_layout()
plt.savefig("%s/fig1supp_scatter_mean_sd.png" % dir_out)
plt.close()
df_crispr_stats[['avg', 'std']].to_csv(path.join(dir_out_srcdata, 'fig_S1a.csv'), index=True)

# relative CV
def relCV(df): # relative coefficient of variation
    return df.std(axis=0)/np.abs(df.mean(axis=0))

# assumes ordering is the same as src_colors
tmp = [dm_data.df_crispr, dm_data.df_rnaseq, dm_data.df_cn, dm_data.df_mut, dm_data.df_lineage]
res = list(map(relCV, tmp))
df = pd.DataFrame(res).T
df.columns = ['CERES', 'RNA-seq', 'Copy number', 'Mutation', 'Lineage']

# Supp Fig S1 B
plt.figure()
ax = sns.boxplot(x="variable", y="value", data=pd.melt(df), whis=[0, 100], width=.6, linewidth=0.7, palette=list(src_colors.values())[:5])
ax.set(xlabel='', ylabel='Relative CV')
ax.set_yscale("log")
ax.set_xticklabels(ax.get_xticklabels(), rotation=-45)
plt.tight_layout()
plt.savefig("%s/fig1supp_relCV.pdf" % dir_out)
plt.close()
df = pd.melt(df).groupby('variable').describe()
df.columns = df.columns.droplevel(level=0)
df.to_csv(path.join(dir_out_srcdata, 'fig_S1b.csv'), index=True)

print('N per data source: ')
print(df.count())

# Supp Fig S2
# heatmaps - ceres
plt.figure(figsize=(5,4))
ax = sns.heatmap(dm_data.df_crispr, yticklabels=False, xticklabels=False, cmap='RdBu', vmin=-1, vmax=1, cbar=False)
ax.set(xlabel='Genes', ylabel='Cell lines')
ax.set_title(label='CERES', color=src_colors['CERES'])
plt.tight_layout()
plt.savefig("%s/fig1supp_datasrc_heatmap_ceres.png" % dir_out)
plt.close()

# heatmaps - rna-seq
df = scale_data(dm_data.df_rnaseq, [], None)
df = pd.DataFrame(df[0], index=dm_data.df_rnaseq.index, columns = dm_data.df_rnaseq.columns)
plt.figure(figsize=(10,4))
ax = sns.heatmap(df, yticklabels=False, xticklabels=False, cmap='RdBu', vmin=-1, vmax=1, cbar=False)
ax.set(xlabel='Genes', ylabel='Cell lines')
ax.set_title(label='RNA-seq', color=src_colors['RNA-seq'])
plt.tight_layout()
plt.savefig("%s/fig1supp_datasrc_heatmap_rnaseq.png" % dir_out)
plt.close()

# heatmaps - cn
df = scale_data(dm_data.df_cn, [], None)
df = pd.DataFrame(df[0], index=dm_data.df_cn.index, columns = dm_data.df_cn.columns)
plt.figure(figsize=(5,4))
ax = sns.heatmap(df, yticklabels=False, xticklabels=False, cmap='RdBu', vmin=-1, vmax=1, cbar=False)
ax.set(xlabel='Genes', ylabel='Cell lines')
ax.set_title(label='Copy number', color=src_colors['CN'])
plt.tight_layout()
plt.savefig("%s/fig1supp_datasrc_heatmap_cn.png" % dir_out)
plt.close()

# heatmaps - mutation
plt.figure(figsize=(5,4))
ax = sns.heatmap(dm_data.df_mut,yticklabels=False, xticklabels=False, cmap='RdBu', vmin=-1, vmax=1, cbar=False)
ax.set(xlabel='Genes', ylabel='Cell lines')
ax.set_title(label='Mutation', color=src_colors['Mut'])
plt.tight_layout()
plt.savefig("%s/fig1supp_datasrc_heatmap_mutation.png" % dir_out)
plt.close()

# heatmaps - lineage
plt.figure(figsize=(5,4))
ax = sns.heatmap(dm_data.df_lineage,yticklabels=False, xticklabels=False, cmap='RdBu', vmin=-1, vmax=1, cbar=False)
ax.set(xlabel='Lineages', ylabel='Cell lines')
ax.set_title(label='Lineage', color=src_colors['Lineage'])
plt.tight_layout()
plt.savefig("%s/fig1supp_datasrc_heatmap_lineage.png" % dir_out)
plt.close()

# table of features with lineage
# generated manually, based on analyses in notebook

######################################################################
# Figure 1 model related
######################################################################
# read in data
dir_in_res = './out/20.0216 feat/reg_rf_boruta'
dir_in_anlyz = path.join(dir_in_res, 'anlyz_filtered')
df_varExp = pd.read_csv(path.join(dir_in_anlyz, 'feat_summary_varExp_filtered.csv'), header=0) #feature summary, more score metrics calculated
df_aggRes = pd.read_csv(path.join(dir_in_anlyz, 'agg_summary_filtered.csv')) #aggregated feat summary
dep_class = pd.read_csv('./out/20.0817 proc_data_baseline/gene_effect/gene_essential_classification.csv', header=None, index_col=0, squeeze=True)

# bar chart of scores (multivariate vs univariate of all contributing features)
df_tmp = df_varExp.merge(dep_class.to_frame(name='target_dep_class'), left_index=False, right_index=True, left_on='target')
df_rd = df_tmp.groupby('target')[['score_rd', 'target_dep_class']].first()
df = pd.concat([pd.DataFrame({'score': df_rd.score_rd,
                              'label': 'multivariate',
                              'target_dep_class': df_rd.target_dep_class}),
                pd.DataFrame({'score': df_varExp.score_ind,
                              'label': 'univariate',
                              'target_dep_class': df_tmp.target_dep_class})
               ])

# Fig 1 F
plt.figure()
ax = sns.boxplot(x='label', y='score', data=df.loc[df.score>0, :], palette=essentiality_colors, hue='target_dep_class')
ax.set(xlabel='Model', ylabel='Score')
handels, labels = plt.gca().get_legend_handles_labels()
labels = [n.replace('_', ' ') for n in labels]
plt.legend(handels, labels, loc='upper right')
plt.tight_layout()
plt.savefig("%s/fig1_compr_score_boxplot.pdf" % dir_out)
plt.close()
df = df.loc[df.score>0, :].groupby('target_dep_class').describe()
df.columns = df.columns.droplevel(level=0)
df.to_csv(path.join(dir_out_srcdata, 'fig_1f.csv'), index=True)

print('N per target class: ')
print(df.loc[df.score>0, :].groupby('target_dep_class')['score'].count())

# scatter plot (multivariate vs most important univariate)
# Fig 1 G
df = df_varExp.loc[df_varExp.feat_idx == 1,:]
plt.figure()
plt.plot([-0.3, 1],[-0.3, 1], ls="--", c='0.3')
ax = sns.scatterplot(df.score_rd, df.score_ind, s=40, alpha=0.8, linewidth=0, color='steelblue')
ax.set(xlabel='Model score (multivariate)', ylabel='Model score (univariate, top feature)',
       xlim=[-0.3, 1], ylim=[-0.3, 1])
plt.tight_layout()
plt.savefig("%s/fig1_compr_score_scatter.pdf" % dir_out)
plt.close()
rename_dict = {'score_rd':'Multivariate','score_ind':'Univariate'}
df[['score_rd', 'score_ind']].rename(columns=rename_dict).to_csv(path.join(dir_out_srcdata, 'fig_1g.csv'), index=True)

# compare to baseline model (nearest neighbor)
model_dirs = {'rf_boruta': './out/20.0819 modcompr/reg_rf_boruta',
              'nn':'./out/21.0406 nearest_neighbor'}
xlab_dict = {'rf_boruta': 'Final model\n(ML genomic+functional)',
              'nn': 'Baseline model\n(Nearest neighbors)'} # xlabel dict mapping

scores_corr = pickle.load(open(path.join(model_dirs['nn'],'scores_corr.pkl'), 'rb'))

# Fig 1 H
plt.figure()
ax = sns.barplot(scores_corr.columns, scores_corr.values[0], color='steelblue')
ax.set(ylabel="Model performance\nPearson's $\it{r}$", xlabel='', title='')
ax.set_xticklabels([xlab_dict[n] for n in model_dirs.keys()], rotation=0, size=15)
plt.tight_layout()
plt.savefig("%s/fig1_compr_nn_model.pdf" % dir_out)
plt.close()
rename_dict = {'rf_boruta':'Final model','nn':'Baseline (nearest neighbors)'}
scores_corr.rename(columns=rename_dict).to_csv(path.join(dir_out_srcdata, 'fig_1h.csv'), index=False)

#------------- Supp -------------
#-- comparison of different ML models
model_dirs = {'elasticnet': './out/20.0819 modcompr/reg_elasticNet_infer/',
              'lm': './out/20.0819 modcompr/reg_lm_infer/',
              'rf': './out/20.0819 modcompr/reg_rf_infer/',
              'rf_boruta': './out/20.0216 feat/reg_rf_boruta'}

xlab_dict = {'elasticnet': 'elastic net',
              'lm': 'linear regression',
              'rf': 'random forest',
              'rf_boruta': 'random forest\niter select+boruta'} # xlabel dict mapping

# scores (test set), full model
scores = {}
for model, model_dir in model_dirs.items():
    res = pd.read_csv(path.join(model_dir, 'model_results.csv'))
    scores.update({model: res.loc[res.model == 'all', 'score_test'].values})
scores = pd.DataFrame.from_dict(scores)
df = pd.melt(scores)

# Supp Fig S4 A
plt.figure()
ax = sns.boxplot(data=df, x='variable', y='value', color='steelblue')
ax.set(ylabel='Score (on test)', xlabel='', title='Model using all features')
ax.set_xticklabels([xlab_dict[n] for n in model_dirs.keys()], rotation=-45, size=11)
plt.tight_layout()
plt.savefig("%s/fig1supp_compr_mod_full.pdf" % dir_out)
plt.close()
rename_dict = {'variable':'Model','value':'Score (on test)'}
df.rename(columns=rename_dict).to_csv(path.join(dir_out_srcdata, 'fig_S4a.csv'), index=False)

print('N per model: ')
print(df.groupby('variable')['value'].count())

# scores (median), reduced top10 feat models
scores_rd10 = {}
for model, model_dir in model_dirs.items():
    res = pd.read_csv(path.join(model_dir, 'anlyz/stats_score_aggRes/stats_score.csv'), index_col=0)
    scores_rd10.update({model: res.loc[res.index == '50%', 'reduced10feat'].values})
scores_rd10 = pd.DataFrame.from_dict(scores_rd10)

# Supp Fig S4 B
plt.figure()
ax = sns.barplot(scores_rd10.columns, scores_rd10.values[0], color='steelblue')
ax.set(ylabel='Score (median)', xlabel='', title='Reduced model with top 10 features')
ax.set_xticklabels([xlab_dict[n] for n in model_dirs.keys()], rotation=-45, size=11)
plt.tight_layout()
plt.savefig("%s/fig1supp_compr_mod_rd10.pdf" % dir_out)
plt.close()
rename_dict = {'elasticnet':'Elastic net', 'lm': 'Linear regression', 'rf': 'Random forest', 'rf_boruta':'Random forest + Boruta'}
scores_rd10.rename(columns=rename_dict).to_csv(path.join(dir_out_srcdata, 'fig_S4b.csv'), index=False)

# train vs test, for linear regression
res = pd.read_csv(path.join(model_dirs['lm'], 'model_results.csv'))
res.loc[res.model=='all', ['score_train', 'score_test']].describe()
df = pd.melt(res.loc[res.model == 'all', ['score_train', 'score_test']])

# Supp Fig S4 C right
plt.figure(figsize=(8, 10))
set_snsfont(2)
ax = sns.boxplot(data=df, x='variable', y='value')
ax.set_yscale('symlog')
ax.set(ylabel='Score', xlabel='', xticklabels=['Train', 'Test'], title='Linear regression',
       ylim=[-1,1.2], yticks=[-1, -0.5, 0, 0.5, 1])
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.tight_layout()
plt.savefig("%s/fig1supp_compr_mod_traintest_lm.pdf" % dir_out)
plt.close()
rename_dict = {'variable':'Train/Test', 'value':'Score'}
df.rename(columns=rename_dict).to_csv(path.join(dir_out_srcdata, 'fig_S4c_right.csv'), index=False)

print('N per test/train for LM: ')
print(df.groupby('variable')['value'].count())

# train vs test, for elastic net
res = pd.read_csv(path.join(model_dirs['elasticnet'], 'model_results.csv'))
res.loc[res.model=='all',['score_train', 'score_test']].describe()
df = pd.melt(res.loc[res.model=='all',['score_train', 'score_test']])

# Supp Fig S4 C left
plt.figure(figsize=(8,10))
set_snsfont(2)
ax = sns.boxplot(data=df, x='variable', y='value')
ax.set_yscale('symlog')
ax.set(ylabel='Score', xlabel='', xticklabels=['Train', 'Test'], title='Elastic net',
       ylim=[-1,1.2], yticks=[-1, -0.5, 0, 0.5, 1])
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.tight_layout()
plt.savefig("%s/fig1supp_compr_mod_traintest_en.pdf" % dir_out)
plt.close()
rename_dict = {'variable':'Train/Test', 'value':'Score'}
df.rename(columns=rename_dict).to_csv(path.join(dir_out_srcdata, 'fig_S4c_left.csv'), index=False)

set_snsfont(1.5) # reset back

#-- validation of random forest boruta model
# top 10 feat vs all important feat - bar
# Supp Fig S5 D
df = pd.concat([pd.DataFrame({'score': df_aggRes.score_rd, 'label': 'All selected features'}),
                pd.DataFrame({'score': df_aggRes.score_rd10, 'label': 'Top 10 features'})])
plt.figure()
ax = sns.boxplot(x='label', y='score', data=df.loc[df.score>0, :], color='steelblue')
ax.set(xlabel='Model', ylabel='Score')
plt.tight_layout()
plt.savefig("%s/fig1supp_compr_score_boxplot.pdf" % dir_out)
plt.close()
rename_dict = {'score':'Score', 'label':'Model with given features'}
df.rename(columns=rename_dict).to_csv(path.join(dir_out_srcdata, 'fig_S5d.csv'), index=False)

print('N per rf model type: ')
print(df.groupby('label')['label'].count())

# top 10 feat vs all important feat - scatter
# Supp Fig S5 E
plt.figure()
ax = sns.scatterplot(df_aggRes.score_rd, df_aggRes.score_rd10, s=40, alpha=0.8, linewidth=0, color='steelblue')
ax.plot([0, 0.9], [0, 0.9], ls="--", c=".3")
ax.set(xlabel='Score (model with all selected features)', ylabel='Score (model based on top 10 features)')
plt.tight_layout()
plt.savefig("%s/fig1supp_compr_score_scatter.pdf" % dir_out)
plt.close()
rename_dict = {'score_rd':'Score (full model)', 'score_rd10':'Score (model with top 10 features)'}
df_aggRes[['score_rd', 'score_rd10']].rename(columns=rename_dict).to_csv(path.join(dir_out_srcdata, 'fig_S5e.csv'), index=False)

# smooth scatter plots on scores/recalls/correlations generated in figures.R

######################################################################
# Figure 2
######################################################################
# read in data
dir_in_res = './out/20.0216 feat/reg_rf_boruta'
dir_in_anlyz = path.join(dir_in_res, 'anlyz_filtered')
df_featSummary = pd.read_csv(path.join(dir_in_anlyz, 'feat_summary.csv')) #feature summary
df_featSummary['feat_sources'] = df_featSummary['feat_sources'].apply(literal_eval)
df_featSummary['feat_genes'] = df_featSummary['feat_genes'].apply(literal_eval)
df_varExp = pd.read_csv(path.join(dir_in_anlyz, 'feat_summary_varExp_filtered.csv'), header=0) #feature summary, more score metrics calculated
topN = getFeatN(df_featSummary)
df_src_allfeats = pd.read_csv('%s/anlyz/featSrcCounts/source_counts_allfeatures.csv' % dir_in_res, header=None, index_col=0, squeeze=True)

# color palette for in/out of same group
in_colors = {'in': (120/255, 120/255, 120/255, 1.0), # dark grey
             'out': (220/255, 220/255, 220/255, 1.0), # light grey
             'out2': (250/255, 250/255, 250/255, 0.25)} # very light grey

# pie chart of feature sources
# Fig 2 A
df_counts = pd.Series([y for x in df_featSummary.feat_sources for y in x]).value_counts()
plotCountsPie(df_counts,
              'Feature source (top %d features)' % topN,
              'fig2_feat_source',
              dir_out,
              colors=[src_colors[s] for s in df_counts.index])
df_counts.name = 'Count'
df_counts.to_csv(path.join(dir_out_srcdata, 'fig_2a.csv'), index=True)

# heatmap of feature sources
# colorbar to be manually shown
# Fig 2 B
df = df_featSummary.loc[:,df_featSummary.columns.str.contains(r'feat_source\d')].copy()
df.replace({'CERES': 0, 'RNA-seq': 1, 'CN': 2, 'Mut': 3, np.nan: -1}, inplace=True)
heatmapColors = [src_colors[n] for n in ['nan', 'CERES', 'RNA-seq', 'CN', 'Mut']]
cmap = LinearSegmentedColormap.from_list('Custom', heatmapColors, len(heatmapColors))
plt.figure()
ax = sns.heatmap(df, cmap=cmap, yticklabels=False, xticklabels=list(range(1, 11)), cbar=False)
ax.set(xlabel='$\it{n}$th Feature', ylabel='Target genes')
plt.tight_layout()
plt.savefig("%s/fig2_heatmap.pdf" % dir_out)
plt.close()
df.to_csv(path.join(dir_out_srcdata, 'fig_2b.csv'), index=False)

def gen_feat_pies(sameGrp_counts, sameGrp_src_counts, feat_summary_annot, dir_out, fnames, labels):
    # pie chart of counts in/not in group
    c = sameGrp_counts.loc[sameGrp_counts.importanceRank == 'top10', 'count'][0]
    df_counts = pd.Series({labels[0]: c, # count for in group
                           labels[1]: feat_summary_annot.shape[0] - c}) # count for not in group

    labels = ['%s (%d)' % (x, y) for x, y in zip(df_counts.index, df_counts.values)]
    plt.figure()
    plt.pie(df_counts.values, autopct='%0.1f%%', colors=[in_colors['in'], in_colors['out']])
    plt.axis("image")
    plt.legend(labels=labels, borderaxespad=0, loc='upper right', bbox_to_anchor=(1.4, 1), prop={'size': 13}, frameon=False)
    plt.tight_layout()
    plt.savefig("%s/%s_pie.pdf" % (dir_out, fnames[0]), bbox_inches='tight')
    plt.close()
    df_counts.name = 'Count'
    df_counts.to_csv(path.join(dir_out_srcdata, f"fig_2c-e_{fnames[0].replace('fig2-same','')}_pie.csv"), index=True)

    # pie chart of feature source
    c = sameGrp_src_counts.loc[sameGrp_src_counts.importanceRank == 'top10', :]
    df_counts = pd.Series(c['count'].values, index=c['source'])
    labels = ['%s (%d)' % (x, y) for x, y in zip(df_counts.index, df_counts.values)]
    plt.figure()
    plt.pie(df_counts.values, autopct='%0.1f%%', colors=[src_colors[s] for s in df_counts.index])
    plt.axis("image")
    plt.legend(labels=labels, borderaxespad=0, loc='upper right', bbox_to_anchor=(1.4, 1), prop={'size': 13}, frameon=False)
    plt.tight_layout()
    plt.savefig("%s/%s_pie.pdf" % (dir_out, fnames[1]), bbox_inches='tight')
    plt.close()
    df_counts.name = 'Count'
    df_counts.to_csv(path.join(dir_out_srcdata, f"fig_2c-e_{fnames[1].replace('fig2-same','')}_pie.csv"), index=True)

    #heatmap
    s1 = feat_summary_annot.columns.str.startswith('inSame')
    s2 = ~feat_summary_annot.columns.str.contains('top')
    df = feat_summary_annot.loc[:, s1 & s2]

    heatmapColors = [in_colors['out2'], in_colors['in']]
    cmap = LinearSegmentedColormap.from_list('Custom', heatmapColors, len(heatmapColors))
    plt.figure()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax = sns.heatmap(df, yticklabels=False, xticklabels=list(range(1, 11)), vmin=0, vmax=1, cmap=cmap, cbar=False)
    ax.set(xlabel='$\it{n}$th Feature', ylabel='Target genes')
    plt.tight_layout()
    plt.savefig("%s/%s_heatmap.png" % (dir_out, fnames[1]))
    plt.close()
    df.to_csv(path.join(dir_out_srcdata, f"fig_2c-e_{fnames[1].replace('fig2-same','')}_heatmap.csv"), index=True)

# on same gene
# Fig 2 C
feat_summary_annot_gene = pd.read_csv(path.join(dir_in_anlyz, 'onsamegene', 'feat_summary_annot.csv'), header=0, index_col=0)
sameGrp_counts, sameGrp_src_counts = getGrpCounts_fromFeatSummaryAnnot(feat_summary_annot_gene)
gen_feat_pies(sameGrp_counts,sameGrp_src_counts,feat_summary_annot_gene,
              dir_out, ['fig2-samegene',  'fig2-samegene_source'], ['On same gene', 'Not on same gene'])

# in same paralog set
# Fig 2 D
gs_name = 'paralog'
feat_summary_annot_paralog = pd.read_csv(path.join(dir_in_anlyz, f'insame{gs_name}', 'feat_summary_annot.csv'), header=0, index_col=0)
sameGrp_counts, sameGrp_src_counts= getGrpCounts_fromFeatSummaryAnnot(feat_summary_annot_paralog)
gen_feat_pies(sameGrp_counts,sameGrp_src_counts,feat_summary_annot_paralog,
              dir_out, [f'fig2-same{gs_name}',  f'fig2-same{gs_name}_source'], [f'In {gs_name}', f'Not in {gs_name}'])

# in same gene set Panther
# Fig 2 E
gs_name = 'Panther'
feat_summary_annot_panther = pd.read_csv(path.join(dir_in_anlyz, f'insamegeneset{gs_name}', 'feat_summary_annot.csv'), header=0, index_col=0)
sameGrp_counts, sameGrp_src_counts= getGrpCounts_fromFeatSummaryAnnot(feat_summary_annot_panther)
gen_feat_pies(sameGrp_counts,sameGrp_src_counts,feat_summary_annot_panther,
              dir_out, [f'fig2-same{gs_name}',  f'fig2-same{gs_name}_source'], [f'In {gs_name}', f'Not in {gs_name}'])

# lethality counts - top 1
# Fig 2 F left
df_src1 = df_featSummary[['target', 'feat_source1']].set_index('target')
df = pd.DataFrame({'hasNoCERES': df_src1.feat_source1.isin(['RNA-seq', 'CN', 'Mut']),
                   'sameGene': feat_summary_annot_gene.inSame_1,
                   'sameParalog': feat_summary_annot_paralog.inSame_1,
                   'sameGS': feat_summary_annot_panther.inSame_1,
                   'hasCERES': df_src1.feat_source1 == 'CERES'
                   })
lethal_dict = {'sameGene': 'Same gene',
               'sameParalog': 'Paralog',
               'sameGS': 'Gene set',
               'hasCERES': 'Functional',
               'hasNoCERES': 'Classic\nsynthetic'}
df_counts = pd.DataFrame({'sum': df.sum(axis=0)})
df_counts['lethality'] = [lethal_dict[n] for n in df_counts.index]

set_snsfont(2)
plt.figure(figsize=(10, 6))
ax = sns.barplot(df_counts['lethality'], df_counts['sum'], color='steelblue')
ax.set(xlabel='Lethality types', ylabel='Number of genes',  ylim=[0, 500], title='Top most important feature')
plt.tight_layout()
plt.savefig("%s/fig2_lethality_counts_top1.pdf" % dir_out)
plt.close()
rename_dict = {'sum':'Number of genes', 'lethality':'Lethality type'}
df_counts.rename(columns=rename_dict).to_csv(path.join(dir_out_srcdata, 'fig_2f_left.csv'), index=True)


# lethality counts - top 10
# Fig 2 F right
df_src = df_featSummary.set_index('target').feat_sources
df = pd.DataFrame({'hasNoCERES': df_src.apply(lambda x: any([n in x for n in ['CN','Mut','RNA-seq','Lineage']])),
                   'sameGene': feat_summary_annot_gene.inSame_top10,
                   'sameParalog': feat_summary_annot_paralog.inSame_top10,
                   'sameGS': feat_summary_annot_panther.inSame_top10,
                   'hasCERES': df_src.apply(lambda x: 'CERES' in x)
                   })
lethal_dict = {'sameGene': 'Same gene',
               'sameParalog': 'Paralog',
               'sameGS': 'Gene set',
               'hasCERES': 'Functional',
               'hasNoCERES': 'Classic\nsynthetic'}
df_counts = pd.DataFrame({'sum': df.sum(axis=0)})
df_counts['lethality'] = [lethal_dict[n] for n in df_counts.index]

set_snsfont(2)
plt.figure(figsize=(10, 6))
ax = sns.barplot(df_counts['lethality'], df_counts['sum'], color='steelblue')
ax.set(xlabel='Lethality types', ylabel='Number of genes', ylim=[0, 500], title='Top 10 features')
plt.tight_layout()
plt.savefig("%s/fig2_lethality_counts_top10.pdf" % dir_out)
plt.close()
rename_dict = {'sum':'Number of genes', 'lethality':'Lethality type'}
df_counts.rename(columns=rename_dict).to_csv(path.join(dir_out_srcdata, 'fig_2f_right.csv'), index=True)

set_snsfont(1.5) # reset back

# redundancy scores
# Fig 2 G
df_source = df_featSummary.loc[:,df_featSummary.columns.str.contains(r'feat_source\d')].copy()
source_vals = [v for v in pd.melt(df_source).value.unique() if v is not np.nan]
df_source.columns = range(df_source.shape[1])
df_gene = df_featSummary.loc[:,df_featSummary.columns.str.contains(r'feat_gene\d')].copy()
df_gene.columns = range(df_source.shape[1])

n_feat = [sum(df_source.count(axis='columns'))]
n_uniq_gene = [pd.melt(df_gene)['value'].nunique()]
for v in source_vals:
    fc = sum(df_source[df_source == v].count())
    gc = pd.melt(df_gene[df_source == v])['value'].nunique()
    n_feat.append(fc)
    n_uniq_gene.append(gc)
score_redun = [round(t / i, 3) if i else np.nan for i, t in zip(n_uniq_gene, n_feat)] # redundancy score
score_unq = [round(i / t, 3) if t else np.nan for i, t in zip(n_uniq_gene, n_feat)] # uniqueness score

df_redun = pd.DataFrame({'feature': ['all'] + source_vals,
                         'uniq_gene_count': n_uniq_gene,
                         'feature_count': n_feat,
                         'uniqueness_score': score_unq,
                         'redundancy_score': score_redun})
df_rd = df_redun.loc[df_redun.feature.isin(source_vals), :]
ax = df_rd.plot.barh(x='feature', y='redundancy_score', width = 0.8,
                     color = [src_colors[c] for c in df_rd.feature], legend = None)
ax.set_xlabel('Redundancy score\n(No. total features/No. unique features)')
ax.set_ylabel('Feature')
plt.tight_layout()
ax.figure.savefig("%s/fig2_redundancy_scores.pdf" % dir_out)
plt.close()
rename_dict = {'feature':'Feature type', 'redundancy_score':'Redundancy score'}
df_rd[['feature', 'redundancy_score']].rename(columns=rename_dict).to_csv(path.join(dir_out_srcdata, 'fig_2g.csv'), index=False)

#------------- Supp -------------
# violin plot of scores, breakdown by source
# Supp Fig S6 C
plt.figure()
ax = sns.violinplot('feat_source', 'score_ind', data=df_varExp.loc[df_varExp.score_ind>0,:], alpha=0.1, jitter=True,
                    order=['CERES', 'RNA-seq', 'CN', 'Mut'],
                    palette=src_colors)
ax.set(xlabel='Feature source', ylabel='Score (univariate)')
plt.tight_layout()
plt.savefig("%s/fig2supp_score_by_source.pdf" % dir_out)
plt.close()
df = df_varExp.loc[df_varExp.score_ind>0,['feat_source', 'score_ind']]
rename_dict = {'feat_source':'Feature source', 'score_ind':'Univariate score'}
df.rename(columns=rename_dict).to_csv(path.join(dir_out_srcdata, 'fig_S6c.csv'), index=False)

# source for all features
# Supp Fig S6 A
labels = ['%s (%d)' % (x, y) for x, y in zip(df_src_allfeats.index, df_src_allfeats.values)]
plt.figure()
plt.pie(df_src_allfeats.values, autopct='%0.2f%%', colors=[src_colors[s] for s in df_src_allfeats.index])
plt.title('Data source summary (all features)')
plt.axis("image")
plt.legend(labels=labels, borderaxespad=0, loc='upper right', bbox_to_anchor=(1.5, 1), prop={'size': 13}, frameon=False)
plt.tight_layout()
plt.savefig("%s/%s_pie.pdf" % (dir_out, 'fig2supp-source_allfeat'), bbox_inches='tight')
plt.close()
df_src_allfeats.name = 'Count'
df_src_allfeats.index.name = None
df_src_allfeats.to_csv(path.join(dir_out_srcdata, 'fig_S6a.csv'), index=True)

# break down of source, by nth feature
# Supp Fig S7
pie_imprank_dir = path.join(dir_out, 'pie_imprank')
if not os.path.exists(pie_imprank_dir):
    os.makedirs(pie_imprank_dir)

df_counts_combined = []
for n in range(1, topN + 1):
    plt.interactive(False)
    plotImpSource(n, df_featSummary, pie_imprank_dir, src_colors_dict=src_colors)

    source_col = f'feat_source{n}'
    df_counts = df_featSummary[df_featSummary[source_col] != ''].groupby(source_col)[source_col].count()
    df_counts_combined.append(df_counts)

df_counts_combined = pd.concat(df_counts_combined, axis=1)
df_counts_combined.fillna(0, inplace=True)
df_counts_combined.astype(int).to_csv(path.join(dir_out_srcdata, 'fig_S7.csv'), index=True)

######################################################################
# Figure 3
######################################################################
# network
dir_in_network = './out/20.0216 feat/reg_rf_boruta/network/'
min_gs_size = 4
modularity = pickle.load(open(path.join(dir_in_network, 'modularity.pkl'), 'rb'))
G = nx.read_adjlist(open(path.join(dir_in_network, 'varExp_filtered.adjlist'), 'rb'))
pos = nx.spring_layout(G, seed=25)  # compute layout
cmap = plt.cm.get_cmap('RdYlBu', len(modularity))

# Fig 3 B
np.random.seed(25)
plt.figure(figsize=(9, 9))
plt.axis('off')
for k,v in modularity.items():
    if(len(v)>min_gs_size):
        val = np.random.randint(0,len(modularity)-1)
        nx.draw_networkx_nodes(G, pos, node_size=5, nodelist=v, node_color=[cmap(val)])
        nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='k', style='solid', width=1, edgelist=G.edges(v))
plt.savefig('%s/fig3_network.pdf' % dir_out)
pd.DataFrame(pos).to_csv(path.join(dir_out_srcdata, 'fig_3b_coords.csv'), index=True)

# network communities
# generated and exported from cytoscape
# network stats, excel created manually, based on stats from cytoscape

#------------- Supp -------------
# power analysis
fname = './out/20.0216 feat/reg_rf_boruta/cytoscape/undirected_stats/undirected_deg_gt1.netstats'

process = False
hist = list()
add = False
with open(fname, 'r') as f:
    for line in f:
        if line.startswith('degreeDist'):
            add = True
            continue
        elif line.startswith('cksDist'):
            add = False
            break
        if add:
            v = line.split('\t')
            hist.append((int(v[0]), int(re.sub('\n','',v[1]))))
df = pd.DataFrame(hist, columns=['degree', 'n'])
df = df.loc[df.degree>1,:]

from scipy.optimize import curve_fit
pl = lambda x, a, b: a * (x**b)
popt, pcov = curve_fit(pl, df.degree, df.n,  maxfev=10000)

# Supp Fig S8 D
x = np.linspace(0.5, 50, 100)
plt.figure()
ax = sns.scatterplot(df.degree, df.n, color='steelblue', s=150, alpha=0.9, linewidth=1)
plt.plot(x, pl(x, *popt), '--', color='r')
textstr = r'$y = %0.2f x^{%0.2f}$' % (popt[0], popt[1])
ax.text(0.72, 0.9, textstr, transform=ax.transAxes, fontsize=10)
ax = plt.gca()
ax.set(xlim=[1,50], ylim=[0.5,1000], yscale='log', xscale='log',
      xlabel='Degree', ylabel='Number of nodes')
plt.tight_layout()
plt.savefig("%s/fig3supp_powerlaw.pdf" % dir_out)
plt.close()
rename_dict = {'degree':'Degree', 'n':'Number of nodes'}
df.rename(columns=rename_dict).to_csv(path.join(dir_out_srcdata, 'fig_S8d.csv'), index=False)

#------------------
# GProfiler residuals between predictor and target
df_target = pd.read_csv('./manuscript/figures_manual/gprofiler/raw/target_gores.csv')
df_predictor = pd.read_csv('./manuscript/figures_manual/gprofiler/raw/predictor_gores.csv')

topN = 100
df_target_top = df_target.sort_values('p_value').iloc[:topN]
df_predictor_top = df_predictor.sort_values('p_value').iloc[:topN]
df_target_top.columns = df_target_top.columns + '_target'
df_predictor_top.columns = df_predictor_top.columns + '_predictor'

def annotate_text(ax, df):
    for xval, yval in ax.collections[0].get_offsets().data:
        if yval > 10 or yval < -10:
            term_name = df[df['p_value_target'] == xval]['term_name_target'].values[0]
            ax.annotate(term_name, (xval-0.1, yval+0.2), fontsize=12)

# of top 100 terms, get common terms between target and predictor
df_common = df_target_top.merge(df_predictor_top, left_on='term_name_target', right_on='term_name_predictor', how='inner')
df_common[['p_value_target', 'p_value_predictor']] = df_common[['p_value_target', 'p_value_predictor']].apply(lambda x: -np.log10(x))

# Fig 3 F
plt.figure(figsize=(10,5))
ax = sns.residplot(x='p_value_target', y='p_value_predictor', data=df_common, scatter_kws={"s": 270})
ax.set(ylabel='Residuals\n(Enrichment of GO terms\nfor predictor over target genes)', xlabel='p-value in target genes');
annotate_text(ax, df_common)
plt.tight_layout()
plt.savefig("%s/fig3_goresiduals.pdf" % dir_out)
plt.close()
rename_dict = {'p_value_target':'p-value of GO terms; target genes', 'p_value_predictor':'p-value of GO terms; predictor genes'}
df_common[['p_value_target', 'p_value_predictor']].rename(columns=rename_dict).to_csv(path.join(dir_out_srcdata, 'fig_3f.csv'), index=False)

# with capped p-values for non-common terms
df_combined = df_target_top.merge(df_predictor_top, left_on='term_name_target', right_on='term_name_predictor', how='outer')
df_combined.p_value_predictor.fillna(df_combined.p_value_predictor.max(), inplace=True)
df_combined.p_value_target.fillna(df_combined.p_value_target.max(), inplace=True)
df_combined[['p_value_target', 'p_value_predictor']] = df_combined[['p_value_target', 'p_value_predictor']].apply(lambda x: -np.log10(x))

# Supp Fig S8 E
plt.figure(figsize=(10,5))
ax = sns.residplot(x='p_value_target', y='p_value_predictor', data=df_combined, scatter_kws={"s": 270})
ax.set(ylabel='Residuals\n(Enrichment of GO terms\nfor predictor over target genes)', xlabel='p-value in target genes');
annotate_text(ax, df_combined)
plt.tight_layout()
plt.savefig("%s/fig3supp_goresiduals_capped.pdf" % dir_out)
plt.close()
rename_dict = {'p_value_target':'p-value of GO terms; target genes', 'p_value_predictor':'p-value of GO terms; predictor genes'}
df_combined[['p_value_target', 'p_value_predictor']].rename(columns=rename_dict).to_csv(path.join(dir_out_srcdata, 'fig_S8e.csv'), index=False)

######################################################################
# Figure 5
######################################################################
# the source data is not written to csv here, as they are over hundreds of MBs
dir_in_Lx = './out/20.0909 Lx/L200only_reg_rf_boruta_all/'
dep_class = pd.read_csv('./out/20.0817 proc_data_baseline/gene_effect/gene_essential_classification.csv', header=None, index_col=0, squeeze=True)

def Lx_predactual_heatmap(y_compr, suffix, fig_suffix=''):
    # heatmap - test
    plt.figure()
    ax = sns.heatmap(y_compr['actual'], yticklabels=False, xticklabels=False, vmin=-3, vmax=3, cmap='RdBu', cbar=False)
    ax.set(xlabel='Genes', ylabel='Cell lines')
    plt.tight_layout()
    plt.savefig(f"{dir_out}/fig5{fig_suffix}_heatmap_yactual_{suffix}.png", dpi=300)
    plt.close()

    plt.figure()
    ax = sns.heatmap(y_compr['predicted'], yticklabels=False, xticklabels=False, vmin=-3, vmax=3, cmap='RdBu', cbar=False)
    ax.set(xlabel='Genes', ylabel='Cell lines')
    plt.tight_layout()
    plt.savefig(f"{dir_out}/fig5{fig_suffix}_heatmap_ypred_{suffix}.png", dpi=300)
    plt.close()

def Lx_predactual_scatter(y_compr, suffix, fig_suffix=''):
    # dependency class based on gene dependency (which is a model fit, to get at the probability of being essential)
    # get actual/predicted from model
    actual = pd.melt(y_compr['actual'])
    actual = actual.rename(columns={'variable': 'gene'})
    actual.set_index('gene', drop=True, inplace=True)
    pred = pd.melt(y_compr['predicted'])
    pred = pred.rename(columns={'variable': 'gene'})
    pred.set_index('gene', drop=True, inplace=True)
    df = pd.concat([actual, pred], axis=1)
    df.columns = ['actual', 'predicted']

    df_merged = df.merge(dep_class, left_index=True, right_index=True)
    df_merged.columns = ['actual', 'predicted', 'class']

    for class_name in df_merged['class'].unique():
        plt.figure()
        ax = sns.scatterplot('actual', 'predicted', data=df_merged.loc[df_merged['class'] == class_name],
                             s=1.5, alpha=0.1, linewidth=0, color=essentiality_colors[class_name])
        ax.set(title=class_name.replace('_', ' '), xlabel='Actual', ylabel='Predicted')
        plt.tight_layout()
        plt.savefig(f"{dir_out}/fig5{fig_suffix}_pred_actual_{suffix}_{class_name}.png", dpi=300)
        plt.close()

#------------------
# 19Q3/4 data
y_compr_tr = pickle.load(open(path.join(dir_in_Lx, 'anlyz', 'y_compr_tr.pkl'), 'rb'))
y_compr_te = pickle.load(open(path.join(dir_in_Lx, 'anlyz', 'y_compr_te.pkl'), 'rb'))

# Fig 5 B heatmap, top
Lx_predactual_heatmap(y_compr_te, 'te')

# Supp Fig S9 B
Lx_predactual_heatmap(y_compr_tr, 'tr', 'supp')

# Fig 5 B scatter, bottom
set_snsfont(1.8)
Lx_predactual_scatter(y_compr_te, 'te')
set_snsfont(1.5)  # reset back

#------------------
# Broad vs Sanger data
dir_in_Lx_sanger = './out/20.0926 feat Sanger/L200only_reg_rf_boruta_all/'
y_compr_ext = pickle.load(open(path.join(dir_in_Lx_sanger, 'anlyz', 'y_compr_ext.pkl'), 'rb'))

# heatmaps
# Fig 5 D heatmap left
Lx_predactual_heatmap(y_compr_ext, 'sanger')

# scatter
# Fig 5 D scatter right
plt.figure()
plt.plot([-3,2], [-3,2], ls="--", c=".3", alpha=0.5)
ax = sns.scatterplot(y_compr_ext['actual'].values.flatten(), y_compr_ext['predicted'].values.flatten(),
                     s=1, alpha=0.05, linewidth=0, color='steelblue')
ax.set(xlabel='Actual', ylabel='Predicted', xlim=[-3, 2], ylim=[-3, 2])
plt.tight_layout()
plt.savefig("%s/fig5_pred_actual_sanger.png" % dir_out, dpi=300)
plt.close()

#------------------
# randomized model
y_compr_te = pickle.load(open(path.join(dir_in_Lx, 'anlyz', 'y_compr_te.pkl'), 'rb'))

np.random.seed(seed=25)
def getDummyInfer(y):
    return np.random.uniform(-4.4923381539, 3.9745784786800002, size=y.shape[0]) #-4.49.. and 3.97.. are the min and max of CERES scores in the tr dataset
y_pred = y_compr_te['actual'].apply(getDummyInfer, axis=0)

# Fig 5 C
plt.figure()
ax = sns.heatmap(y_pred, yticklabels=False, xticklabels=False, vmin=-3, vmax=3, cmap='RdBu', cbar=False)
ax.set(xlabel='Genes', ylabel='Cell lines')
plt.tight_layout()
plt.savefig(f"{dir_out}/fig5_heatmap_ypred_te_random.png", dpi=300)
plt.close()

#------------- Additional Supp -------------
# saturation analysis
dir_in_Lx_parent = './out/20.0909 Lx/'
recall_cutoff = 0.95
Lx_range = [25, 100, 200, 300]

def getLxPct(x, model_name, cutoff=0.95):
    # get the fraction of targets with recall > cutoff
    df_results = pd.read_csv('%s/L%sonly_reg_rf_boruta_all/model_results.csv' % (dir_in_Lx_parent, x))

    df_results = df_results.loc[df_results.model == model_name, :].copy()
    n_total = df_results.shape[0]
    n_pass = sum(df_results.corr_test_recall > cutoff)

    return n_pass / n_total

def getLxStats(model_name):
    df_stats = {'Lx': [], 'recall_pct': []}
    for x in Lx_range:
        df_stats['Lx'].append(x)
        recall_pct = getLxPct(x, model_name, recall_cutoff)
        df_stats['recall_pct'].append(recall_pct)
    df_stats = pd.DataFrame(df_stats)
    return df_stats

df_stats = getLxStats('top10feat')
df_stats['normalized'] =  df_stats.recall_pct / df_stats.recall_pct[0]

# Supp Fig S9 A
plt.figure()
ax = sns.scatterplot(df_stats.Lx, df_stats.normalized,
                     s=150, alpha=0.9, linewidth=0, color='steelblue')
ax.set(xlabel='Lx', ylabel='Normalized proportions of \npredictable gene targets',
       ylim=[0.8,2.9])
plt.tight_layout()
plt.savefig("%s/fig5supp_saturation.pdf" % dir_out)
plt.close()
rename_dict = {'normalized':'Normalized proportions of predictable gene targets'}
df_stats[['Lx', 'normalized']].rename(columns=rename_dict).to_csv(path.join(dir_out_srcdata, 'fig_S9a.csv'), index=False)

# concordance
df_conc_tr = pd.read_csv(path.join(dir_in_Lx, 'anlyz', 'concordance', 'concordance_tr.csv'))
df_conc_te = pd.read_csv(path.join(dir_in_Lx, 'anlyz', 'concordance', 'concordance_te.csv'))

df1 = df_conc_tr['concordance'].to_frame().copy()
df1['dataset'] = 'train'
df2 = df_conc_te['concordance'].to_frame().copy()
df2['dataset'] = 'test'
df = pd.concat([df1,df2])
df['cat'] = 'one'

# Supp Fig S9 C
plt.figure()
ax = sns.violinplot(y='cat', x='concordance', hue='dataset', data=df, split=True, linewidth=1.6)
ax.set(xlim=[0.34,1.05], xlabel='Concordance', ylabel='', yticks=[])
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig("%s/fig5supp_concordance.pdf" % dir_out)
plt.close()
rename_dict = {'cat':'Train/Test', 'concordance':'Concordance'}
df[['dataset', 'concordance']].rename(columns=rename_dict).to_csv(path.join(dir_out_srcdata, 'fig_S9c.csv'), index=False)

# examples
# Supp Fig S9 D left
genename = 'TP53BP1'
res = pickle.load(open('%s/model_perf/y_compr_%s.pkl' % (dir_in_Lx, genename),'rb'))
df = res['te']
plt.figure()
plt.plot([-2.5,1], [-2.5, 1], ls="--", c=".3", alpha=0.5)
ax = sns.scatterplot(df.y_actual, df.y_pred, s=70, alpha=0.7, linewidth=0, color='steelblue')
ax.set(xlabel='Actual', ylabel='Predicted', xlim=[-2.5, 1], ylim=[-2.5, 1], title=genename)
plt.tight_layout()
plt.savefig("%s/fig5supp_ycompr_%s.pdf" % (dir_out, genename))
plt.close()
rename_dict = {'y_actual':'Actual', 'y_pred':'Predicted'}
df.rename(columns=rename_dict).to_csv(path.join(dir_out_srcdata, 'fig_S9d_left.csv'), index=False)

# Supp Fig S9 D right
genename = 'XRCC6'
res = pickle.load(open('%s/model_perf/y_compr_%s.pkl' % (dir_in_Lx, genename),'rb'))
df = res['te']
plt.figure()
plt.plot([-2.5,1], [-2.5,1], ls="--", c=".3", alpha=0.5)
ax = sns.scatterplot(df.y_actual, df.y_pred, s=70, alpha=0.7, linewidth=0, color='steelblue')
ax.set(xlabel='Actual', ylabel='Predicted', xlim=[-2.5, 1], ylim=[-2.5, 1], title=genename)
plt.tight_layout()
plt.savefig("%s/fig5supp_ycompr_%s.pdf" % (dir_out, genename))
plt.close()
rename_dict = {'y_actual':'Actual', 'y_pred':'Predicted'}
df.rename(columns=rename_dict).to_csv(path.join(dir_out_srcdata, 'fig_S9d_right.csv'), index=False)

######################################################################
# Figure 6
######################################################################
pc9_dir = './out/21.0423 Lx PC9/L200only_reg_rf_boruta/anlyz'
pc9_standalone_dir = './out/21.0720 Lx PC9Standalone/L200only_reg_rf_boruta/anlyz'
to_dir = './out/21.0506 Lx To/L200only_reg_rf_boruta/anlyz'
to_org_dir = './data/ceres_external/To'

df_pc9 = pickle.load(open(path.join(pc9_dir,'y_compr_ext.pkl'),'rb'))
df_pc9_standalone = pickle.load(open(path.join(pc9_standalone_dir,'y_compr_ext.pkl'),'rb'))
df_to = pickle.load(open(path.join(to_dir,'y_compr_ext.pkl'),'rb'))
df_to_org = pd.read_csv(path.join(to_org_dir,'ToCellCERES.csv'), index_col = 0) # original To et al file containing drug names

# format data
# PC9 (Brunello library)
df_pc9 = pd.concat([df_pc9['actual'], df_pc9['predicted']], axis = 0).T
df_pc9.columns = ['actual','predicted']

# PC9 (L200 standalone library)
df_pc9_standalone = pd.concat([df_pc9_standalone['actual'],df_pc9_standalone['predicted']], axis = 0).T
df_pc9_standalone.columns = ['actual','predicted']

# PC9, pred vs pred
df_pc9_pred = pd.concat([df_pc9['predicted'].T, df_pc9_standalone['predicted'].T], axis = 1)
df_pc9_pred.columns = ['standalone', 'brunello']
df_pc9_pred = df_pc9_pred.dropna()

# To et al, drop DMSO and put all actual and predicted values together
# for scatter plot
df_to_actual = df_to['actual'].drop(0).melt()
df_to_predicted = df_to['predicted'].drop(0).melt()
df_to_scatter = pd.concat([df_to_actual['value'], df_to_predicted['value']], axis = 1)
df_to_scatter.columns = ['actual','predicted']

# for Venn Diagram
df_to_venn = df_to
df_to_venn['actual'] = df_to['actual'].drop(0).T
df_to_venn['predicted'] = df_to['predicted'].drop(0).T
df_to_venn['actual'].columns = ['actual_'+ drug for drug in df_to_org.columns[1:]]
df_to_venn['predicted'].columns = ['predicted_'+ drug for drug in df_to_org.columns[1:]]
df_to_venn = pd.concat([df_to_venn['actual'], df_to_venn['predicted']], axis = 1)

#------------------
# plot density plots
def plotDensity(df, title_txt, fname = None, 
                xcol = 'actual', ycol = 'predicted', xlab = 'Measured', ylab = 'Predicted'):
    x = df[xcol]
    y = df[ycol]
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=30, alpha=0.3)
    corr = pearsonr(x, y)[0]
    ax.text(0.05,0.9, f'rho = {corr:.3f}', transform=ax.transAxes)
    ax.set_title(title_txt)
    ax.set_xlabel(xlab);
    ax.set_ylabel(ylab);
    ax.set_ylim([-3,1.5])
    ax.set_xlim([-3,1.5])
    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname)
        plt.close()

# Fig 6 B left
plotDensity(df_pc9, 'PC9; L200 (Brunello)', f"{dir_out}/fig6_scatter_pc9_L200_brunello.png",
            xcol = 'actual', ycol = 'predicted', 
            xlab = 'Measured', 
            ylab = 'Inferred from\nL200 (Brunello)')
rename_dict = {'actual':'Actual', 'predicted':'Predicted'}
df_pc9.rename(columns=rename_dict).to_csv(path.join(dir_out_srcdata, 'fig_6b_left.csv'), index=False)

# Fig 6 B middle
plotDensity(df_pc9_standalone, 'PC9; L200 standalone', f"{dir_out}/fig6_scatter_pc9_L200_standalone.png",
            xcol = 'actual', ycol = 'predicted', 
            xlab = 'Measured', 
            ylab = 'Inferred from\nL200 standalone library')
rename_dict = {'actual':'Actual', 'predicted':'Predicted'}
df_pc9_standalone.rename(columns=rename_dict).to_csv(path.join(dir_out_srcdata, 'fig_6b_middle.csv'), index=False)

# Fig 6 B right
plotDensity(df_pc9_pred, 'PC9; Inference comparison', f"{dir_out}/fig6_scatter_pc9_inf_compare.png",
            xcol = 'standalone', ycol = 'brunello', 
            xlab = 'Inferred from\nL200 standalone library', 
            ylab = 'Inferred from\nL200 (Brunello)')
rename_dict = {'actual':'Actual', 'predicted':'Predicted'}
df_pc9_pred.rename(columns=rename_dict).to_csv(path.join(dir_out_srcdata, 'fig_6b_right.csv'), index=False)

# Supp Fig S10 left
plotDensity(df_to_scatter, 'To et al.; 7 drugs', f"{dir_out}/fig6supp_scatter_To_et_al.png")
rename_dict = {'actual':'Actual', 'predicted':'Predicted'}
df_to_scatter.rename(columns=rename_dict).to_csv(path.join(dir_out_srcdata, 'fig_S10_left.csv'), index=False)

#------------------
# plot Venn diagrams
hits_n = 500

def get_venn_subset(df, hits_n, suffix = None):
    # get the hits from given dataframe
    suffix = '_' + suffix if suffix else ''

    top_actual = df['actual' + suffix].T.sort_values().head(hits_n).index
    top_predicted = df['predicted' + suffix].T.sort_values().head(hits_n).index
    intersect = len(set(top_actual).intersection(top_predicted))
    non_intersect = hits_n - intersect
    
    return (non_intersect, non_intersect, intersect)

# PC9
# Fig 6 C
top_standalone = df_pc9_standalone['predicted'].T.sort_values().head(hits_n).index
top_brunello = df_pc9['predicted'].T.sort_values().head(hits_n).index
top_actual = df_pc9['actual'].T.sort_values().head(hits_n).index

fig, ax = plt.subplots()
out = venn3([set(top_standalone), set(top_brunello), set(top_actual)], 
      ('Inferred from\nL200 standalone','Inferred from\nL200 (Brunello)', 'Measured'), 
      alpha = 0.5)
for text in out.set_labels:
    text.set_fontsize(14)
plt.tight_layout()
plt.savefig(f"{dir_out}/fig6_venn_pc9.png")
plt.close()
df = pd.concat([pd.Series(top_standalone), pd.Series(top_brunello), pd.Series(top_actual)], axis=1, ignore_index=True)
df.columns = ['Inferred from L200 standalone','Inferred from L200 (Brunello)', 'Measured']
df.to_csv(path.join(dir_out_srcdata, 'fig_6c.csv'), index=False)

# To et al.
# Supp Fig S10 right
ovp = []
for drug in df_to_org.columns[1:]:
    venn_subset = get_venn_subset(df_to_venn, hits_n, suffix = drug)
    ovp.append(venn_subset[2]/hits_n)
avg_intersect = round(np.mean(ovp),1)
non_intersect = round(1 - avg_intersect,1)
venn_subset = (non_intersect, non_intersect, avg_intersect)
bar_subset= (1 - avg_intersect, avg_intersect, 1-avg_intersect)

fig, ax = plt.subplots()
venn2(subsets = venn_subset, set_labels = ('Actual hits', 'Predicted hits'))
ax.set_title(f'To et al. (top {hits_n} hits)\nAveraged percent overlap across 7 drugs');
plt.savefig(f"{dir_out}/fig6supp_venn_To_et_al.png")
plt.close()
df = pd.DataFrame({'Proportion overlap between actual and predicted':ovp, 'drug':df_to_org.columns[1:]})
df.to_csv(path.join(dir_out_srcdata, 'fig_S10_right.csv'), index=False)

#TODO
# combine the csv into an Excel
