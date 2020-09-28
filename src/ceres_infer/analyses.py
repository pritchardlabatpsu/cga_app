#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyses
@author: boyangzhao
"""

import os
import numpy as np
import pandas as pd
import re
import pickle

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import logging

from ceres_infer.utils import *

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


######################################################################
# Supporting methods
# #####################################################################
def plotCountsPie(df_counts, titleTxt, filename, outdir_sub='./',
                  autopct='%0.1f%%', colors=None):
    # plot pie chart
    labels = ['%s (%d)' % (x, y) for x, y in zip(df_counts.index, df_counts.values)]
    plt.figure()
    plt.pie(df_counts.values, labels=labels, autopct=autopct, colors=colors)
    if titleTxt is not None:
        plt.title(titleTxt)
    plt.axis("image")
    plt.tight_layout()
    if outdir_sub is not None:
        plt.savefig("%s/%s_pie.pdf" % (outdir_sub, filename))
        plt.close()


def getFeatN(feat_summary):
    # get maximum number of features from feat_summary
    s = np.concatenate([re.findall('feat_sources?(.*)$', n) for n in feat_summary.columns.values])
    return max([int(n) for n in s if n.isdigit()])


def plotImpSource(impRank, df, outdir_sub='./',
                  autopct='%0.1f%%', src_colors_dict=None):
    source_col = 'feat_source%d' % impRank
    df_counts = df[df[source_col] != ''].groupby(source_col)[source_col].count()
    src_colors = None if src_colors_dict is None else [src_colors_dict[s] for s in df_counts.index]
    plotCountsPie(df_counts,
                  'Data source summary (for %s important feature)' % int2ordinal(impRank),
                  'global_imprank-%d' % impRank,
                  outdir_sub,
                  autopct=autopct,
                  colors=src_colors)


def parseGenesets(fname):
    genesets = dict()
    f = open(fname)
    for gs in f:
        gs_name = re.sub('\\t\\t.*\\n', '', gs)
        genes = re.sub('.*\\t\\t', '', gs).replace('\t\n', '').split(sep='\t')
        genes = np.hstack(genes)
        genesets[gs_name] = genes
    f.close()

    return genesets


def isInSameGS(target, features, genesets):
    # check if both target and feature is in the same geneset
    # target is one value; features can be an array
    # requires: genesets
    if not isinstance(features, list):
        features = [features]
    isInBools = [(len(set(features).intersection(gs)) > 0) and (target not in features) and (target in gs) for _, gs in
                 genesets.items()]
    return sum(isInBools) > 0


def isInSameGS_sources(target, features, sources, genesets):
    # check if both target and feature is in the same geneset
    # return concatenated sources
    # requires: genesets
    if not isinstance(features, list):
        features = [features]
    if not isinstance(sources, list):
        sources = [sources]
    idx = [isInSameGS(target, f, genesets) for f in features]
    return '_'.join(pd.Series(sources)[idx].sort_values().unique())


def isInSameGene(target, features, func_args=''):
    # check if target (gene) is in features list
    # func_args is a placeholder argument
    if not isinstance(features, list):
        features = [features]
    return (target in features)


def isInSameGene_sources(target, features, sources, func_args=''):  # gene in features list of sources
    # func_args is a placeholder argument
    if not isinstance(features, list):
        features = [features]
    if not isinstance(sources, list):
        sources = [sources]
    idx = [isInSameGene(target, f) for f in features]
    return '_'.join(pd.Series(sources)[idx].sort_values().unique())


# get contribution by importance rank and by source
def getGrpCounts(isInSame_func, isInSame_sources_func, feat_summary, func_args=None):
    # tally counts for if feat and target are in the same group, with the same
    # assessed by the isInSame_func function
    topN = getFeatN(feat_summary)

    sameGs_counts = pd.DataFrame()
    sameGs_src_counts = pd.DataFrame()
    feat_summary_annot = feat_summary.copy()
    for i in range(1, topN + 1):
        gene_col = 'feat_gene%d' % i
        source_col = 'feat_source%d' % i

        # get counts for same gene between target and given i-th important feature
        sameGs_bool = [isInSame_func(g, f, func_args) for g, f in zip(feat_summary.target, feat_summary[gene_col])]
        sameGs_counts = sameGs_counts.append(pd.DataFrame({'importanceRank': [str(i)],
                                                           'count': [sum(sameGs_bool)]}))

        # break down by source, for the same gene (between target and i-th important feature)
        df = feat_summary.loc[sameGs_bool,]
        src_count = df[df[source_col] != ''].groupby(source_col)[source_col].count()
        c = pd.DataFrame({'source': src_count.index.values,
                          'count': src_count.values,
                          'importanceRank': str(i)})
        sameGs_src_counts = sameGs_src_counts.append(c)

        feat_summary_annot['inSame_%d' % i] = sameGs_bool

    # add in for the top N combined, if any gene in top N features are the same gene as the target
    sameGs_bool = [isInSame_func(g, f, func_args) for g, f in zip(feat_summary.target, feat_summary.feat_genes)]
    sameGs_counts = sameGs_counts.append(pd.DataFrame({'importanceRank': ['top%d' % topN],
                                                       'count': [sum(sameGs_bool)]}))

    feat_summary_annot['inSame_top%d' % topN] = sameGs_bool

    # add in for breakdown by source
    sameGs_src = [isInSame_sources_func(g, f, s, func_args) for g, f, s in
                  zip(feat_summary.target, feat_summary.feat_genes, feat_summary.feat_sources)]
    df = pd.DataFrame({'source': sameGs_src})
    src_count = df[df.source != ''].groupby('source')['source'].count()
    c = pd.DataFrame({'source': src_count.index.values,
                      'count': src_count.values,
                      'importanceRank': 'top%d' % topN})
    sameGs_src_counts = sameGs_src_counts.append(c)

    # calc percentages
    sameGs_counts['percent'] = sameGs_counts['count'] * 100 / feat_summary.shape[0]
    sameGs_src_counts['percent'] = sameGs_src_counts['count'] * 100 / feat_summary.shape[0]

    return sameGs_counts, sameGs_src_counts, feat_summary_annot


def getGrpCounts_fromFeatSummaryAnnot(feat_summary_annot, remove_zero=True):
    # get group counts, based on the feat_summary_annot file,
    # useful when reading in just the feat_summary_annot and need to recreate the sameGs_counts and sameGs_src_counts
    # NOTE, for the topx, here it is calculated differently (as the sum, grouped by source)
    # whereas in the original sameGs_counts/sameGs_src_counts, we can try concentate the sources
    # different calculations for different goals

    df1 = feat_summary_annot.loc[:, feat_summary_annot.columns.str.startswith('inSame')].apply(sum, axis=0)
    df1 = df1.to_frame(name='count')
    df1['percent'] = df1['count'] * 100 / feat_summary_annot.shape[0]
    df1['importanceRank'] = df1.index.str.extract('_(.*)').values

    topx_name = [re.findall('top.*', n) for n in df1['importanceRank'].unique() if re.match('top.*', n)][0][0]
    df2 = pd.DataFrame()
    for n in df1['importanceRank'].unique():
        if n != topx_name:
            df_tmp = feat_summary_annot.groupby('feat_source%s' % n)['inSame_%s' % n].apply(sum)
            df_tmp = df_tmp.to_frame(name='count')
            df_tmp['percent'] = df_tmp['count'] * 100 / feat_summary_annot.shape[0]
            df_tmp['importanceRank'] = n
            df_tmp['source'] = df_tmp.index

            df2 = pd.concat([df2, df_tmp], ignore_index=True, sort=False)

    df_tmp = df2.groupby('source')['count'].apply(sum)
    df_tmp = df_tmp.to_frame(name='count')
    df_tmp['percent'] = df_tmp['count'] * 100 / feat_summary_annot.shape[0]
    df_tmp['importanceRank'] = topx_name
    df_tmp['source'] = df_tmp.index

    df2 = pd.concat([df2, df_tmp], ignore_index=True, sort=False)

    sameGrp_counts = df1.loc[df1['count'] > 0, :].copy()
    sameGrp_src_counts = df2.loc[df2['count'] > 0, :].copy()

    return sameGrp_counts, sameGrp_src_counts


def plotGrpCounts(sameGrp_counts, sameGrp_src_counts, feat_summary_annot, pfx, outdir_sub='./'):
    if (np.logical_not(os.path.exists(outdir_sub))): os.mkdir(outdir_sub)

    # get prefix and create new subfolder, where outputs go
    pfx_cat = pfx.replace(' ', '')
    outdir_sub = '%s/%s/' % (outdir_sub, pfx_cat)
    if np.logical_not(os.path.exists(outdir_sub)): os.mkdir(outdir_sub)

    topN = getFeatN(feat_summary_annot)

    # -- csv
    feat_summary_annot.to_csv('%s/feat_summary_annot.csv' % (outdir_sub))

    if sameGrp_counts['count'].sum() < 1:
        # no matches to group (count=0), nothing more to do
        return True

    # -- plots
    # bar plot
    plt.figure()
    ax = sns.barplot('importanceRank', 'percent', data=sameGrp_counts, color='royalblue')
    ax.set(xlabel='Feature rank', ylabel='%s (percent of total targets)' % pfx)
    plt.tight_layout()
    plt.savefig("%s/impRank_bar_pct.pdf" % (outdir_sub))
    plt.close()

    plt.figure()
    ax = sns.barplot('importanceRank', 'count', data=sameGrp_counts, color='royalblue')
    ax.set(xlabel='Feature rank', ylabel='%s (count)' % pfx)
    plt.tight_layout()
    plt.savefig("%s/impRank_bar_n.pdf" % (outdir_sub))
    plt.close()

    plt.figure()
    df = sameGrp_src_counts.pivot('importanceRank', 'source')['percent']
    df = df.reindex(sameGrp_src_counts.importanceRank.unique())
    ax = df.plot(kind='bar', stacked=True)
    ax.set(xlabel='Feature rank', ylabel='%s (percent of total targets)' % pfx)
    plt.tight_layout()
    plt.savefig("%s/impRank_source_bar_pct.pdf" % (outdir_sub))
    plt.close()

    plt.figure()
    df = sameGrp_src_counts.pivot('importanceRank', 'source')['count']
    df = df.reindex(sameGrp_src_counts.importanceRank.unique())
    ax = df.plot(kind='bar', stacked=True)
    ax.set(xlabel='Feature rank', ylabel='%s (count)' % pfx)
    plt.tight_layout()
    plt.savefig("%s/impRank_source_bar_n.pdf" % (outdir_sub))
    plt.close()

    # pie charts
    def plotGrpPies(impRankTxt, pfx):
        if not any(sameGrp_counts.importanceRank.str.contains(impRankTxt)):
            return None

        pfx_cat = pfx.replace(' ', '')
        c = sameGrp_counts.loc[sameGrp_counts.importanceRank == impRankTxt, 'count'][0]
        df_counts = pd.Series({pfx: c,
                               'not %s' % pfx: feat_summary_annot.shape[0] - c})
        plotCountsPie(df_counts,
                      'Of the %s important feature' % int2ordinal(impRankTxt),
                      'imprank-%s' % (impRankTxt),
                      outdir_sub)

        # check the data source, of the ones where same gene as the target
        c = sameGrp_src_counts.loc[sameGrp_src_counts.importanceRank == impRankTxt,]
        df_counts = pd.Series(c['count'].values, index=c['source'])
        plotCountsPie(df_counts,
                      'Of the %s important feature, feat/target %s' % (int2ordinal(impRankTxt), pfx),
                      'imprank-%s_source' % (impRankTxt),
                      outdir_sub)

    plotGrpPies('1', pfx)  # proportion of genes where the top feature is the same gene as the target
    plotGrpPies('2', pfx)  # proportion of genes where the top 2nd feature is the same gene as the target
    plotGrpPies('top%d' % topN, pfx)

    # Score ranked


#    plt.figure(figsize=(50,7))
#    ax = sns.barplot('target','score_test',
#                data=feat_summary_annot.sort_values('score_test', ascending=False),
#                hue='inSame_top%d'%topN)
#    ax.set(xticklabels=[], xlabel='Target gene', ylabel='Score test')
#    plt.title(pfx)
#    plt.savefig("%s/score_test_rank.pdf" % (outdir_sub))
#    plt.close()

def generate_featSummary(varExp, outdir_sub='./'):
    topN = max(varExp.feat_idx)  # max number of features in reduced model
    varExp_noNeg = varExp.loc[varExp.score_ind > 0, :]
    feature_cat = varExp_noNeg.groupby('target')['feature'].apply(lambda x: ','.join(x))
    score_ind_cat = varExp_noNeg.groupby('target')['score_ind'].apply(lambda x: ','.join(round(x, 3).map(str)))
    feat_summary = varExp_noNeg.groupby('target')['target', 'score_rd', 'score_full'].first()

    feat_summary = feat_summary.merge(feature_cat, left_index=True, right_index=True)
    feat_summary = feat_summary.merge(score_ind_cat, left_index=True, right_index=True)

    feat_summary['feat_sources'] = feat_summary.apply(lambda x: getFeatSource(x['feature']), axis=1)
    feat_summary['feat_genes'] = feat_summary.apply(lambda x: getFeatGene(x['feature']), axis=1)
    for i in range(1, topN + 1):
        feat_summary['feat_gene%d' % i] = feat_summary.apply(
            lambda x: x.feat_genes[i - 1] if len(x.feat_genes) > (i - 1) else '',
            axis=1)  # get the nth most important feature, per gene
        feat_summary['feat_source%d' % i] = feat_summary.apply(
            lambda x: x.feat_sources[i - 1] if len(x.feat_sources) > (i - 1) else '',
            axis=1)  # get the nth most important feature, per gene

    feat_summary['feats_n'] = feat_summary.feat_genes.apply(lambda x: len(x))

    feat_summary.to_csv('%s/feat_summary.csv' % outdir_sub, index=False)

    return feat_summary


def plotFeatSrcCounts(feat_summary, outdir_sub='./'):
    if not os.path.exists(outdir_sub): os.mkdir(outdir_sub)

    # analyze feat_summary
    topN = getFeatN(feat_summary)
    for n in range(1, topN + 1):
        plotImpSource(n, feat_summary, outdir_sub)

    df_counts = pd.Series([y for x in feat_summary.feat_sources for y in x]).value_counts()
    plotCountsPie(df_counts,
                  'Data source summary (top %d features)' % topN,
                  'imprank-top%d' % topN,
                  outdir_sub)

    # number of top features per gene distribution


#    plt.figure()
#    ax = sns.countplot(feat_summary.feats_n, color='royalblue')
#    ax.set(xlabel='Number of features in model', ylabel='Number of genes (predicted)')
#    plt.title('Size of reduced model')
#    plt.savefig("%s/model_size_bar.pdf" % (outdir_sub))
#    plt.close()


######################################################################
# Feature analyses
# #####################################################################

def anlyz_varExp(varExp, suffix='', outdir_sub='./'):
    # summarize the scores; given _varExp data
    if not os.path.exists(outdir_sub): os.mkdir(outdir_sub)

    # Score grouped by
    plt.figure()
    ax = sns.boxplot('feat_idx', 'score_ind', data=varExp.loc[varExp.score_ind > 0,], color='royalblue')
    ax.set(xlabel='Feature rank', ylabel='Score (univariate)', yscale='log')
    plt.title('Score (univariate)\nnegative score excluded; %s' % suffix)
    plt.savefig("%s/grp_score_uni_by_featRank_log.pdf" % (outdir_sub))
    plt.close()

    plt.figure()
    ax = sns.violinplot('feat_idx', 'score_ind', data=varExp.loc[varExp.score_ind > 0,], color='royalblue')
    ax.set(xlabel='Feature rank', ylabel='Score (univariate)')
    plt.title('Score (univariate)\nnegative score excluded; %s' % suffix)
    plt.savefig("%s/grp_score_uni_by_featRank.pdf" % (outdir_sub))
    plt.close()

    plt.figure()
    ax = sns.violinplot('feat_source', 'score_ind', data=varExp.loc[varExp.score_ind > 0, :], alpha=0.1, jitter=True)
    ax.set(xlabel='Feature source', ylabel='Score (univariate)')
    plt.title('Score (univariate), grouped by source\nnegative score excluded; %s' % suffix)
    plt.savefig("%s/grp_score_uni_by_featSource.pdf" % (outdir_sub))
    plt.close()

    df = varExp.groupby('feat_idx')['score_ind'].apply(np.nanmedian)
    plt.figure()
    ax = sns.barplot(df.index, df, color='royalblue')
    ax.set(xlabel='Feature rank', ylabel='median score (univariate)')
    plt.title('median score (univariate), grouped by feature rank; %s' % suffix)
    plt.savefig("%s/grp_score_uni_by_featRank_med_raw.pdf" % (outdir_sub))
    plt.close()

    df = varExp.loc[varExp.score_ind > 0,].groupby('feat_idx')['score_ind'].apply(np.nanmedian)
    plt.figure()
    ax = sns.barplot(df.index, df, color='royalblue')
    ax.set(xlabel='Feature rank', ylabel='median score (univariate)')
    plt.title('median score (univariate), grouped by feature rank\nnegative score excluded; %s' % suffix)
    plt.savefig("%s/grp_score_uni_by_featRank_med.pdf" % (outdir_sub))
    plt.close()

    df = varExp.loc[varExp.score_ind > 0,].groupby('feat_source')['score_ind'].apply(np.nanmedian)
    plt.figure()
    ax = sns.barplot(df.index, df, color='royalblue')
    ax.set(xlabel='Feature source', ylabel='median score (univariate)')
    plt.title('median score (univariate), grouped by source\nnegative score excluded; %s' % suffix)
    plt.savefig("%s/grp_score_uni_by_featSource_med.pdf" % (outdir_sub))
    plt.close()

    # Score distributions
    score_vals_all = varExp.loc[varExp.score_full > 0,].groupby('target')['score_full'].apply(lambda x: x.iloc[0])
    plt.figure()
    ax = sns.distplot(score_vals_all)
    ax.set(xlabel='Score of full model', ylabel='Count')
    plt.title('Distribution of score (full model)\nnegative score excluded; %s' % suffix)
    plt.savefig("%s/score_dist_all.pdf" % (outdir_sub))
    plt.close()

    score_vals_rd = varExp.loc[varExp.score_rd > 0,].groupby('target')['score_rd'].apply(lambda x: x.iloc[0])
    plt.figure()
    ax = sns.distplot(score_vals_rd)
    ax.set(xlabel='Score of reduced model', ylabel='Count')
    plt.title('Distribution of score (reduced model)\nnegative score excluded; %s' % suffix)
    plt.savefig("%s/score_dist_rd.pdf" % (outdir_sub))
    plt.close()

    score_vals_uni = varExp.score_ind[varExp.score_ind > 0]
    plt.figure()
    ax = sns.distplot(score_vals_uni)
    ax.set(xlabel='Score of univariate', ylabel='Count')
    plt.title('Distribution of score (univariate model)\nnegative score excluded; %s' % suffix)
    plt.savefig("%s/score_dist_uni.pdf" % (outdir_sub))
    plt.close()

    score_stats = pd.DataFrame({'univariate': score_vals_uni.describe(),
                                'reduced': score_vals_rd.describe(),
                                'full': score_vals_all.describe()})
    score_stats.to_csv("%s/stats_score.csv" % (outdir_sub))

    # Score compares
    df_fullrd = varExp.groupby('target')[['score_full',
                                          'score_rd']].first()  # for full/rd, keep first, the rest are redundant (row is unique by univariate)
    df = pd.concat([pd.DataFrame({'score': df_fullrd.score_full, 'label': 'full model'}),
                    pd.DataFrame({'score': df_fullrd.score_rd, 'label': 'reduced model'}),
                    pd.DataFrame({'score': varExp.score_ind, 'label': 'univariate'})])
    plt.figure()
    ax = sns.boxplot(x='label', y='score', data=df.loc[df.score > 0, :], color='royalblue')
    ax.set(xlabel='Model', ylabel='Score')
    plt.title('Score\nnegative score excluded; %s' % suffix)
    plt.savefig("%s/compr_score_boxplot.pdf" % (outdir_sub))
    plt.close()

    df = varExp.loc[varExp.score_full > 0, :]
    ax = sns.scatterplot(df.score_rd, df.score_full, s=40, alpha=0.03, color='steelblue')
    ax.plot([0, 0.9], [0, 0.9], ls="--", c=".3")
    ax.set(xlabel='Score reduced model', ylabel='Score full model')
    plt.title('Score\nnegative score (full) excluded; %s' % suffix)
    plt.savefig("%s/compr_score_scatter.pdf" % (outdir_sub))
    plt.close()

    df = varExp
    ax = sns.scatterplot(df.score_rd, df.score_full, s=40, alpha=0.03, color='steelblue')
    ax.plot([0, 0.9], [0, 0.9], ls="--", c=".3")
    ax.set(xlabel='Score reduced model', ylabel='Score full model')
    plt.title('Score\n%s' % suffix)
    plt.savefig("%s/compr_score_scatter_all.pdf" % (outdir_sub))
    plt.close()

    # Score ratios
    plt.figure()
    ax = sns.boxplot('feat_idx', 'varExp_ofFull',
                     data=varExp.loc[(varExp.varExp_ofFull > 0) & (np.abs(varExp.varExp_ofFull) != np.inf),],
                     color='royalblue')
    ax.set(xlabel='Feature rank', ylabel='Score of univariate / score of full model',
           yscale='log')
    plt.title('Proportion of score (univariate vs full model)\nnegative score excluded; %s' % suffix)
    plt.savefig("%s/ratio_score_UniVsFull.pdf" % (outdir_sub))
    plt.close()

    plt.figure()
    ax = sns.boxplot('feat_idx', 'varExp_ofRd',
                     data=varExp.loc[(varExp.varExp_ofRd > 0) & (np.abs(varExp.varExp_ofRd) != np.inf),],
                     color='royalblue')
    ax.set(xlabel='Feature rank', ylabel='Score of univariate / score of reduced model',
           yscale='log')
    plt.title('Proportion of score (univariate vs reduced model)\nnegative score excluded; %s' % suffix)
    plt.savefig("%s/ratio_score_UniVsRd.pdf" % (outdir_sub))
    plt.close()

    df = varExp.loc[varExp.varExp_ofFull > 0,].groupby('feat_idx')['varExp_ofFull'].apply(np.nanmedian)
    plt.figure()
    ax = sns.barplot(df.index, df, color='royalblue')
    ax.set(xlabel='Feature rank', ylabel='Score of univariate / score of full model')
    plt.title('Proportion of score (univariate vs full model)\nnegative score excluded; %s' % suffix)
    plt.savefig("%s/ratio_score_UniVsFull_med.pdf" % (outdir_sub))
    plt.close()

    df = varExp.loc[varExp.varExp_ofRd > 0,].groupby('feat_idx')['varExp_ofRd'].apply(np.nanmedian)
    plt.figure()
    ax = sns.barplot(df.index, df, color='royalblue')
    ax.set(xlabel='Feature rank', ylabel='Score of univariate / score of reduced model')
    plt.title('Proportion of score (univariate vs reduced model)\nnegative score excluded; %s' % suffix)
    plt.savefig("%s/ratio_score_UniVsRd_med.pdf" % (outdir_sub))
    plt.close()

    plt.figure()
    ax = sns.distplot(varExp.varExp_ofRd[(varExp.varExp_ofRd > 0) & (np.abs(varExp.varExp_ofRd) != np.inf)])
    ax.set(xlabel='Score of univariate / score of reduced model', ylabel='Count')
    plt.title('Distribution of score univariate / score reduced\nnegative score excluded; %s' % suffix)
    plt.savefig("%s/ratio_score_dist_UniVsRd.pdf" % (outdir_sub))
    plt.close()

    # number of features
    feats_n = varExp.groupby('target')[['target', 'score_rd', 'score_full']].apply(lambda x: x.iloc[0, :]).copy()
    n = varExp.loc[varExp.score_ind > 0, :].groupby('target')['target'].count()
    feats_n['N'] = 0
    feats_n.loc[n.index, 'N'] = n

    plt.figure()
    ax = sns.boxplot('N', 'score_rd', data=feats_n.loc[feats_n.N > 0, :], color='royalblue')
    ax.set(xlabel='No of features', ylabel='Score (reduced model)')
    plt.title('Score of target gene, stratified by number of features\nnegative score excluded; %s' % suffix)
    plt.savefig("%s/nFeat_score_rd.pdf" % (outdir_sub))
    plt.close()

    plt.figure()
    ax = sns.countplot(feats_n.N, color='royalblue')
    ax.set(xlabel='No of features', ylabel='Count (target genes)')
    plt.title('Number of features, per target gene\nnegative score excluded; %s' % suffix)
    plt.savefig("%s/nFeat.pdf" % (outdir_sub))
    plt.close()

    # statistics
    f = open("%s/stats_score_median.txt" % (outdir_sub), "w")
    f.write('Median score (full model): %0.2f\n' % np.nanmedian(score_vals_all))
    f.write('Median score (reduced model, top %d): %0.2f\n' % (max(varExp.feat_idx), np.nanmedian(score_vals_rd)))

    for n in range(1, max(varExp.feat_idx) + 1):
        f.write('Median score (univariate, %s feature): %0.2f\n' % (
        int2ordinal(n), np.nanmedian(varExp.loc[varExp.feat_idx == n, 'score_ind'].astype(float))))

    for n in range(1, max(varExp.feat_idx) + 1):
        f.write('Median score (univariate, %s feature, non-neg score): %0.2f\n' % (int2ordinal(n), np.nanmedian(
            varExp.loc[varExp.score_ind > 0,].loc[varExp.feat_idx == n, 'score_ind'].astype(float))))

    f.close()


def anlyz_varExp_wSource(varExp, dm_data=None, suffix='', outdir_sub='./', ):
    # analyze the model results, based on merge with raw source data
    if not os.path.exists(outdir_sub): os.mkdir(outdir_sub)

    if dm_data is None:
        # if dm_data is not given, then try to retrieve it
        from src.ceres_infer.data import depmap_data
        dm_data = depmap_data()
        dm_data.dir_datasets = '../datasets/DepMap/'
        dm_data.load_data()
        dm_data.preprocess_data()

    # merge with source CERES
    crispr = dm_data.df_crispr.copy()
    crispr.columns = pd.Series(crispr.columns).apply(getFeatGene, firstOnly=True)
    crispr_stats = crispr.describe()

    feat_withceres = varExp.groupby('target').first().reset_index(drop=False).loc[:,
                     ['target', 'score_rd', 'score_full']]
    feat_withceres = pd.merge(feat_withceres, crispr_stats.T, how='left', left_on='target', right_index=True)
    feat_withceres.to_csv('%s/merge_ceres_score_merge.csv' % outdir_sub)

    plt.figure()
    ax = sns.scatterplot('mean', 'score_rd', data=feat_withceres, s=60, alpha=0.5)
    ax.set(ylabel='Score (reduced model)', xlabel='CERES (mean)')
    plt.title('Score of reduced model vs mean CERES; %s' % suffix)
    plt.savefig("%s/merge_ceres_scoreVsMean.pdf" % (outdir_sub))
    plt.close()

    plt.figure()
    ax = sns.scatterplot('std', 'score_rd', data=feat_withceres, s=60, alpha=0.5)
    ax.set(ylabel='Score (reduced model)', xlabel='CERES (standard deviation)')
    plt.title('Score of reduced model vs std CERES; %s' % suffix)
    plt.savefig("%s/merge_ceres_scoreVsSD.pdf" % (outdir_sub))
    plt.close()

    plt.figure()
    ax = sns.scatterplot('mean', 'std', data=feat_withceres, s=60, alpha=0.5)
    ax.set(xlabel='CERES (mean)', ylabel='CERES (standard deviation)')
    plt.title('mean CERES vs SD CERES ; %s' % suffix)
    plt.savefig("%s/merge_ceres_meanVsSD.pdf" % (outdir_sub))
    plt.close()

    df = feat_withceres.copy()
    df.dropna(subset=['score_rd', 'mean', 'std'], inplace=True)
    df1 = df.loc[df.score_rd <= 0.2, :]
    df2 = df.loc[df.score_rd > 0.2, :]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df1['mean'], df1['std'], df1['score_rd'], s=50, alpha=0.05, color='darkgray')
    ax.scatter(df2['mean'], df2['std'], df2['score_rd'], s=50, alpha=0.1, color='darkred')
    ax.set(xlabel='CERES (mean)', ylabel='CERES (SD)', zlabel='Score (reduced model)')
    ax.view_init(azim=-120, elev=30)
    plt.savefig("%s/merge_ceres_3d_meanSD.pdf" % (outdir_sub))
    plt.close()


def anlyz_varExp_feats(varExp, gs_dir='../datasets/gene_sets/', outdir_sub='./'):
    # analyze features
    if not os.path.exists(outdir_sub): os.mkdir(outdir_sub)

    feat_summary = generate_featSummary(varExp, outdir_sub)

    plotFeatSrcCounts(feat_summary, '%s/featSrcCounts/' % outdir_sub)

    # analyze overlay with gene sets
    genesets_combined = dict()

    # in same gene sets (KEGG)
    genesets = parseGenesets('%s/KEGG_2019_Human.txt' % gs_dir)
    genesets_combined.update(genesets)
    sameGs_counts, sameGs_src_counts, feat_summary_annot = getGrpCounts(isInSameGS, isInSameGS_sources, feat_summary,
                                                                        genesets)
    plotGrpCounts(sameGs_counts, sameGs_src_counts, feat_summary_annot, 'in same gene set KEGG', outdir_sub)

    # in same gene sets (Reactome)
    genesets = parseGenesets('%s/Reactome_2016.txt' % gs_dir)
    genesets_combined.update(genesets)
    sameGs_counts, sameGs_src_counts, feat_summary_annot = getGrpCounts(isInSameGS, isInSameGS_sources, feat_summary,
                                                                        genesets)
    plotGrpCounts(sameGs_counts, sameGs_src_counts, feat_summary_annot, 'in same gene set Reactome', outdir_sub)

    # in same gene sets (Panther)
    genesets = parseGenesets('%s/Panther_2016.txt' % gs_dir)
    genesets_combined.update(genesets)
    sameGs_counts, sameGs_src_counts, feat_summary_annot = getGrpCounts(isInSameGS, isInSameGS_sources, feat_summary,
                                                                        genesets)
    plotGrpCounts(sameGs_counts, sameGs_src_counts, feat_summary_annot, 'in same gene set Panther', outdir_sub)

    # in same gene sets (Panther/KEGG/Reactome)
    genesets = genesets_combined
    sameGs_counts, sameGs_src_counts, feat_summary_annot = getGrpCounts(isInSameGS, isInSameGS_sources, feat_summary,
                                                                        genesets)
    plotGrpCounts(sameGs_counts, sameGs_src_counts, feat_summary_annot, 'in same gene set KEGG-Panther-Reactome',
                  outdir_sub)

    # in same gene
    sameGene_counts, sameGene_src_counts, feat_summary_annot = getGrpCounts(isInSameGene, isInSameGene_sources,
                                                                            feat_summary)
    plotGrpCounts(sameGene_counts, sameGene_src_counts, feat_summary_annot, 'on same gene', outdir_sub)

    # in same paralog
    genesets = parseGenesets('%s/paralogs.txt' % gs_dir)
    genesets_combined.update(genesets)
    sameGs_counts, sameGs_src_counts, feat_summary_annot = getGrpCounts(isInSameGS, isInSameGS_sources, feat_summary,
                                                                        genesets)
    plotGrpCounts(sameGs_counts, sameGs_src_counts, feat_summary_annot, 'in same paralog', outdir_sub)


def anlyz_scoresGap(varExp, useGene_dependency, outdir_sub='./'):
    # 'score' is used in the var names here, but since for AUC metrics, we
    # will look at the gain (score - 0.5), the plots and outputs we will call it 'gain'
    if useGene_dependency:
        # the score would be AUC, just focus on feats with AUC>0.5
        # and will assess based on deviation from 0.5
        df = varExp.loc[varExp.score_ind > 0.5, :].copy()
        df.score_full = df.score_full - 0.5
        df.score_rd = df.score_rd - 0.5
        df.score_ind = df.score_ind - 0.5
    else:
        # the score would be R2, just focus on feats with R2>0
        df = varExp.loc[varExp.score_ind > 0, :].copy()

    score_fullrd = df.groupby('target').first().loc[:, ['score_full', 'score_rd']]
    featsN = df.groupby('target')['target'].count()
    featsN.name = 'featsN'
    sum_score_ind = df.groupby('target')['score_ind'].apply(sum)
    sum_score_ind.name = 'sum_score_ind'

    scoreVals = pd.concat([featsN, sum_score_ind], axis=1)
    scoreVals = scoreVals.merge(score_fullrd, left_index=True, right_index=True)
    scoreVals.reset_index(drop=False, inplace=True)
    scoreVals['score_gap'] = scoreVals.score_rd - scoreVals.sum_score_ind
    scoreVals['score_gap_frac'] = scoreVals.sum_score_ind / scoreVals.score_rd

    # plots and stats
    plt.figure()
    ax = sns.distplot(scoreVals.score_gap)
    ax.set(xlabel='Gain (reduced model) - gain (sum of score (univariate))', ylabel='Count')
    plt.savefig('%s/gain_gap.pdf' % outdir_sub)
    plt.close()

    plt.figure()
    ax = sns.distplot(scoreVals.score_gap_frac[np.abs(scoreVals.score_gap_frac) != np.inf])
    ax.set(xlabel='Gain (sum of score (univariate))/gain (reduced model)', ylabel='Fraction')
    plt.savefig('%s/gain_gap_frac.pdf' % outdir_sub)
    plt.close()

    totalN = scoreVals.shape[0]
    f = open('%s/gain_gap_stats.txt' % outdir_sub, 'w')
    f.write('Fraction of data with (rd - sum(ind)) > 0: %0.3f\n' % (sum(scoreVals.score_gap > 0) / totalN))
    f.write('Fraction of data with (sum(ind)/rd) < 80%% and positive: %0.3f\n' % (
                sum(scoreVals.score_gap_frac < 0.8) / totalN))
    f.close()

    scoreVals.to_csv('%s/gain_gap.csv' % outdir_sub, index=False)


def genBarPlotGene(model_results, gene, score_name, lineVal=None, outdir_sub='./'):
    # generate bar plot, given the model results and gene
    # if lineVal is not None, then will try a dotted line at the given value
    df = model_results.copy()
    df = df[df.target == gene]
    if df.shape[0] < 1:
        logging.warning('Gene %s not found in results' % gene)
        return None
    df['feature'][df.model == 'topfeat'] = 'topfeat'

    plt.figure()
    ax = sns.barplot('feature', score_name, data=df, color='royalblue')
    if lineVal is not None:
        ax.plot([-0.5, max(ax.get_xticks()) + 0.5], [lineVal, lineVal], 'r--', alpha=.75)
    ax.set(ylabel='Score', xlabel='')
    ax.set(ylim=[-0.3, 0.9])
    plt.title('Target gene: %s' % (gene))
    plt.xticks(rotation=-30, horizontalalignment="left")
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.savefig("%s/%s_score_bar.pdf" % (outdir_sub, gene))
    plt.close()


######################################################################
# Aggregate summary
# #####################################################################

def anlyz_aggRes(aggRes,ext_set_name='sanger', suffix='', outdir_sub='./'):
    # summarize the scores; given _varExp data
    if not os.path.exists(outdir_sub): os.mkdir(outdir_sub)
    if aggRes.empty: return None

    score_vals_full = aggRes.groupby('target')['score_full'].apply(lambda x: x.iloc[0])
    plt.figure()
    ax = sns.distplot(score_vals_full)
    ax.set(xlabel='Score of full model', ylabel='Count')
    plt.title('Distribution of score (full model); %s' % suffix)
    plt.savefig("%s/score_dist_full.pdf" % (outdir_sub))
    plt.close()

    score_vals_rd = aggRes.groupby('target')['score_rd'].apply(lambda x: x.iloc[0])
    plt.figure()
    ax = sns.distplot(score_vals_rd)
    ax.set(xlabel='Score of reduced model', ylabel='Count')
    plt.title('Distribution of score (reduced model); %s' % suffix)
    plt.savefig("%s/score_dist_rd.pdf" % (outdir_sub))
    plt.close()

    score_vals_rd10 = aggRes.groupby('target')['score_rd10'].apply(lambda x: x.iloc[0])
    plt.figure()
    ax = sns.distplot(score_vals_rd10)
    ax.set(xlabel='Score of reduced model (top 10 feat)', ylabel='Count')
    plt.title('Distribution of score (reduced model top10 feat); %s' % suffix)
    plt.savefig("%s/score_dist_rd10.pdf" % (outdir_sub))
    plt.close()

    score_stats = pd.DataFrame({'full': score_vals_full.describe(),
                                'reduced': score_vals_rd.describe(),
                                'reduced10feat': score_vals_rd10.describe()})
    score_stats.to_csv("%s/stats_score.csv" % (outdir_sub))

    # Score compares
    df = pd.concat([pd.DataFrame({'score': aggRes.score_full, 'label': 'full model'}),
                    pd.DataFrame({'score': aggRes.score_rd, 'label': 'reduced model'}),
                    pd.DataFrame({'score': aggRes.score_rd10, 'label': 'reduced model top10 feat'})])
    plt.figure()
    ax = sns.boxplot(x='label', y='score', data=df.loc[df.score > 0, :], color='royalblue')
    ax.set(xlabel='Model', ylabel='Score')
    plt.title('Score; %s' % suffix)
    plt.savefig("%s/compr_score_boxplot.pdf" % (outdir_sub))
    plt.close()

    plt.figure()
    ax = sns.scatterplot(aggRes.score_rd, aggRes.score_full, s=60, alpha=0.1, color='steelblue')
    ax.plot([0, 0.9], [0, 0.9], ls="--", c=".3")
    ax.set(xlabel='Score reduced model', ylabel='Score full model')
    plt.title('Score; %s' % suffix)
    plt.savefig("%s/compr_score_scatter.pdf" % (outdir_sub))
    plt.close()

    plt.figure()
    ax = sns.scatterplot(aggRes.score_rd, aggRes.score_rd10, s=60, alpha=0.1, color='steelblue')
    ax.plot([0, 0.9], [0, 0.9], ls="--", c=".3")
    ax.set(xlabel='Score reduced model', ylabel='Score reduced model top10 feat')
    plt.title('Score; %s' % suffix)
    plt.savefig("%s/compr_score_scatter_top10.pdf" % (outdir_sub))
    plt.close()

    # recall compares
    plt.figure()
    ax = sns.scatterplot(aggRes.recall_rd10, aggRes[ext_set_name+'_recall_rd10'], s=60, alpha=0.1, color='steelblue')
    ax.plot([0, 1.0], [0, 1.0], ls="--", c=".3")
    ax.set(xlabel='Recall P19Q3 test set (rd10 model)', ylabel=f'Recall {ext_set_name} (rd10 model)')
    plt.title('Recall; %s' % suffix)
    plt.savefig("%s/compr_recall_scatter_q3_%s.pdf" % (outdir_sub,ext_set_name))
    plt.close()

    # recall vs score
    plt.figure()
    ax = sns.scatterplot(aggRes.score_rd10, aggRes.recall_rd10, s=60, alpha=0.1, color='steelblue')
    ax.set(xlabel='Score (rd10 model)', ylabel='Recall (rd10 model)', xlim=(0, 1.1), ylim=(0, 1.1))
    plt.title('Test set; %s' % suffix)
    plt.savefig("%s/score_recall.pdf" % (outdir_sub))
    plt.close()

    plt.figure()
    ax = sns.scatterplot(aggRes[ext_set_name+'_score_rd10'], aggRes[ext_set_name+'_recall_rd10'], s=60, alpha=0.1, color='steelblue')
    ax.set(xlabel='Score (rd10 model)', ylabel='Recall (rd10 model)', xlim=(0, 1.1), ylim=(0, 1.1))
    plt.title('%s; %s' % (ext_set_name,suffix))
    plt.savefig("%s/score_recall_{ext_set_name}.pdf" % (outdir_sub))
    plt.close()


def anlyz_model_results(model_results, suffix='', outdir_sub='./'):
    # similar to anlyz_aggRes, but instead of taking in the aggregated results data frame
    # this method takes in the model_results data frame
    # summarize the scores; given model_results data frame
    if not os.path.exists(outdir_sub): os.mkdir(outdir_sub)
    if model_results.empty:
        return None

    df_results = model_results.copy()
    df_results = df_results.loc[df_results.model.str.match('(all|topfeat|top10feat)'), :]

    # Score compares with train vs test
    df1 = df_results[['model', 'score_train']].copy()
    df1.rename(columns={'score_train': 'score'}, inplace=True)
    df1['score_type'] = 'score_train'
    df2 = df_results[['model', 'score_test']].copy()
    df2['score_type'] = 'score_test'
    df2.rename(columns={'score_test': 'score'}, inplace=True)
    df = pd.concat([df1, df2])

    ax = sns.boxplot(x='model', hue='score_type', y='score', data=df)
    ax.set(xlabel='Model', ylabel='Score')
    plt.title('Score (train vs test); %s' % suffix)
    plt.savefig("%s/compr_score_train-test_boxplot.pdf" % (outdir_sub))
    plt.close()

    ax = sns.distplot(df_results.score_train - df_results.score_test)
    ax.set(xlabel='[Score train - Score test]', ylabel='Count')
    plt.title('Difference between model score train and test; %s' % suffix)
    plt.savefig("%s/compr_score_train-test_distplot.pdf" % (outdir_sub))
    plt.close()


def constructYCompr(genes2analyz, compr_pfx, outdir_modtmp):
    # Extract y actual and predicted from pickle file and format into two data frames, respectively; for all given genes
    # compr_pfx specifies the prefix, e.g. tr, te

    df_y_actual = pd.DataFrame()
    df_y_pred = pd.DataFrame()
    for gene2analyz in genes2analyz:
        y_compr = pickle.load(open('%s/y_compr_%s.pkl' % (outdir_modtmp, gene2analyz), "rb"))

        df_y_actual = pd.concat(
            [df_y_actual, pd.DataFrame(y_compr[compr_pfx]['y_actual'].values, columns=[gene2analyz])], axis=1)
        df_y_pred = pd.concat(
            [df_y_pred, pd.DataFrame(y_compr[compr_pfx]['y_pred'].values, columns=[gene2analyz])], axis=1)

    return df_y_actual, df_y_pred


def yComprHeatmap(df_y_actual, df_y_pred, pfx, outdir_ycompr):
    # heatmap
    plt.figure()
    ax = sns.heatmap(df_y_actual, yticklabels=False, xticklabels=False, vmin=-5, vmax=5, cmap='RdBu')
    ax.set(xlabel='Genes', ylabel='Cell lines')
    plt.savefig("%s/%s_heatmap_yactual.png" % (outdir_ycompr, pfx))
    plt.close()

    plt.figure()
    ax = sns.heatmap(df_y_pred, yticklabels=False, xticklabels=False, vmin=-5, vmax=5, cmap='RdBu')
    ax.set(xlabel='Genes', ylabel='Cell lines')
    plt.savefig("%s/%s_heatmap_yinferred.png" % (outdir_ycompr, pfx))
    plt.close()


def getConcordance(df, threshold=-0.6):
    df['concordance'] = 0
    df.loc[(df.y_actual <= threshold) & (df.y_pred <= threshold), 'concordance'] = 1
    df.loc[(df.y_actual > threshold) & (df.y_pred > threshold), 'concordance'] = 1
    return sum(df.concordance == 1) / len(df)
