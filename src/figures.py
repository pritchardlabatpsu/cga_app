import pandas as pd
import os
from ast import literal_eval
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import pickle
import networkx as nx
from matplotlib_venn import venn2
from ceres_infer.analyses import *
from ceres_infer.data import stats_Crispr, scale_data

# settings
plt.interactive(False)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(font_scale=1.5)
sns.set_style("white")

# output directory
dir_out = './manuscript/figures/'

src_colors = {'CERES':(214/255, 39/255, 40/255, 1.0), #red
              'RNA-seq':(31/255, 119/255, 180/255, 1.0), #blue
              'CN':(255/255, 127/255, 14/255, 1.0), #orange
              'Mut':(44/255, 160/255, 44/255, 1.0), #green
              'Lineage':(188/255, 189/255, 34/255, 1.0), #yellow
              'nan':(220/255, 220/255, 220/255, 1.0)} #grey

if not os.path.exists(dir_out):
    os.makedirs(dir_out)

#------ Figure 1 data source -----------
# read in data
dm_data = pickle.load(open('./out/20.0216 feat/reg_rf_boruta/dm_data.pkl','rb'))

# pie chart
df_counts = pd.DataFrame([{'CERES':dm_data.df_crispr.shape[1],
                           'RNA-seq':dm_data.df_rnaseq.shape[1],
                           'CN':dm_data.df_cn.shape[1],
                           'Mut':dm_data.df_mut.shape[1],
                           'Lineage':dm_data.df_lineage.shape[1]}])

plotCountsPie(df_counts.T[0],
              'Data source',
              'fig1_datasrc_pie',
              dir_out,
              autopct='%0.2f%%',
              colors=[src_colors[s] for s in df_counts.T.index])

# heatmaps - ceres
plt.figure(figsize=(5,4))
ax = sns.heatmap(dm_data.df_crispr,yticklabels=False, xticklabels=False, cmap='RdBu', vmin=-1, vmax=1, cbar=False)
ax.set(xlabel='Genes', ylabel='Cell lines', title='CERES')
plt.tight_layout()
plt.savefig("%s/fig1_datasrc_heatmap_ceres.png" % dir_out)
plt.close()

# heatmaps - rna-seq
df = scale_data(dm_data.df_rnaseq, [], None)
df = pd.DataFrame(df[0], index=dm_data.df_rnaseq.index, columns = dm_data.df_rnaseq.columns)
plt.figure(figsize=(10,4))
ax = sns.heatmap(df, yticklabels=False, xticklabels=False, cmap='RdBu', vmin=-1, vmax=1, cbar=False)
ax.set(xlabel='Genes', ylabel='Cell lines', title='RNA-seq')
plt.tight_layout()
plt.savefig("%s/fig1_datasrc_heatmap_rnaseq.png" % dir_out)
plt.close()

# heatmaps - cn
df = scale_data(dm_data.df_cn, [], None)
df = pd.DataFrame(df[0], index=dm_data.df_cn.index, columns = dm_data.df_cn.columns)
plt.figure(figsize=(5,4))
ax = sns.heatmap(df, yticklabels=False, xticklabels=False, cmap='RdBu', vmin=-1, vmax=1, cbar=False)
ax.set(xlabel='Genes', ylabel='Cell lines', title='Copy number')
plt.tight_layout()
plt.savefig("%s/fig1_datasrc_heatmap_cn.png" % dir_out)
plt.close()

# heatmaps - mutation
plt.figure(figsize=(5,4))
ax = sns.heatmap(dm_data.df_mut,yticklabels=False, xticklabels=False, cmap='RdBu', vmin=-1, vmax=1, cbar=False)
ax.set(xlabel='Genes', ylabel='Cell lines', title='Mutation')
plt.tight_layout()
plt.savefig("%s/fig1_datasrc_heatmap_mutation.png" % dir_out)
plt.close()

# heatmaps - lineage
plt.figure(figsize=(5,4))
ax = sns.heatmap(dm_data.df_lineage,yticklabels=False, xticklabels=False, cmap='RdBu', vmin=-1, vmax=1, cbar=False)
ax.set(xlabel='Lineages', ylabel='Cell lines', title='Lineage')
plt.tight_layout()
plt.savefig("%s/fig1_datasrc_heatmap_lineage.png" % dir_out)
plt.close()

# Supp
df_crispr_stats = stats_Crispr(dm_data)

plt.figure()
ax = sns.scatterplot(x='avg',y='std', data=df_crispr_stats,s=90)
ax.set(xlabel='mean (CERES)', ylabel='SD (CERES)')
plt.tight_layout()
plt.savefig("%s/fig1supp_scatter_mean_sd.pdf" % dir_out)
plt.close()

# table of features with lineage
# generated manually, based on analyses in notebook

#------ Figure 1 model related -----------
# read in data
dir_in_res = './out/20.0216 feat/reg_rf_boruta'
dir_in_anlyz = os.path.join(dir_in_res, 'anlyz_filtered')
df_varExp = pd.read_csv(os.path.join(dir_in_anlyz, 'feat_summary_varExp_filtered.csv'), header=0) #feature summary, more score metrics calculated
df_aggRes = pd.read_csv(os.path.join(dir_in_anlyz, 'agg_summary_filtered.csv')) #aggregated feat summary
dep_class = pd.read_csv('../out/20.0817 proc_data_baseline/gene_effect/gene_essential_classification.csv', header=None, index_col=0, squeeze=True)

# bar chart of scores (multivariate vs univariate of all contributing features)
df_tmp = df_varExp.merge(dep_class.to_frame(name='target_dep_class'), left_index=False, right_index=True, left_on='target')
df_rd = df_tmp.groupby('target')[['score_rd', 'target_dep_class']].first()
df = pd.concat([pd.DataFrame({'score':df_rd.score_rd,
                              'label':'multivariate',
                              'target_dep_class':df_rd.target_dep_class}),
                pd.DataFrame({'score':df_varExp.score_ind,
                              'label':'univariate',
                              'target_dep_class':df_tmp.target_dep_class})
               ])

plt.figure()
ax = sns.boxplot(x='label',y='score',data=df.loc[df.score>0,:], color='royalblue', hue='target_dep_class')
ax.set(xlabel='Model', ylabel='Score')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("%s/fig1_compr_score_boxplot.pdf" % dir_out)
plt.close()

# scatter plot (multivariate vs most important univariate)
df = df_varExp.loc[df_varExp.feat_idx == 1,:]
plt.figure()
plt.plot([-0.3,1],[-0.3,1], ls="--", c='0.3')
ax = sns.scatterplot(df.score_rd, df.score_ind, s=40, alpha=0.8, linewidth=0, color='steelblue')
ax.set(xlabel='Model score (multivariate)', ylabel='Model score (univariate, top feature)',
       xlim=[-0.3,1], ylim=[-0.3,1])
plt.tight_layout()
plt.savefig("%s/fig1_compr_score_scatter.pdf" % dir_out)
plt.close()

# Supp Figs
# top 10 feat vs all important feat - bar
df = pd.concat([pd.DataFrame({'score':df_aggRes.score_rd, 'label':'All selected features'}),
                pd.DataFrame({'score':df_aggRes.score_rd10, 'label':'Top 10 features'})])
plt.figure()
ax = sns.boxplot(x='label',y='score',data=df.loc[df.score>0,:], color='royalblue')
ax.set(xlabel='Model', ylabel='Score')
plt.tight_layout()
plt.savefig("%s/fig1supp_compr_score_boxplot.pdf" % dir_out)
plt.close()

# top 10 feat vs all important feat - scatter
plt.figure()
ax = sns.scatterplot(df_aggRes.score_rd, df_aggRes.score_rd10, s=40, alpha=0.8, linewidth=0, color='steelblue')
ax.plot([0, 0.9], [0, 0.9], ls="--", c=".3")
ax.set(xlabel='Score (model with all selected features)', ylabel='Score (model based on top 10 features)')
plt.tight_layout()
plt.savefig("%s/fig1supp_compr_score_scatter.pdf" % dir_out)
plt.close()

# smooth scatter plots on scores/recalls/correlations generated in figures.R

#------ Figure 2 -----------
# read in data
dir_in_res = './out/20.0216 feat/reg_rf_boruta'
dir_in_anlyz = os.path.join(dir_in_res, 'anlyz_filtered')
df_featSummary = pd.read_csv(os.path.join(dir_in_anlyz, 'feat_summary.csv')) #feature summary
df_featSummary['feat_sources'] = df_featSummary['feat_sources'].apply(literal_eval)
df_featSummary['feat_genes'] = df_featSummary['feat_genes'].apply(literal_eval)
df_varExp = pd.read_csv(os.path.join(dir_in_anlyz, 'feat_summary_varExp_filtered.csv'), header=0) #feature summary, more score metrics calculated
topN = getFeatN(df_featSummary)
df_src_allfeats = pd.read_csv('%s/anlyz/featSrcCounts/source_counts_allfeatures.csv' % dir_in_res, header=None, index_col=0, squeeze=True)

# pie chart of feature sources
df_counts = pd.Series([y for x in df_featSummary.feat_sources for y in x]).value_counts()
plotCountsPie(df_counts,
              'Feature source (top %d features)' % topN,
              'fig2_feat_source',
              dir_out,
              colors=[src_colors[s] for s in df_counts.index])

# heatmap of feaure sources
df = df_featSummary.loc[:,df_featSummary.columns.str.contains(r'feat_source\d')]
df.replace({'CERES':0, 'RNA-seq':1, 'CN':2, 'Mut':3, np.nan:-1}, inplace=True)
heatmapColors = [src_colors[n] for n in ['nan', 'CERES', 'RNA-seq', 'CN', 'Mut']]
cmap = LinearSegmentedColormap.from_list('Custom', heatmapColors, len(heatmapColors))
plt.figure()
ax = sns.heatmap(df, cmap=cmap, yticklabels=False, xticklabels=list(range(1,11)))
ax.set(xlabel='$\it{n}$th Feature', ylabel='Target genes')
cbar = ax.collections[0].colorbar
cbar.set_ticks([-1,0,1,2,3])
cbar.set_ticklabels(['NaN','CERES','RNA-seq','CN','Mut'])
plt.tight_layout()
plt.savefig("%s/fig2_heatmap.pdf" % dir_out)
plt.close()

def gen_feat_pies(sameGrp_counts, sameGrp_src_counts, feat_summary_annot, dir_out, fnames, labels):
    # pie chart of counts in/not in group
    c = sameGrp_counts.loc[sameGrp_counts.importanceRank == 'top10', 'count'][0]
    df_counts = pd.Series({labels[0]: c,
                           labels[1]: feat_summary_annot.shape[0] - c})
    plotCountsPie(df_counts,
                  None,
                  fnames[0],
                  dir_out)

    # pie chart of feature source
    c = sameGrp_src_counts.loc[sameGrp_src_counts.importanceRank == 'top10',]
    df_counts = pd.Series(c['count'].values, index=c['source'])
    plotCountsPie(df_counts,
                  None,
                  fnames[1],
                  dir_out,
                  colors=[src_colors[s] for s in df_counts.index])

    #heatmap
    s1 = feat_summary_annot.columns.str.startswith('inSame')
    s2 = ~feat_summary_annot.columns.str.contains('top')
    df = feat_summary_annot.loc[:, s1 & s2]

    plt.figure()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax = sns.heatmap(df, yticklabels=False, xticklabels=list(range(1, 11)), vmin=-1, vmax=1, cmap='RdBu', cbar=False)
    ax.set(xlabel='$\it{n}$th Feature', ylabel='Target genes')
    plt.tight_layout()
    plt.savefig("%s/%s_heatmap.png" % (dir_out, fnames[1]))
    plt.close()

# on same gene
feat_summary_annot_gene = pd.read_csv(os.path.join(dir_in_anlyz, 'onsamegene', 'feat_summary_annot.csv'), header=0, index_col=0)
sameGrp_counts, sameGrp_src_counts = getGrpCounts_fromFeatSummaryAnnot(feat_summary_annot_gene)
gen_feat_pies(sameGrp_counts,sameGrp_src_counts,feat_summary_annot_gene,
              dir_out, ['fig2-samegene',  'fig2-samegene_source'], ['On same gene','Not on same gene'])

# in same gene set KEGG
gs_name = 'KEGG'
feat_summary_annot_kegg = pd.read_csv(os.path.join(dir_in_anlyz, 'insamegeneset%s'%gs_name, 'feat_summary_annot.csv'), header=0, index_col=0)
sameGrp_counts, sameGrp_src_counts= getGrpCounts_fromFeatSummaryAnnot(feat_summary_annot_kegg)
gen_feat_pies(sameGrp_counts,sameGrp_src_counts,feat_summary_annot_kegg,
              dir_out, ['fig2-same%s'%gs_name,  'fig2-same%s_source'%gs_name], ['In %s'%gs_name,'Not in %s'%gs_name])

# in same gene set Panther
gs_name = 'Panther'
feat_summary_annot_panther = pd.read_csv(os.path.join(dir_in_anlyz, 'insamegeneset%s'%gs_name, 'feat_summary_annot.csv'), header=0, index_col=0)
sameGrp_counts, sameGrp_src_counts= getGrpCounts_fromFeatSummaryAnnot(feat_summary_annot_panther)
gen_feat_pies(sameGrp_counts,sameGrp_src_counts,feat_summary_annot_panther,
              dir_out, ['fig2-same%s'%gs_name,  'fig2-same%s_source'%gs_name], ['In %s'%gs_name,'Not in %s'%gs_name])

# lethality counts
df_src1 = df_featSummary[['target','feat_source1']].set_index('target')
df = pd.DataFrame({'isNotCERES': df_src1.feat_source1.isin(['RNA-seq', 'CN', 'Mut']),
                   'sameGene': feat_summary_annot_gene.inSame_1,
                   'sameGS': feat_summary_annot_kegg.inSame_1 | feat_summary_annot_panther.inSame_1,
                   'isCERES': df_src1.feat_source1 == 'CERES'
                   })

lethal_dict = {'sameGene': 'Ortholog',
               'sameGS': 'Gene set',
               'isCERES': 'Functional',
               'isNotCERES': 'Classic synthetic'}

df_counts = pd.DataFrame({'sum':df.sum(axis=0)})
df_counts['lethality'] = [lethal_dict[n] for n in df_counts.index]

plt.figure(figsize=(10,6))
ax = sns.barplot(df_counts['lethality'], df_counts['sum'], color='steelblue')
ax.set(xlabel='Lethality types', ylabel='Number of genes')
plt.tight_layout()
plt.savefig("%s/fig2_lethality_counts.pdf" % dir_out)
plt.close()

# Supp
# violin plot of scores, breakdown by source
plt.figure()
ax = sns.violinplot('feat_source', 'score_ind', data=df_varExp.loc[df_varExp.score_ind>0,:], alpha=0.1, jitter=True,
                    order=['CERES', 'RNA-seq', 'CN', 'Mut'],
                    palette=src_colors)
ax.set(xlabel='Feature source', ylabel='Score (univariate)')
plt.tight_layout()
plt.savefig("%s/fig2supp_score_by_source.pdf" % dir_out)
plt.close()

# source for all features
plotCountsPie(df_src_allfeats,
              'Data source summary (all features)',
              'fig2supp-source_allfeat',
              dir_out,
              autopct='%0.2f%%',
              colors=[src_colors[s] for s in df_src_allfeats.index])

# break down of source, by nth feature
for n in range(1, topN + 1):
    plt.interactive(False)
    plotImpSource(n, df_featSummary, os.path.join(dir_out, 'pie_imprank'), src_colors_dict=src_colors)

#------ Figure 3 -----------
# network
dir_in_network = './out/20.0216 feat/reg_rf_boruta/network/'
min_gs_size = 4
modularity = pickle.load(open(os.path.join(dir_in_network, 'modularity.pkl'), 'rb'))
G = nx.read_adjlist(open(os.path.join(dir_in_network, 'varExp_filtered.adjlist'), 'rb'))
pos = nx.spring_layout(G, seed=25)  #compute layout
cmap = plt.cm.get_cmap('RdYlBu', len(modularity))

plt.figure(figsize=(9, 9))
plt.axis('off')
for k,v in modularity.items():
    if(len(v)>min_gs_size):
        val = np.random.randint(0,len(modularity)-1)
        nx.draw_networkx_nodes(G, pos, node_size=5, nodelist=v, node_color=cmap(val))
        nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='k', style='solid', width=1, edgelist=G.edges(v))
plt.savefig('%s/fig3_network.pdf' % dir_out)

# network communities
# generated and exported from cytoscape
# network stats, excel created manually, based on stats from cytoscape

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

x = np.linspace(0.5,50,100)
plt.figure()
ax = sns.scatterplot(df.degree, df.n, color='steelblue', s=150, alpha=0.9, linewidth=1)
plt.plot(x, pl(x, *popt), '--', color='r')
textstr = r'$y = %0.2f x^{%0.2f}$' % (popt[0], popt[1])
ax.text(0.72, 0.9, textstr, transform=ax.transAxes, fontsize=10)
ax = plt.gca()
ax.set(xlim=[1,50], ylim=[0.5,1000], yscale='log', xscale='log',
      xlabel='Degree', ylabel='Number of nodes')
plt.tight_layout()
plt.savefig("%s/fig3_powerlaw.pdf" % dir_out)
plt.close()

#------ Figure 4 -----------
dir_in_Lx = './out/20.0518 Lx/L100only_reg_rf_boruta_all/'

y_compr_tr = pickle.load(open(os.path.join(dir_in_Lx, 'anlyz', 'y_compr_tr.pkl'), 'rb'))
y_compr_te = pickle.load(open(os.path.join(dir_in_Lx, 'anlyz', 'y_compr_te.pkl'), 'rb'))
df_conc_tr = pd.read_csv(os.path.join(dir_in_Lx, 'anlyz', 'concordance', 'concordance_tr.csv'))
df_conc_te = pd.read_csv(os.path.join(dir_in_Lx, 'anlyz', 'concordance', 'concordance_te.csv'))

# heatmap
plt.figure()
ax = sns.heatmap(y_compr_tr['actual'], yticklabels=False, xticklabels=False, vmin=-4, vmax=4, cmap='RdBu')
ax.set(xlabel='Genes', ylabel='Cell lines')
plt.tight_layout()
plt.savefig("%s/fig4_heatmap_yactual_tr.png" % dir_out, dpi=300)
plt.close()

plt.figure()
ax = sns.heatmap(y_compr_tr['predicted'], yticklabels=False, xticklabels=False, vmin=-4, vmax=4, cmap='RdBu')
ax.set(xlabel='Genes', ylabel='Cell lines')
plt.tight_layout()
plt.savefig("%s/fig4_heatmap_ypred_tr.png" % dir_out, dpi=300)
plt.close()

plt.figure()
ax = sns.heatmap(y_compr_te['actual'], yticklabels=False, xticklabels=False, vmin=-4, vmax=4, cmap='RdBu')
ax.set(xlabel='Genes', ylabel='Cell lines')
plt.tight_layout()
plt.savefig("%s/fig4_heatmap_yactual_te.png" % dir_out, dpi=300)
plt.close()

plt.figure()
ax = sns.heatmap(y_compr_te['predicted'], yticklabels=False, xticklabels=False, vmin=-4, vmax=4, cmap='RdBu')
ax.set(xlabel='Genes', ylabel='Cell lines')
plt.tight_layout()
plt.savefig("%s/fig4_heatmap_ypred_te.png" % dir_out, dpi=300)
plt.close()

# scatter
plt.figure()
plt.plot([-3,2], [-3,2], ls="--", c=".3", alpha=0.5)
ax = sns.scatterplot(y_compr_tr['actual'].values.flatten(), y_compr_tr['predicted'].values.flatten(),
                     s = 1, alpha=0.05, linewidth=0, color='steelblue')
ax.set(xlabel='Actual', ylabel='Predicted', xlim=[-3,2], ylim=[-3,2])
plt.tight_layout()
plt.savefig("%s/fig4_pred_actual_tr.png" % dir_out, dpi=300)
plt.close()

plt.figure()
plt.plot([-3,2], [-3,2], ls="--", c=".3", alpha=0.5)
ax = sns.scatterplot(y_compr_te['actual'].values.flatten(), y_compr_te['predicted'].values.flatten(),
                     s = 1, alpha=0.05, linewidth=0, color='steelblue')
ax.set(xlabel='Actual', ylabel='Predicted', xlim=[-3,2], ylim=[-3,2])
plt.tight_layout()
plt.savefig("%s/fig4_pred_actual_te.png" % dir_out, dpi=300)
plt.close()

# concordance
df1 = df_conc_tr['concordance'].to_frame().copy()
df1['dataset'] = 'train'
df2 = df_conc_te['concordance'].to_frame().copy()
df2['dataset'] = 'test'
df = pd.concat([df1,df2])
df['cat'] = 'one'

plt.figure()
ax = sns.violinplot(y='cat', x='concordance', hue='dataset', data=df, split=True, linewidth=1.6)
ax.set(xlim=[0.34,1.05], xlabel='Concordance', ylabel='', yticks=[])
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig("%s/fig4supp_concordance.pdf" % dir_out)
plt.close()

# examples
genename = 'MDM2'
res = pickle.load(open('%s/model_perf/y_compr_%s.pkl' % (dir_in_Lx, genename),'rb'))
df = res['te']
plt.figure()
plt.plot([-2.5,1], [-2.5,1], ls="--", c=".3", alpha=0.5)
ax = sns.scatterplot(df.y_actual, df.y_pred, s=70, alpha=0.7, linewidth=0, color='steelblue')
ax.set(xlabel='actual', ylabel='predicted', xlim=[-2.5,1], ylim=[-2.5,1], title=genename)
plt.tight_layout()
plt.savefig("%s/fig4supp_ycompr_MDM2.pdf" % dir_out)
plt.close()

genename = 'XRCC6'
res = pickle.load(open('%s/model_perf/y_compr_%s.pkl' % (dir_in_Lx, genename),'rb'))
df = res['te']
plt.figure()
plt.plot([-2.5,1], [-2.5,1], ls="--", c=".3", alpha=0.5)
ax = sns.scatterplot(df.y_actual, df.y_pred, s=70, alpha=0.7, linewidth=0, color='steelblue')
ax.set(xlabel='actual', ylabel='predicted', xlim=[-2.5,1], ylim=[-2.5,1], title=genename)
plt.tight_layout()
plt.savefig("%s/fig4supp_ycompr_XRCC6.pdf" % dir_out)
plt.close()


# randomized model
np.random.seed(seed=25)
def getDummyInfer(y):
    return np.random.uniform(-4.4923381539,3.9745784786800002, size=y.shape[0]) #-4.49.. and 3.97.. are the min and max of CERES scores in the tr dataset
y_pred = y_compr_tr['actual'].apply(getDummyInfer, axis=0)

# heatmap
plt.figure()
ax = sns.heatmap(y_pred, yticklabels=False, xticklabels=False, vmin=-3, vmax=3, cmap='RdBu')
ax.set(xlabel='Genes', ylabel='Cell lines')
plt.tight_layout()
plt.savefig("%s/fig4_heatmap_ypred_tr_random.png" % dir_out, dpi=300)
plt.close()

# scatter
plt.figure()
plt.plot([-3,2], [-3,2], ls="--", c=".3", alpha=0.5)
ax = sns.scatterplot(y_compr_tr['actual'].values.flatten(), y_pred.values.flatten(),
                     s = 1, alpha=0.05, linewidth=0, color='steelblue')
ax.set(xlabel='Actual', ylabel='Predicted', xlim=[-3,2], ylim=[-3,2])
plt.tight_layout()
plt.savefig("%s/fig4_pred_actual_tr_random.png" % dir_out, dpi=300)
plt.close()
