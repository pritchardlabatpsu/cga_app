import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from tqdm import tqdm
from src.lib.data import stats_Crispr

tqdm.pandas()

dir_in = './out/20.0216 feat/reg_rf_boruta/' # location of the dm_data
outdir = './out/20.0216 feat/reg_rf_boruta/dm_data_baseline/' # output directory
if(np.logical_not(os.path.exists(outdir))): os.mkdir(outdir)
plt.interactive(False)

#----------------------
#load data
dm_data = pickle.load(open('%s/dm_data.pkl' % dir_in,'rb'))
df_crispr_stats = stats_Crispr(dm_data)

#----------------------
#plot stats
plt.figure()
ax = sns.distplot(df_crispr_stats['avg'])
ax.set(xlabel='CERES [mean]', ylabel='Freq')
plt.savefig("%s/dist_ceres_mean.pdf" % (outdir))
plt.close()

plt.figure()
ax = sns.distplot(df_crispr_stats['std'])
ax.set(xlabel='CERES [SD]', ylabel='Freq')
plt.savefig("%s/dist_ceres_sd.pdf" % (outdir))
plt.close()

plt.figure()
ax = sns.scatterplot(x='diff',y='std', data=df_crispr_stats,s=90)
ax.set(xlabel='CERES range', ylabel='CERES sd')
plt.savefig("%s/scatter_range.sd.pdf" % outdir)
plt.close()

plt.figure()
ax = sns.scatterplot(x='avg',y='std', data=df_crispr_stats,s=90)
ax.set(xlabel='CERES mean', ylabel='CERES sd')
plt.savefig("%s/scatter_mean_sd.pdf" % outdir)
plt.close()

plt.figure()
ax = sns.scatterplot(x='avg',y='diff', data=df_crispr_stats,s=90)
ax.set(xlabel='CERES mean', ylabel='CERES range')
plt.savefig("%s/scatter_mean_range.pdf" % outdir)
plt.close()

#----------------------
# get gene dependency classifications (selective essential, common essentials, common non-essential)
df_genedep = pd.read_csv('%s/%s' % (dm_data.dir_datasets, dm_data.fname_gene_dependency), header=0, index_col=0)
df_genedep.columns = df_genedep.columns.str.extract('^(.*)\s').squeeze().values

def classifyDep(x):
    if all(x > 0.5):
        return 'common_essential'
    elif all(x < 0.5):
        return 'common_nonessential'
    else:
        return 'selective_essential'

dep_class = df_genedep.apply(lambda x: classifyDep(x), axis=0)
dep_class.to_csv("%s/gene_essential_classification.csv" % outdir, header=False, index=True)

