#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Building and running models, with CERES landmark genes
"""

import os
import pandas as pd
import pickle
import logging

from sklearn.model_selection import train_test_split
from tqdm import tqdm
tqdm.pandas()

from src.lib.data import depmap_data
from src.lib.data import scale_data, build_data_gene, qc_feats, stats_Crispr
from src.lib.models import depmap_model, selectUnivariate, model_infer_iter,model_univariate,model_infer_iter_ens,model_infer
from src.lib.models import _featSelect_base as sf_base

from src.lib.utils import getFeatGene, getFeatSource

######################################################################
# Parameters
######################################################################
outdir = './out/20.0518 Lx/'
opt_scale_data = False #scale input data True/False
opt_scale_data_types = '\[(?:RNA-seq|CN)\]' #data source types to scale; in regexp
outdir_run = 'L100only_reg_rf_boruta_all' #the output folder for the run
subdir_modtmp = '%s/%s/model_perf/' % (outdir,outdir_run) #intermediate files for each model
model_data_source = ['CERES_Lx'] #,'RNA-seq','CN','Mut','Lineage']
anlyz_set_topN = 10 #for analysis set how many of the top features to look at
perm_null = 1000 #number of samples to get build the null distribution, for corr
session_notes = 'L100 landmarks only, regression model (rf_boruta), to predict on all genes'
dir_Lx = './out/19.1013 tight cluster/landmarks_n100_k100.csv' #directory for the landmarks Lx

# useGene_dependency = True  #whether to use CERES gene dependency (true) or gene effect (false)
# model_name = 'rfc'
# model_params = {'n_estimators':1000,'max_depth':15,'min_samples_leaf':5,'max_features':'log2'}
# model_paramsgrid = {}

useGene_dependency = False #whether to use CERES gene dependency (true) or gene effect (false)
model_name = 'rf'
model_params = {'n_estimators':1000,'max_depth':15,'min_samples_leaf':5,'max_features':'log2'}
model_paramsgrid = {}

# useGene_dependency = False #whether to use CERES gene dependency (true) or gene effect (false)
# model_name = 'lm'
# model_params = {}
# model_paramsgrid = {}

# useGene_dependency = False #whether to use CERES gene dependency (true) or gene effect (false)
# model_name = 'elasticNet'
# model_params = {'alpha':0.03, 'l1_ratio':0.5}
# model_paramsgrid = {}

# useGene_dependency = False #whether to use CERES gene dependency (true) or gene effect (false)
# model_name = 'mlp'
# model_params = {'num_layers':2, 'feat_size':100}
# model_paramsgrid = {}

model_pipeline = model_infer_iter_ens #model_infer_iter_ens, model_infer_iter, model_univariate, model_infer

######################################################################
# Set settings
######################################################################
# set up dir
if(not os.path.exists(outdir)): os.mkdir(outdir)
outdir_sub = '%s/%s/' % (outdir, outdir_run)
if(not os.path.exists(outdir_sub)): os.mkdir(outdir_sub)
if(subdir_modtmp is not None):
    if(not os.path.exists(subdir_modtmp)): os.mkdir(subdir_modtmp)

# save run settings
f = open("%s/run_settings.txt" % (outdir_sub), "w")
f.write('Model name: %s\n' % model_name)
f.write('Use gene dependency: %s\n' % useGene_dependency)
for k,v in model_params.items():
    f.write('Model parameter %s: %s\n' % (k,v))
for k,v in model_paramsgrid.items():
    f.write('Model parameter grid search, %s: %s\n' % (k,v))
f.write('Model data source: %s\n' % model_data_source)
f.write('Model pipeline: %s\n' % model_pipeline)
f.write('Scale data: %s\n' % opt_scale_data)
f.write('Scale data types: %s\n' % opt_scale_data_types)
f.write('Number of features in analysis set: %d\n' % anlyz_set_topN)
f.write('Number of samples to draw for null distribution, per target gene: %d\n' % perm_null)
f.write('Run session notes: %s\n' % session_notes)
f.close()

######################################################################
# Load in datasets
######################################################################
#------------------
# parse P19Q3 data
dm_data = depmap_data()
dm_data.dir_datasets = '../datasets/DepMap/19Q3/'
dm_data.data_name = 'data_19Q3'
dm_data.load_data(useGene_dependency)
dm_data.preprocess_data() #handles formatting and missing data
dm_data.filter_samples() #only keep the shared_idx samples
dm_data.filter_baseline() #baseline filter, to remove invariant and low variant features
dm_data.df_landmark = pd.read_csv(dir_Lx, header=0)

#------------------
# parse P19Q3 data (repeat, for use to subtract out from Q4 below)
dm_data_Q3 = depmap_data()
dm_data_Q3.dir_datasets = '../datasets/DepMap/19Q3/'
dm_data_Q3.data_name = 'data_19Q3'
dm_data_Q3.load_data(useGene_dependency)
dm_data_Q3.preprocess_data() #handles formatting and missing data

# parse P19Q4 data
dm_data_Q4 = depmap_data()
dm_data_Q4.dir_datasets = '../datasets/DepMap/19Q4/'
dm_data_Q4.data_name = 'data_19Q4'
dm_data_Q4.load_data(useGene_dependency)
dm_data_Q4.preprocess_data() #handles formatting and missing data
dm_data_Q4.df_landmark = pd.read_csv(dir_Lx, header=0)

# only keep the Q4 new cell lines
samples_q3 = dm_data_Q3.df_crispr.index
samples_q4 = dm_data_Q4.df_crispr.index
new_samples_q4 = set(samples_q4) - set(samples_q3)
dm_data_Q4.filter_samples(list(new_samples_q4)) #keep just the shared idx and only Q4

# match features to that in Q3 (used for training)
dm_data_Q4.match_feats(dm_data)

#------------------
# print dataset stats
dm_data.printDataStats(outdir_sub)
dm_data_Q4.printDataStats(outdir_sub)

#------------------
# select which target genes to analyze
# pick genes to analyze
# - specific genes
#genes2analyz = ['SOX10','KRAS','CDK4','EMC4','MAX','PTPN11']

#- all genes 
genes2analyz = dm_data.df_crispr.columns.str.replace('\s.*','')

#- selective dependent genes
# if(useGene_dependency):
#     df_crispr_stats = stats_Crispr(dm_data)
#     filtered = df_crispr_stats.loc[~df_crispr_stats.entropy.isnull(),:]
#     filtered = filtered.loc[(filtered.dependent>10) & (filtered.not_dependent>10),:]
#     filtered = filtered.loc[filtered.entropy>0.5,:]
#     genes2analyz = filtered.index.map(lambda x: getFeatGene(x, firstOnly=True))
# else:
#     df_crispr_stats = stats_Crispr(dm_data)
#     gene_sel1 = set(df_crispr_stats[df_crispr_stats['std']>0.25].index)
#     gene_sel2 = df_crispr_stats[df_crispr_stats['diff']>0.6].index
#     gene_sel = gene_sel1.intersection(gene_sel2)
#     genes2analyz = pd.Series(list(gene_sel)).apply(getFeatGene, firstOnly=True)

# make sure target genes are not landmarks themselves
genes2analyz = list(set(genes2analyz) - set(dm_data.df_landmark.landmark.str.replace('\s.*','').values))

######################################################################
# Run model pipeline, per target gene
######################################################################
model_results = pd.DataFrame()

for gene2anlyz in tqdm(genes2analyz):
    df_res = pd.DataFrame()

    #--- create datasets ---
    data_name, df_x, df_y, df_y_null = build_data_gene(model_data_source, dm_data, gene2anlyz)
    data_name, df_x_q4, df_y_q4, df_yn_q4 = build_data_gene(model_data_source, dm_data_Q4, gene2anlyz)

    # data checks
    if(len(df_x)<1 or len(df_y)<1): #empty x or y data
        continue

    if(not qc_feats([df_x, df_x_q4])):
        raise ValueError('Feature name/order across the datasets do not match')

    # set up the data matrices and feature labels
    feat_labels = pd.DataFrame({'name':df_x.columns.values,
                                'gene':pd.Series(df_x.columns).apply(getFeatGene, firstOnly=True),
                                'source':pd.Series(df_x.columns).apply(getFeatSource, firstOnly=True)})
    feat_labels.index.name = 'feat_id'
    x_vals = df_x.values
    y_vals = df_y.values.ravel()
    yn_vals = df_y_null.values
    x_q4 = df_x_q4.values
    y_q4 = df_y_q4.values.ravel()
    yn_q4_vals = df_yn_q4.values
    
    # split to train/test
    if(useGene_dependency): #if doing classification, make sure the split datasets are balanced
        x_train, x_test, y_train, y_test, yn_train, yn_test = train_test_split(x_vals, y_vals, yn_vals, test_size=0.15, random_state=42, stratify=y_vals)
    else:
        x_train, x_test, y_train, y_test, yn_train, yn_test  = train_test_split(x_vals, y_vals, yn_vals, test_size=0.15, random_state=42)
    
    # scale data if needed
    if(opt_scale_data):
        to_scale_idx = df_x.columns.str.contains(opt_scale_data_types)
        if(any(to_scale_idx)):
            x_train, x_test, x_q4 = scale_data(x_train, [x_test, x_q4], to_scale_idx)
        else:
            logging.info("Trying to scale data, but the given data type is not found and cannot be scaled for gene %s" % gene2anlyz)
    
    # set up model
    dm_model = depmap_model(model_name, model_params, model_paramsgrid, outdir=outdir_sub)
    
    
    #--- model pipeline ---
    data = {'train': {'x':x_train, 'y':y_train}, 'test': {'x':x_test, 'y':y_test}}
    data_null = {'test': {'x': x_test, 'y': y_test, 'y_null':yn_test}}
    
    df_res = pd.DataFrame()
    df_res, sf = model_pipeline(data, dm_model, feat_labels, gene2anlyz, df_res, useGene_dependency, data_null, perm_null)
    
    if(sf is None):
        #feature selection in the end is empty
        model_results = model_results.append(df_res, ignore_index=True)
        continue
        
    feats = sf.importance_sel

    #--- analysis set ---
    # pick a list of top N features
    feat_sel = sf.importance_sel.iloc[0:anlyz_set_topN,:]
    x_tr, x_te, x_q4_rd = sf_base().transform_set(x_train, x_test, x_q4, feat_idx=feat_sel.index)
    
    # reduced model on the top N features
    data = {'train': {'x':x_tr, 'y':y_train}, 
            'test': {'x':x_te, 'y':y_test},
            'p19q4': {'x':x_q4_rd, 'y':y_q4}}
    data_null = {'test': {'x': x_te, 'y': y_test, 'y_null':yn_test},
                 'p19q4': {'x': x_q4_rd, 'y': y_q4, 'y_null':yn_q4_vals}}
    dm_model.fit(x_tr, y_train, x_te, y_test)
    df_res_sp = dm_model.evaluate(data, 'top10feat', 'top10feat', gene2anlyz, data_null, perm_null)
    df_res = df_res.append(df_res_sp, sort=False)
    if (subdir_modtmp is not None):
        pickle.dump(dm_model.model, open('%s/model_rd10_%s.pkl' % (subdir_modtmp, gene2anlyz), "wb"))  # pickle top10feat model

    # save the y actual and predicted, using the reduced model
    y_compr = {'tr': pd.DataFrame({'y_actual': y_train,
                                   'y_pred': dm_model.predict(x_tr)}),
               'te': pd.DataFrame({'y_actual': y_test,
                                   'y_pred': dm_model.predict(x_te)})}
    if (subdir_modtmp is not None):
        pickle.dump(y_compr, open('%s/y_compr_%s.pkl' % (subdir_modtmp, gene2anlyz), "wb"))  # pickle y_actual vs y_pred

    # univariate on the top N features
    sf = selectUnivariate(dm_model)
    sf.fit(x_tr, y_train, x_te, y_test, feat_sel.feature, target_name=gene2anlyz)
    df_res = df_res.append(sf.importance.reset_index(), sort=False)
    
    #--- saving results ---
    model_results = model_results.append(df_res, ignore_index=True, sort=False)
    if(subdir_modtmp is not None):
        feats.to_csv('%s/feats_%s.csv' % (subdir_modtmp, gene2anlyz), index=True)


#-------
# save results
model_results.reset_index(inplace=True, drop=True)
model_results.to_csv('%s/model_results.csv' % (outdir_sub), index=False)

