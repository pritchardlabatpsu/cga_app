#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data
@author: boyangzhao
"""

import numpy as np
import pandas as pd
import os
import pickle

from sklearn import preprocessing
import warnings

class depmap_data:
    def __init__(self):
        self.dir_datasets = '../datasets/DepMap/19Q3/'
        self.data_name = 'data'
        self.fname_rnaseq = 'CCLE_expression_full.csv'
        self.fname_cn = 'CCLE_gene_cn.csv'
        self.fname_mut = 'CCLE_mutations.csv'
        self.fname_sample_info = 'sample_info.csv'
        self.fname_gene_effect = 'Achilles_gene_effect.csv'
        self.fname_gene_dependency = 'Achilles_gene_dependency.csv'
        
        self.baseline_filter_stats = None
        
    def load_data(self, gene_dependency=False):
        print('loading rna-seq...')
        self.df_rnaseq = pd.read_csv('%s/%s' % (self.dir_datasets,self.fname_rnaseq), header=0)
        
        print('loading copy number...')
        self.df_cn = pd.read_csv('%s/%s' % (self.dir_datasets,self.fname_cn), header=0)
        
        print('loading mutations...')
        self.df_mut = pd.read_csv('%s/%s' % (self.dir_datasets,self.fname_mut), header=0, index_col='DepMap_ID')
        
        print('loading sample_info...')
        self.df_info = pd.read_csv('%s/%s' % (self.dir_datasets,self.fname_sample_info), header=0)
        
        self.gene_dependency = gene_dependency
        if gene_dependency:
            print('loading Achilles gene dependency...')
            self.df_crispr = pd.read_csv('%s/%s' % (self.dir_datasets,self.fname_gene_dependency), header=0)
        else:
            print('loading Achilles gene effect...')
            self.df_crispr = pd.read_csv('%s/%s' % (self.dir_datasets,self.fname_gene_effect), header=0)


    def preprocess_data(self):
        #----------------------
        #preprocess data
        def reindex(df):
            if(df.index.name == 'DepMap_ID'):
              return df
            #make the first column as the index
            df = df.rename(columns={df.columns[0]: 'DepMap_ID'})
            df.set_index('DepMap_ID', inplace=True)
            return df
        
        
        #crispr
        print('processing CERES...')
        self.df_crispr = reindex(self.df_crispr)
        self.df_crispr.columns = self.df_crispr.columns + ' [CERES]'
        linesN_missing = max(self.df_crispr.isna().sum()) #maximum number of cell lines with missing data
        if linesN_missing>0:
            print('There are %d cell lines out of %d that will be dropped, due to missing data' % 
                  (linesN_missing, self.df_crispr.shape[0]))
            self.df_crispr.dropna(axis=0, how='any',inplace=True) #remove cell lines with any NaNs

        if self.gene_dependency:
            #using Achilles gene dependency, convert proba to dependent/not dependent
            self.df_crispr = self.df_crispr > 0.5
            self.df_crispr = self.df_crispr.astype(int)
            
        #rna-seq
        print('processing rna-seq...')
        self.df_rnaseq = reindex(self.df_rnaseq)
        self.df_rnaseq.columns = self.df_rnaseq.columns + ' [RNA-seq]'
        
        #copy number
        print('processing copy number...')
        self.df_cn = reindex(self.df_cn)
        self.df_cn.columns = self.df_cn.columns + ' [CN]'
        naCount = sum(self.df_cn.isna().sum())
        if naCount>0:
            print('There are %d NAs in the copy number dataset and will be replaced by zeros' % naCount)
            self.df_cn.fillna(0, inplace=True)
        
        #mutation data, broken into damaging, hotspot (damaging), hotspot (non-damaging)
        print('processing mutations...')
        def getMutMatrix(geneslist, mut_subset, suffix):
            #mapping to a binary matrix, with genes as given
            #mut_subset is a subset of mutations (based on criteria filtered elsewhere)
            def getMut(x):
                val = geneslist.isin(x.geneid)*1
                return val
                
            df_mut_subset = mut_subset.groupby('DepMap_ID').apply(getMut)
            df_mut_subset.columns = geneslist
            df_mut_subset.columns = df_mut_subset.columns + ' ' + suffix + ' [Mut]'
            
            return df_mut_subset
            
        mut_genes = self.df_mut['Hugo_Symbol'].astype(str) + ' (' + self.df_mut['Entrez_Gene_Id'].astype(str) + ')'
        self.df_mut['geneid'] = mut_genes
        mut_geneslist = mut_genes.unique()
        mut_geneslist.sort()
        mut_geneslist = pd.Series(mut_geneslist)
        
        #damaging genes
        mut_damaging = self.df_mut.loc[self.df_mut.Variant_annotation == 'damaging',:].copy()
        df_mut_damaging = getMutMatrix(mut_geneslist, mut_damaging, '(damaging)')
        
        #hotspot (nondamaging)
        mut_hotspot = self.df_mut.loc[(self.df_mut.Variant_annotation != 'damaging') & 
                                 (self.df_mut.isCOSMIChotspot | self.df_mut.isTCGAhotspot),:].copy()
        df_mut_hotspot = getMutMatrix(mut_geneslist, mut_hotspot, '(hotspot)')
        
        #hotspot (nondamaging)
        mut_hotspot = self.df_mut.loc[(self.df_mut.Variant_annotation == 'other non-conserving') |
                                 (self.df_mut.Variant_annotation == 'other conserving'),:].copy()
        df_mut_other = getMutMatrix(mut_geneslist, mut_hotspot, '(other)')
        
        #combine and overwrite the old df_mut
        self.df_mut = df_mut_damaging.merge(df_mut_hotspot, left_index=True, right_index=True)
        self.df_mut = self.df_mut.merge(df_mut_other, left_index=True, right_index=True)
        
        #sample info - tissue origin
        print('processing sample_info...')
        self.df_info = reindex(self.df_info)
        self.df_info.columns = self.df_info.columns.str.replace(' ','_') #some older versions had space in the column names, add _ so it's consistent with later version
        self.df_info['lineage'] = self.df_info['CCLE_Name'].str.replace('\[.*\]','')
        self.df_info['lineage'] = self.df_info.lineage.str.extract('_(.*)')
        self.df_info.loc[self.df_info.index=='ACH-001142','lineage'] = 'CENTRAL_NERVOUS_SYSTEM' #data says PRIMARY, but this is a CNS lineage
        #some samples have CJ1-3_resistant label for skin, just say skin instead
        self.df_info.loc[self.df_info.lineage=='SKIN_CJ3_RESISTANT',:] = 'SKIN'
        self.df_info.loc[self.df_info.lineage=='SKIN_CJ2_RESISTANT',:] = 'SKIN'
        self.df_info.loc[self.df_info.lineage=='SKIN_CJ1_RESISTANT',:] = 'SKIN'
        
        self.df_lineage = pd.get_dummies(self.df_info.lineage)
        self.df_lineage.index = self.df_info.index
        self.df_lineage.columns = self.df_lineage.columns + ' [Lineage]'
        
        #cell lines that are shared in all datasets
        print('finalizing processing...')
        self.shared_idx = self.df_crispr.index.intersection(self.df_rnaseq.index)
        self.shared_idx = self.shared_idx.intersection(self.df_cn.index)
        self.shared_idx = self.shared_idx.intersection(self.df_mut.index)
        self.shared_idx = self.shared_idx.intersection(self.df_lineage.index)


    def filter_samples(self, samples_idx=None):
        #filter dataset to only keep the samples with shared_idx
        #samples_idx is given, this will be used, but still with intersection with
        #self.shared_idx
        print('filtering: keep only shared samples...')
        
        if samples_idx is None:
            samples_idx = self.shared_idx
        else:
            samples_idx = list(set(self.shared_idx) & set(samples_idx))
            self.shared_idx = samples_idx #updated shared idx
            
        self.df_mut = self.df_mut.loc[samples_idx,:]
        self.df_cn = self.df_cn.loc[samples_idx,:]
        self.df_rnaseq = self.df_rnaseq.loc[samples_idx,:]
        self.df_crispr = self.df_crispr.loc[samples_idx,:]
        self.df_lineage = self.df_lineage.loc[samples_idx,:]
    
    
    def filter_baseline(self):
        #baseline filter on the datasets, to prune down on the features
        #returns the stats of the baseline filter
        print('filtering: baseline feature pruning...')
        
        #---
        #remove invariant features
        def removeInvariant(df):
            return df.loc[:, df.var() != 0.0], sum(df.var() == 0.0)
        
        self.df_crispr, c1 = removeInvariant(self.df_crispr)
        self.df_mut, c2 = removeInvariant(self.df_mut)
        self.df_lineage, c3 = removeInvariant(self.df_lineage)
        
        self.df_rnaseq, c4 = removeInvariant(self.df_rnaseq)
        self.df_cn, c5 = removeInvariant(self.df_cn)
        
        counts1 = pd.DataFrame({'source':['ceres','mutation','lineage','rnaseq','cn'],
                                'counts_invariant':[c1,c2,c3,c4,c5]})
        
        #---
        #for categorical data
        #remove features where only a few samples support a category
        def removeLowVariantCat(df, threshold=10): #i.e. keep with >threshold
            vals = np.unique(df.values)
            feat_toRemove = None
            for val in vals:
                count = df.apply(lambda x: sum(x==val), axis=0)
                feat_idx_bool = count <= threshold
                if(feat_toRemove is None):
                    feat_toRemove = feat_idx_bool
                else:
                    feat_toRemove = feat_toRemove | feat_idx_bool #cumulative tally of feat to remove
            
            df = df.loc[:, ~feat_toRemove] #keep the ones that are not in the list
            
            return df, sum(feat_toRemove)
    
        
        if self.gene_dependency:
            self.df_crispr, c1 = removeLowVariantCat(self.df_crispr)
        else:
            c1 = None
        self.df_mut, c2 = removeLowVariantCat(self.df_mut)
        self.df_lineage, c3 = removeLowVariantCat(self.df_lineage)
            
        counts2 = pd.DataFrame({'source':['ceres','mutation','lineage','rnaseq','cn'],
                               'counts_cat_lowvariant':[c1,c2,c3,None,None]})
    
        counts = counts1.merge(counts2, on='source')

        #---
        #for continuous variables
        #remove features where most of the values are the same (remove ones with only x values are different, per feature)
        #where x is <= threshold
        def removeLowVariantCont(df, threshold=10):
            df_lowvar = df.nunique() < threshold
            df_lowvar = df.loc[:,df_lowvar]
            lowvar_count = df_lowvar.apply(lambda x: pd.value_counts(x).max()) #occurence count of the most freq val
            feat_toRemove = lowvar_count[lowvar_count>=(df.shape[0]-threshold)] #mark the ones to remove
            feat_toRemove = df.columns.isin(feat_toRemove.index)
            
            df = df.loc[:,~feat_toRemove]
            
            return df, sum(feat_toRemove)
        
        
        if not self.gene_dependency:
            self.df_crispr, c1 = removeLowVariantCont(self.df_crispr)
        else:
            c1 = None
        
        self.df_rnaseq, c2 = removeLowVariantCont(self.df_rnaseq)
        self.df_cn, c3 = removeLowVariantCont(self.df_cn)
           
        counts3 = pd.DataFrame({'source':['ceres','mutation','lineage','rnaseq','cn'],
                               'counts_cont_lowvariant':[c1,None,None,c2,c3]})

        counts = counts.merge(counts3, on='source')
        
        #---
        #for RNA-seq
        #remove non-expressed genes
        feat_toRemove = self.df_rnaseq.max()<1
        self.df_rnaseq = self.df_rnaseq.loc[:,~feat_toRemove]
        
        counts4 = pd.DataFrame({'source':['ceres','mutation','lineage','rnaseq','cn'],
                               'counts_nonexpressed':[None,None,None,sum(feat_toRemove),None]})
    
        counts = counts.merge(counts4, on='source')
        
        #---
        #summary 
        counts5 = pd.DataFrame({'source':['ceres','mutation','lineage','rnaseq','cn'],
                               'feats_left':[self.df_crispr.shape[1],
                                            self.df_mut.shape[1],
                                            self.df_lineage.shape[1],
                                            self.df_rnaseq.shape[1],
                                            self.df_cn.shape[1] ]})
        counts = counts.merge(counts5, on='source')
        
        self.baseline_filter_stats = counts
        
        
    def match_feats(self, dm_data):
        #match the dataset features to that in the provided dm_data
        self.df_mut = self.df_mut.loc[:,dm_data.df_mut.columns]
        self.df_cn = self.df_cn.loc[:,dm_data.df_cn.columns]
        self.df_rnaseq = self.df_rnaseq.loc[:,dm_data.df_rnaseq.columns]
        self.df_crispr = self.df_crispr.loc[:,dm_data.df_crispr.columns]
        self.df_lineage = self.df_lineage.loc[:,dm_data.df_lineage.columns]
    
    
    def printDataStats(self, outdir_sub='./'):
        #print out dataset datasets
        
        #dataset stats
        def getDFinfo(df):
            #return DepMap unique ID count, 
            return (len(df.index.unique()), df.shape[1])
            
        f = open("%s/%s_info.txt" % (outdir_sub, self.data_name), "w")
        f.write('CERES: %d cell lines and %d genes\n' % getDFinfo(self.df_crispr))
        f.write('Mutations: %d cell lines and %d genes\n' % getDFinfo(self.df_mut))
        f.write('Lineage: %d cell lines and %d lineages\n' % getDFinfo(self.df_lineage))
        f.write('RNA-seq: %d cell lines and %d genes\n' % getDFinfo(self.df_rnaseq))
        f.write('Copy number: %d cell lines and %d genes \n' % getDFinfo(self.df_cn))
        f.close()
        
        #baseline filter stats
        if(self.baseline_filter_stats is not None):
            self.baseline_filter_stats.to_csv('%s/%s_base_filter.csv' % (outdir_sub,self.data_name))
        

#-------------------------------------------- 
def build_data_gene(datatype, dm_data, gene, sample_idx=None):
    #build x based on datatype, for y (CERES) of single gene
    #datatype is an array of data sources, e.g. ['CERES','RNA-seq','CN','Mut']
    #sample_idx defines the samples to choose from the datasets, this should be e.g. samples
    #   overlapping in all datasets

    df_x = []
    df_y = []
    data_name = ''
    
    #----------------------
    #reduce data to only cell lines that are shared across all data sources
    if sample_idx is not None:
        samples_idx = list(set(dm_data.shared_idx) & set(samples_idx)) #make sure the sample_idx are in ones shared across all datasets
        df_mut_shared = dm_data.df_mut.loc[sample_idx,:]
        df_cn_shared = dm_data.df_cn.loc[sample_idx,:]
        df_rnaseq_shared = dm_data.df_rnaseq.loc[sample_idx,:]
        df_crispr_shared = dm_data.df_crispr.loc[sample_idx,:]
        df_lineage_shared = dm_data.df_lineage.loc[sample_idx,:]
    else:
        df_mut_shared = dm_data.df_mut
        df_cn_shared = dm_data.df_cn
        df_rnaseq_shared = dm_data.df_rnaseq
        df_crispr_shared = dm_data.df_crispr
        df_lineage_shared = dm_data.df_lineage

    #get the gene data
    df_crispr_y = df_crispr_shared.filter(regex=('^%s\s' % gene))
    df_crispr_y_null = df_crispr_shared.filter(regex=('^(?!%s\s)' % gene))
    df_crispr_rd = df_crispr_shared.copy()
    df_crispr_rd = df_crispr_rd.drop(columns=set(df_crispr_rd.columns).intersection(df_crispr_y.columns))

    #landmark CERES if landmarks are defined
    df_crispr_Lx = None
    if hasattr(dm_data, 'df_landmark') and (dm_data.df_landmark is not None):
        col_sel = [n in dm_data.df_landmark.landmark.values for n in df_crispr_shared.columns.str.replace('\s\[.*','')]
        df_crispr_Lx = df_crispr_shared.loc[:,col_sel].copy()
    
    if df_crispr_y.shape[1] < 1:
        warnings.warn('gene %s not found in shared CERES dataset...' % gene)
        return data_name, df_x, df_y
    
    #construct the dataset
    data_name = '_'.join(datatype)
    
    df_dict = {'CERES': df_crispr_rd,
               'CERES_Lx': df_crispr_Lx,
               'RNA-seq': df_rnaseq_shared,
               'CN': df_cn_shared,
               'Mut': df_mut_shared,
               'Lineage': df_lineage_shared}
    
    df_y = df_crispr_y
    df_y_null = df_crispr_y_null
    df_x = pd.DataFrame()
    for d in datatype:
        df_x = pd.concat([df_x, df_dict[d]], axis=1)

    return data_name, df_x, df_y, df_y_null

    
def scale_data(df_ref, df_toScale, to_scale_idx=None):
    #scale data
    #labels is a boolean vector that marks which columns to scale
    
    scaler = preprocessing.StandardScaler()
    
    #fit to reference
    df_ref= df_ref.copy() #modify and return the copy, and not touch the original
    if to_scale_idx is not None:
        df_ref = pd.DataFrame(df_ref)
        df_ref.loc[:,to_scale_idx] = scaler.fit_transform(df_ref.loc[:,to_scale_idx])
        df_ref = df_ref.values
    else:
        df_ref = scaler.fit_transform(df_ref)
    
    #transform data
    df_scaled = []
    if(type(df_toScale) == np.ndarray):
        df_toScale = [df_toScale] #put it into a list if the given df is just a matrix
    for df in df_toScale:
        df = df.copy()  #modify and return the copy, and not touch the original
        if(to_scale_idx is not None):
            df = pd.DataFrame(df)
            df.loc[:,to_scale_idx] = scaler.transform(df.loc[:,to_scale_idx])
            df = df.values
        else:
            df = scaler.transform(df)
            
        df_scaled.append(df)
        
    return (df_ref, *df_scaled)
    
def qc_feats(dfs):
    #quality checks
    #make sure all the given datasets (data frames) have the same features in the same order
    if(not np.all([len(dfs[0].columns) ==len(df.columns) for df in dfs])):
        return False
    
    return np.all([dfs[0].columns[i] == df.columns[i] for df in dfs for i in range(len(df.columns))])

def stats_Crispr(dm_data):
    if dm_data.gene_dependency:  #y is categorical
        n_0 = dm_data.df_crispr.apply(lambda x: sum(x==0), axis=0)
        n_1 = dm_data.df_crispr.apply(lambda x: sum(x==1), axis=0)
        n_total = dm_data.df_crispr.shape[0]
        
        df_stats = pd.DataFrame({'not_dependent':n_0, 'dependent':n_1})
        p0 = df_stats.not_dependent/n_total
        p1 = df_stats.dependent/n_total
        df_stats['entropy'] = p0*np.log2(1/p0) + p1*np.log2(1/p1)
    else:  #y is continuous
        df_stats = pd.DataFrame({'min':dm_data.df_crispr.apply(min),
                                 'max':dm_data.df_crispr.apply(max),
                                 'avg':dm_data.df_crispr.apply(np.mean),
                                 'std':dm_data.df_crispr.apply(np.std)
                                 })
        df_stats['diff'] = df_stats['max'] - df_stats['min']
    
    return df_stats


def preprocessDataQ3Q4(useGene_dependency, dir_out, dir_depmap = '../datasets/DepMap/'):
    # Preprocess depmap data, Q3 and Q4

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    # ------------------
    # parse P19Q3 data
    dm_data = depmap_data()
    dm_data.dir_datasets = os.path.join(dir_depmap, '19Q3')
    dm_data.data_name = 'data_19Q3'
    dm_data.load_data(useGene_dependency)
    dm_data.preprocess_data()  # handles formatting and missing data
    dm_data.filter_samples()  # only keep the shared_idx samples
    dm_data.filter_baseline()  # baseline filter, to remove invariant and low variant features

    # ------------------
    # parse P19Q3 data (repeat)
    dm_data_Q3 = depmap_data()
    dm_data_Q3.dir_datasets = os.path.join(dir_depmap, '19Q3')
    dm_data_Q3.data_name = 'data_19Q3'
    dm_data_Q3.load_data(useGene_dependency)
    dm_data_Q3.preprocess_data()  # handles formatting and missing data

    # parse P19Q4 data
    dm_data_Q4 = depmap_data()
    dm_data_Q4.dir_datasets = os.path.join(dir_depmap, '19Q4')
    dm_data_Q4.data_name = 'data_19Q4'
    dm_data_Q4.load_data(useGene_dependency)
    dm_data_Q4.preprocess_data()  # handles formatting and missing data

    # only keep the Q4 new cell lines
    samples_q3 = dm_data_Q3.df_crispr.index
    samples_q4 = dm_data_Q4.df_crispr.index
    new_samples_q4 = set(samples_q4) - set(samples_q3)
    dm_data_Q4.filter_samples(list(new_samples_q4))  # keep just the shared idx and only Q4

    # match features to that in Q3 (used for training)
    dm_data_Q4.match_feats(dm_data)

    # ------------------
    # print dataset stats
    dm_data.printDataStats(dir_out)
    dm_data_Q4.printDataStats(dir_out)

    pickle.dump(dm_data, open('%s/dm_data.pkl' % dir_out, 'wb'))
    pickle.dump(dm_data_Q4, open('%s/dm_data_Q4.pkl' % dir_out, 'wb'))

    return dm_data, dm_data_Q4
