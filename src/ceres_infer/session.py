# sessions
import os
import pandas as pd
import pickle
import logging
import glob
from sklearn.model_selection import train_test_split
import networkx as nx
import community
import matplotlib
import multiprocessing as mp
from functools import partial

from tqdm import tqdm
tqdm.pandas()

from ceres_infer.data import depmap_data
from ceres_infer.data import scale_data, build_data_gene, qc_feats, stats_Crispr
from ceres_infer.models import depmap_model, selectUnivariate
from ceres_infer.models import _featSelect_base as sf_base
from ceres_infer.utils import getFeatGene, getFeatSource
from ceres_infer.analyses import *

class workflow:
    def __init__(self, params):
        # Settings

        self.params = params
        self.pipeline = ['load_processed_data', 'infer', 'analyze', 'analyze_filtered', 'derive_genesets', 'run_Rscripts'] # default pipeline
        self.genes2analyz = None
        self.outdir_anlyz = None
        self.outdir_anlyzfilter = None
        self.outdir_network = None

        # set up dir
        if not os.path.exists(self.params['outdir_run']):
            os.makedirs(self.params['outdir_run'])

        if self.params['outdir_modtmp'] is not None:
            if not os.path.exists(self.params['outdir_modtmp']):
                os.mkdir(self.params['outdir_modtmp'])

        # save run settings
        f = open("%s/run_settings.txt" % (self.params['outdir_run']), "w")
        f.write('Model name: %s\n' % self.params['model_name'])
        f.write('Scope: %s\n' % self.params['scope'])
        f.write('Use gene dependency: %s\n' % self.params['useGene_dependency'])
        for k, v in self.params['model_params'].items():
            f.write('Model parameter %s: %s\n' % (k, v))
        for k, v in self.params['model_paramsgrid'].items():
            f.write('Model parameter grid search, %s: %s\n' % (k, v))
        f.write('Model data source: %s\n' % self.params['model_data_source'])
        f.write('External dataset used: %s\n' % self.params['external_data_name'])
        f.write('Model pipeline: %s\n' % self.params['model_pipeline'])
        f.write('Pipeline params: %s\n' % self.params['pipeline_params'])
        f.write('Scale data: %s\n' % self.params['opt_scale_data'])
        f.write('Scale data types: %s\n' % self.params['opt_scale_data_types'])
        f.write('Number of features in analysis set: %d\n' % self.params['anlyz_set_topN'])
        f.write('Number of samples to draw for null distribution, per target gene: %d\n' % self.params['perm_null'])
        f.write('Parallelization: %s\n' % self.params['parallelize'])
        f.write('Processes/CPUs to use: %s\n' % self.params['processes'])
        f.write('Run session notes: %s\n' % self.params['session_notes'])
        f.close()

    def create_pipe(self, pipeline):
        # Create pipeline
        self.pipeline = pipeline

    def run_pipe(self):
        # Run pipeline

        if self.pipeline is None:
            logging.warning('Pipeline is empty')
            return

        for compnt in self.pipeline:
            if hasattr(self, compnt) and callable(getattr(self, compnt)):
                getattr(self, compnt)()
            else:
                logging.warning('%s is not a component of the pipeline' % compnt)
                break

    def load_processed_data(self):
        # Load preprocessed data
        logging.info('Loading preprocessed data...')
        self.dm_data = pickle.load(open(self.params['indir_dmdata_Q3'],'rb'))
        self.dm_data_external = pickle.load(open(self.params['indir_dmdata_external'], 'rb'))

        if self.params['indir_landmarks'] is not None:
            self._add_landmarks()

    def _add_landmarks(self):
        logging.info('Adding landmarks...')
        self.dm_data.df_landmark = pd.read_csv(self.params['indir_landmarks'], header=0)
        self.dm_data_external.df_landmark = pd.read_csv(self.params['indir_landmarks'], header=0)

    def _select_scope(self):
        # Select which target genes to analyze

        if type(self.params['scope']) == list:
            # specific set of genes
            genes2analyz = self.params['scope'] # e.g. ['SOX10','KRAS','CDK4','EMC4','MAX','PTPN11']

        elif self.params['scope'] == 'all':
            # all genes
            genes2analyz = self.dm_data.df_crispr.columns.str.replace('\s.*','')

        elif self.params['scope'] == 'differential':
            # differentially dependent genes
            if (self.params['useGene_dependency']):
                df_crispr_stats = stats_Crispr(self.dm_data)
                filtered = df_crispr_stats.loc[~df_crispr_stats.entropy.isnull(), :]
                filtered = filtered.loc[(filtered.dependent > 10) & (filtered.not_dependent > 10), :]
                filtered = filtered.loc[filtered.entropy > 0.5, :]
                genes2analyz = filtered.index.map(lambda x: getFeatGene(x, firstOnly=True))
            else:
                df_crispr_stats = stats_Crispr(self.dm_data)
                gene_sel1 = set(df_crispr_stats[df_crispr_stats['std'] > 0.25].index)
                gene_sel2 = df_crispr_stats[df_crispr_stats['diff'] > 0.6].index
                gene_sel = gene_sel1.intersection(gene_sel2)
                genes2analyz = pd.Series(list(gene_sel)).apply(getFeatGene, firstOnly=True)

        if hasattr(self.dm_data, 'df_landmark') and (self.dm_data.df_landmark is not None):
            # if landmarks are used, exclude landmarks from the list of target genes
            genes2analyz = list(set(genes2analyz) - set(self.dm_data.df_landmark.landmark.str.replace('\s.*', '').values))

        self.genes2analyz = genes2analyz


    def infer(self):
        # Run inference
        logging.info('Running model building and inference...')

        # get scope
        if self.genes2analyz is None:
            self._select_scope()

        if self.params['parallelize']:
            model_results = pd.DataFrame()
            logging.info('Total number of processors available: %d' % mp.cpu_count())
            logging.info('Total number of processors to use: %d' % self.params['processes'])
            with mp.Pool(self.params['processes']) as pool:
                pfunc = partial(workflow.infer_gene,
                                params=self.params,
                                dm_data=self.dm_data,
                                dm_data_external=self.dm_data_external)
                processesN = min(self.params['processes'], mp.cpu_count())
                chunksize = max(1, len(self.genes2analyz) // processesN)
                for df_res in tqdm(pool.imap_unordered(pfunc, self.genes2analyz, chunksize), total=len(self.genes2analyz)):
                    model_results = model_results.append(df_res, ignore_index=True, sort=False)

        else:
            model_results = pd.DataFrame()
            for gene2anlyz in tqdm(self.genes2analyz):
                df_res = self._infer_gene(gene2anlyz)
                model_results = model_results.append(df_res, ignore_index=True, sort=False)

        # change the score/corr columns to type float
        for col in model_results.columns[model_results.columns.str.startswith('score') | model_results.columns.str.startswith('corr')]:
            model_results[col] = pd.to_numeric(model_results[col])

        # -------
        # save results
        model_results.reset_index(inplace=True, drop=True)
        model_results.to_csv('%s/model_results.csv' % (self.params['outdir_run']), index=False)

        self.model_results = model_results


    def load_model_results(self):
        # Load model results
        if self.genes2analyz is None:
            self._select_scope()

        logging.info('Loading model results...')
        self.model_results = pd.read_csv('%s/model_results.csv' % self.params['outdir_run'], header=0)


    def analyze(self):
        # Analyze results
        logging.info('Analyzing model results...')

        self.outdir_anlyz = os.path.join(self.params['outdir_run'], 'anlyz')
        if not os.path.exists(self.outdir_anlyz):
            os.makedirs(self.outdir_anlyz)

        # settings
        plt.interactive(False)

        # some processing of model results
        model_results = self.model_results
        counts = model_results.groupby('target')['target'].count()
        model_results = model_results.loc[model_results.target.isin(counts[counts > 1].index), :]  # exclude ones with no reduced models

        #------- high-level stats -------
        df = model_results.loc[model_results.model == 'topfeat',]
        f = open("%s/model_results_stats.txt" % (self.outdir_anlyz), "w")
        f.write('There are %d genes out of %d with no feature in reduced data\n' % (sum(df.feature == ''), df.shape[0]))
        f.close()

        anlyz_model_results(model_results, outdir_sub='%s/stats_score_aggRes/' % self.outdir_anlyz, suffix='')

        #------- aggregate summaries -------
        def getAggSummary(x):
            # get variance explained
            df = x.loc[x.model == 'all', ['target', self.params['metric_eval']]].copy()
            df.columns = ['target', 'score_full']
            df.score_full = round(df.score_full, 5)
            df['score_rd'] = round(x.loc[x.model == 'topfeat', self.params['metric_eval']].values[0], 5) if sum(
                x.model == 'topfeat') > 0 else np.nan
            df['score_rd10'] = round(x.loc[x.model == 'top10feat', self.params['metric_eval']].values[0], 5) if sum(
                x.model == 'top10feat') > 0 else np.nan
            df['corr_rd10'] = round(x.loc[x.model == 'top10feat', 'corr_test'].values[0], 5) if sum(
                x.model == 'top10feat') > 0 else np.nan
            df['recall_rd10'] = round(x.loc[x.model == 'top10feat', 'corr_test_recall'].values[0], 5) if sum(
                x.model == 'top10feat') > 0 else np.nan
            df['external_score_rd10'] = round(x.loc[x.model == 'top10feat', 'score_external'].values[0], 5) if sum(
                x.model == 'top10feat') > 0 else np.nan
            df['external_corr_rd10'] = round(x.loc[x.model == 'top10feat', 'corr_external'].values[0], 5) if sum(
                x.model == 'top10feat') > 0 else np.nan
            df['external_recall_rd10'] = round(x.loc[x.model == 'top10feat', 'corr_external_recall'].values[0], 5) if sum(
                x.model == 'top10feat') > 0 else np.nan

            return df

        aggRes = model_results.groupby('target').apply(getAggSummary)
        aggRes.reset_index(inplace=True, drop=True)

        # write varExp
        aggRes.to_csv("%s/agg_summary.csv" % self.outdir_anlyz, index=False)

        anlyz_aggRes(aggRes, self.params, outdir_sub='%s/stats_score_aggRes/' % self.outdir_anlyz, suffix='' )

        #------- aggregate feature summaries -------
        # -- feature summary, variance ratios etc --
        def getVarExp(x):
            # get variance explained
            df = x.loc[x.model == 'univariate', ['feature', 'target', self.params['metric_eval']]].copy()
            df.columns = ['feature', 'target', 'score_ind']
            df.score_ind = round(df.score_ind, 5)
            df['score_rd'] = round(x.loc[x.model == 'top10feat', self.params['metric_eval']].values[0], 5) if sum(
                x.model == 'topfeat') > 0 else np.nan
            df['score_full'] = round(x.loc[x.model == 'all', self.params['metric_eval']].values[0], 5)
            df['varExp_ofFull'] = round(df.score_ind / df.score_full, 5)
            df['varExp_ofRd'] = round(df.score_ind / df.score_rd, 5)
            df['feat_idx'] = list(range(1, df.shape[0] + 1))
            return df

        varExp = model_results.groupby('target').apply(getVarExp)
        varExp.reset_index(inplace=True, drop=True)
        varExp['feat_gene'] = varExp['feature'].apply(getFeatGene, firstOnly=True)
        varExp['feat_source'] = varExp['feature'].apply(getFeatSource, firstOnly=True)

        # write varExp
        varExp.to_csv("%s/feat_summary_varExp.csv" % self.outdir_anlyz, index=False)

        # -- analyze varExp (related to R2s) (not collapsed - each row = one feat-target pair) --
        anlyz_varExp(varExp, outdir_sub='%s/stats_score_feat/' % self.outdir_anlyz, suffix='')
        anlyz_varExp_wSource(varExp, self.dm_data, outdir_sub='%s/stats_score_feat/' % self.outdir_anlyz, suffix='')
        anlyz_scoresGap(varExp, self.params['useGene_dependency'], outdir_sub='%s/stats_score_feat/' % self.outdir_anlyz)

        # -- analyze feat summary (related to feat (gene/source)) (collapsed - each row = one target gene) --
        anlyz_varExp_feats(varExp, outdir_sub=self.outdir_anlyz, gs_dir=self.params['indir_genesets'])

        #------- Y predictions comparisons -------
        # create dir
        outdir_ycompr = '%s/heatmaps/' % self.outdir_anlyz
        if not os.path.exists(outdir_ycompr):
            os.makedirs(outdir_ycompr)

        genes2analyz = model_results.target.unique()
        y_compr_fnames = glob.glob(os.path.join(self.params['outdir_modtmp'], 'y_compr_*.pkl'))

        if (len(y_compr_fnames) > 0) and (len(genes2analyz) > 0):
            for data_suffix in ['tr', 'te', 'ext']:
                df_y_actual, df_y_pred = constructYCompr(genes2analyz, data_suffix, self.params['outdir_modtmp'])
                pickle.dump({'actual': df_y_actual, 'predicted': df_y_pred}, open(f'{self.outdir_anlyz}/y_compr_{data_suffix}.pkl', 'wb'))
                yComprHeatmap(df_y_actual, df_y_pred, data_suffix, outdir_ycompr)

        #------- Concordance -------
        outdir_concord = '%s/concordance/' % self.outdir_anlyz
        if not os.path.exists(outdir_concord):
            os.makedirs(outdir_concord)

        y_compr_fnames = glob.glob(os.path.join(self.params['outdir_modtmp'], 'y_compr_*.pkl'))
        if len(y_compr_fnames) > 0:
            for data_suffix in ['tr', 'te', 'ext']:
                df_conc = pd.DataFrame()
                for fname in y_compr_fnames:
                    f = re.sub('.*_compr_', '', fname)
                    gene = re.sub('\.pkl', '', f)
                    df = pickle.load(open(fname, 'rb'))

                    tmp = pd.DataFrame([{'gene': gene, 'concordance': getConcordance(df[data_suffix])}])
                    df_conc = pd.concat([df_conc, tmp])

                df_conc.to_csv(f'{outdir_concord}/concordance_{data_suffix}.csv', index=False)

                plt.figure()
                ax = sns.distplot(df_conc.concordance)
                ax.set(xlim=[0, 1.05], xlabel='Concordance', title='Concordance between actual and predicted')
                plt.savefig(f"{outdir_concord}/concordance_{data_suffix}.pdf")
                plt.close()

        #------- Examine sources -------
        # check for in the selected features
        counts = {'CERES': 0,
                  'RNA-seq': 0,
                  'CN': 0,
                  'Mut': 0,
                  'Lineage': 0}
        df_lineage = pd.DataFrame()
        for fname in glob.glob('%s/feats_*.csv' % self.params['outdir_modtmp']):
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
        df_counts.to_csv('%s/featSrcCounts/source_counts_allfeatures.csv' % self.outdir_anlyz, index=True, header=False)
        df_lineage.to_csv('%s/featSrcCounts/source_allfeatures_lineage.csv' % self.outdir_anlyz)

        plotCountsPie(df_counts,
                      'Data source summary (all features)',
                      'imprank-allfeat_pie',
                      '%s/featSrcCounts/' % self.outdir_anlyz,
                      autopct='%0.2f%%')


    def analyze_filtered(self):
        if self.outdir_anlyz is None:
            logging.warning('workflow::analyze has to be run first before running analyze_filtered')
            return

        logging.info('Analyzing filtered results...')

        #------- settings -------
        plt.interactive(False)

        # create dir
        self.outdir_anlyzfilter = os.path.join(self.params['outdir_run'], 'anlyz_filtered')
        if not os.path.exists(self.outdir_anlyzfilter):
            os.makedirs(self.outdir_anlyzfilter)

        # save settings
        f = open(os.path.join(self.outdir_anlyzfilter, 'filter_settings.txt'), 'w')
        for k, v in self.params['thresholds'].items():
            f.write('%s threshold: %.2f\n' % (k, v))
        f.close()

        #-------  model results filtering and analyses -------
        model_results = self.model_results
        counts = model_results.groupby('target')['target'].count()
        model_results = model_results.loc[model_results.target.isin(counts[counts > 1].index), :]  # exclude ones with no reduced models
        df_res_filtered = model_results.copy()
        genes_pass1 = df_res_filtered.loc[
            (df_res_filtered.model == 'top10feat') & (df_res_filtered.score_test > self.params['thresholds']['score_rd10']), 'target']
        genes_pass2 = df_res_filtered.loc[(df_res_filtered.model == 'top10feat') & (
                    df_res_filtered.corr_test_recall > self.params['thresholds']['recall_rd10']), 'target']
        genes_pass = set(genes_pass1).intersection(genes_pass2)
        df_res_filtered = df_res_filtered.loc[df_res_filtered.target.isin(genes_pass), :]
        anlyz_model_results(df_res_filtered, outdir_sub='%s/stats_score_aggRes/' % self.outdir_anlyzfilter, suffix='')

        #-------  aggRes filtering and analyses -------
        # read in files
        aggRes = pd.read_csv(os.path.join(self.outdir_anlyz, 'agg_summary.csv'), header=0)

        # filter based on thresholds
        aggRes_filtered = aggRes.copy()
        for k, v in self.params['thresholds'].items():
            aggRes_filtered = aggRes_filtered.loc[aggRes_filtered[k] > v, :]

        # save file
        aggRes_filtered.reset_index(inplace=True, drop=True)
        aggRes_filtered.to_csv("%s/agg_summary_filtered.csv" % self.outdir_anlyzfilter, index=False)

        # analyze
        anlyz_aggRes(aggRes_filtered, self.params, outdir_sub='%s/stats_score_aggRes/' % self.outdir_anlyzfilter, suffix='')

        #-------  varExp filtering and analyses -------
        # read in files
        varExp = pd.read_csv(os.path.join(self.outdir_anlyz, 'feat_summary_varExp.csv'), header=0)

        # filter based on thresholds
        varExp_filtered = varExp.loc[varExp.target.isin(aggRes_filtered.target), :].copy()

        # save file
        varExp_filtered.reset_index(inplace=True, drop=True)
        varExp_filtered.to_csv("%s/feat_summary_varExp_filtered.csv" % self.outdir_anlyzfilter, index=False)

        # stats
        f = open(os.path.join(self.outdir_anlyzfilter, 'filter_stats.txt'), 'w')
        f.write('varExp: %d source-target pairs, %d genes\n' % (varExp.shape[0], len(varExp.target.unique())))
        f.write('varExp_filtered: %d source-target pairs, %d genes\n' % (
        varExp_filtered.shape[0], len(varExp_filtered.target.unique())))
        f.close()

        # analyze
        anlyz_varExp(varExp_filtered, outdir_sub='%s/stats_score_feat/' % self.outdir_anlyzfilter, suffix='score filtered')
        anlyz_varExp_feats(varExp_filtered, outdir_sub=self.outdir_anlyzfilter, gs_dir=self.params['indir_genesets'])
        anlyz_varExp_wSource(varExp_filtered, self.dm_data, outdir_sub='%s/stats_score_feat/' % self.outdir_anlyzfilter, suffix='')
        anlyz_scoresGap(varExp_filtered, self.params['useGene_dependency'], outdir_sub='%s/stats_score_feat/' % self.outdir_anlyzfilter)

    def derive_genesets(self):
        if self.outdir_anlyzfilter is None:
            logging.warning('workflow::analyze_filtered has to be run first before running derive_genesets')
            return

        logging.info('Deriving gene sets...')

        #------- settings -------
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        plt.interactive(False)

        self.outdir_network = os.path.join(self.params['outdir_run'], 'network')
        if not os.path.exists(self.outdir_network):
            os.makedirs(self.outdir_network)

        # save settings
        f = open('%s/run_settings.txt' % self.outdir_network, 'w')
        f.write('Minimum gene set size: %d\n' % self.params['min_gs_size'])
        f.close()

        # read in files
        varExp_filtered = pd.read_csv(os.path.join(self.outdir_anlyzfilter, 'feat_summary_varExp_filtered.csv'), header=0)

        #------- derive networks -------
        df = varExp_filtered[['feat_gene', 'target']]
        df.to_csv('%s/varExp_filtered.adjlist' % self.outdir_network, sep=' ', header=False, index=False)

        # load into networkx
        f = open('%s/varExp_filtered.adjlist' % self.outdir_network, 'rb')
        G = nx.read_adjlist(f)

        # derive communities
        communities = community.best_partition(G)  # based on the Louvain method
        nx.set_node_attributes(G, communities, 'modularity')

        modularity = {}
        for k, v in communities.items():
            if v not in modularity:
                modularity[v] = [k]
            else:
                modularity[v].append(k)

        pickle.dump(modularity, open('%s/modularity.pkl' % self.outdir_network, 'wb'))

        # write gene sets file
        f = open('%s/gs.txt' % self.outdir_network, 'w')
        for n in modularity:
            if (len(modularity[n]) > self.params['min_gs_size']):
                f.write('gene_set_%d\t\t%s\n' % (n, '\t'.join(modularity[n])))
        f.close()

        # merge with varExp_filtered
        df_communities = pd.DataFrame.from_dict(communities, orient='index', columns=['Class'])
        varExp_filtered = varExp_filtered.merge(df_communities, left_on='feat_gene', right_index=True)
        varExp_filtered.to_csv("%s/feat_summary_varExp_filtered_class.csv" % self.outdir_network, index=False)

        #------- stats -------
        # gene set sizes
        gps_len = []
        for k, v in modularity.items():
            gps_len.append(len(v))

        plt.plot()
        ax = sns.distplot(gps_len)
        ax.set(xlabel='Gene set size', ylabel='Frequency', title='Median: %d' % np.median(gps_len))
        plt.savefig('%s/gs_sizes.png' % self.outdir_network)
        plt.close()

        # write gene sets stats to file
        pd.Series(gps_len).describe().to_csv('%s/gs_stats.csv' % self.outdir_network, header=False)

        # network stats
        # properties
        f = open('%s/network_stats.txt' % self.outdir_network, 'w')
        f.write('Total number of nodes: %d\n' % len(G.nodes))
        f.write('Total number of edges: %d\n' % len(G.edges))
        f.close()

        deg = sorted(d for n, d in G.degree())
        plt.figure()
        ax = sns.distplot(deg)
        ax.set(xlabel='Degree', ylabel='Frequency')
        plt.savefig('%s/network_degree_dist.png' % self.outdir_network)

        # visualize network
        pos = nx.spring_layout(G)  # compute layout
        cmap = plt.cm.get_cmap('RdYlBu', len(modularity))

        plt.figure(figsize=(9, 9))
        plt.axis('off')
        for k, v in modularity.items():
            if (len(v) > self.params['min_gs_size']):
                val = np.random.randint(0, len(modularity) - 1)
                nx.draw_networkx_nodes(G, pos, node_size=5, nodelist=v, node_color=cmap(val))
                nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='k', style='solid', width=1, edgelist=G.edges(v))
        plt.savefig('%s/network.png' % self.outdir_network)
        # plt.show(G)

        plt.figure(figsize=(9, 9))
        plt.axis('off')
        for k, v in modularity.items():
            if (len(v) > 30):
                val = np.random.randint(0, len(modularity) - 1)
                nx.draw_networkx_nodes(G, pos, node_size=5, nodelist=v, node_color=cmap(val))
                nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='k', style='solid', width=1, edgelist=G.edges(v))
        plt.savefig('%s/network_zoom.png' % self.outdir_network)
        # plt.show(G)

    def _get_analysis_paths(self):
        # get the output paths for analysis results
        # useful if analyses are already done, and want to populate the oudir_ variables with the correct paths

        self.outdir_anlyz = os.path.join(self.params['outdir_run'], 'anlyz')
        self.outdir_anlyzfilter = os.path.join(self.params['outdir_run'], 'anlyz_filtered')
        self.outdir_network = os.path.join(self.params['outdir_run'], 'network')

    def run_Rscripts(self):
        os.system('Rscript "%s" "%s" "%s"' % (os.path.join(os.path.dirname(__file__), 'analyses.R'),
                                              os.path.abspath(self.outdir_anlyz),
                                              os.path.abspath(self.outdir_anlyzfilter)) )

    def _infer_gene(self, gene2analyz):
        '''
        This is the class method that will call the static infer_gene method
        used as a helper link between a class instance to calling the static method
        :param gene2analyz: target gene name to infer
        :return: results data frame
        '''
        return workflow.infer_gene(gene2analyz, self.params, self.dm_data, self.dm_data_external)

    @staticmethod
    def infer_gene(gene2anlyz, params, dm_data, dm_data_external):
        '''
        Static method that builds model to infer target gene
        :param gene2anlyz: target gene name to infer
        :param params: params of workflow
        :param dm_data: obj for dm_data Q3
        :param dm_data_Q4: obj for dm_data Q4
        :param pbar: progress bar obj from tqdm
        :return: results data frame
        '''
        df_res = pd.DataFrame()

        # --- create datasets ---
        data_name, df_x, df_y, df_y_null = build_data_gene(params['model_data_source'], dm_data, gene2anlyz)
        data_name, df_x_external, df_y_external, df_yn_external = build_data_gene(params['model_data_source'], dm_data_external, gene2anlyz)

        # data checks
        if len(df_x) < 1 or len(df_y) < 1 or len(df_x_external) < 1 or len(df_y_external) < 1:  # empty x or y data
             return None

        if not qc_feats([df_x, df_x_external]):
            raise ValueError('Feature name/order across the datasets do not match')

        # set up the data matrices and feature labels
        feat_labels = pd.DataFrame({'name': df_x.columns.values,
                                    'gene': pd.Series(df_x.columns).apply(getFeatGene, firstOnly=True),
                                    'source': pd.Series(df_x.columns).apply(getFeatSource, firstOnly=True)})
        feat_labels.index.name = 'feat_id'
        x_vals = df_x.values
        y_vals = df_y.values.ravel()
        yn_vals = df_y_null.values
        x_external = df_x_external.values
        y_external = df_y_external.values.ravel()
        yn_external_vals = df_yn_external.values

        # split to train/test
        if params['useGene_dependency']:  # if doing classification, make sure the split datasets are balanced
            x_train, x_test, y_train, y_test, yn_train, yn_test = train_test_split(x_vals, y_vals, yn_vals,
                                                                                   test_size=0.15, random_state=42,
                                                                                   stratify=y_vals)
        else:
            x_train, x_test, y_train, y_test, yn_train, yn_test = train_test_split(x_vals, y_vals, yn_vals,
                                                                                   test_size=0.15, random_state=42)

        # scale data if needed
        if params['opt_scale_data']:
            to_scale_idx = df_x.columns.str.contains(params['opt_scale_data_types'])
            if any(to_scale_idx):
                x_train, x_test, x_external = scale_data(x_train, [x_test, x_external], to_scale_idx)
            else:
                logging.info( "Trying to scale data, but the given data type is not found and cannot be scaled for gene %s" % gene2anlyz)

        # set up model
        dm_model = depmap_model(params['model_name'], params['model_params'], params['model_paramsgrid'],
                                outdir=params['outdir_run'])

        # --- model pipeline ---
        data = {'train': {'x': x_train, 'y': y_train}, 'test': {'x': x_test, 'y': y_test}}
        data_null = {'test': {'x': x_test, 'y': y_test, 'y_null': yn_test}}

        df_res = pd.DataFrame()
        df_res, sf = params['model_pipeline'](data, dm_model, feat_labels, gene2anlyz, df_res,
                                              params['useGene_dependency'], data_null,
                                              params['perm_null'],
                                              **params['pipeline_params'])

        if sf is None:
            return df_res  # feature selection in the end is empty

        feats = sf.importance_sel

        # --- analysis set ---
        # pick a list of top N features
        feat_sel = sf.importance_sel.iloc[0:params['anlyz_set_topN'], :]
        x_tr, x_te, x_external_rd = sf_base().transform_set(x_train, x_test, x_external, feat_idx=feat_sel.index)

        # reduced model on the top N features
        data = {'train': {'x': x_tr, 'y': y_train},
                'test': {'x': x_te, 'y': y_test},
                'external': {'x': x_external_rd, 'y': y_external}}
        data_null = {'test': {'x': x_te, 'y': y_test, 'y_null': yn_test},
                     'external': {'x': x_external_rd, 'y': y_external, 'y_null': yn_external_vals}}

        dm_model.fit(x_tr, y_train, x_te, y_test)
        df_res_sp = dm_model.evaluate(data, 'top10feat', 'top10feat', gene2anlyz, data_null, params['perm_null'])
        df_res = df_res.append(df_res_sp, sort=False)
        if params['outdir_modtmp'] is not None:
            pickle.dump(dm_model.model,
                        open('%s/model_rd10_%s.pkl' % (params['outdir_modtmp'], gene2anlyz),
                             "wb"))  # pickle top10feat model

        # save the y actual and predicted, using the reduced model
        y_compr = {'tr': pd.DataFrame({'y_actual': y_train,
                                       'y_pred': dm_model.predict(x_tr)}),
                   'te': pd.DataFrame({'y_actual': y_test,
                                       'y_pred': dm_model.predict(x_te)}),
                   'ext': pd.DataFrame({'y_actual': y_external,
                                        'y_pred': dm_model.predict(x_external_rd)})
                   }
        if params['outdir_modtmp'] is not None:
            pickle.dump(y_compr,
                        open('%s/y_compr_%s.pkl' % (params['outdir_modtmp'], gene2anlyz),
                             "wb"))  # pickle y_actual vs y_pred

        # univariate on the top N features
        sf = selectUnivariate(dm_model)
        sf.fit(x_tr, y_train, x_te, y_test, feat_sel.feature, target_name=gene2anlyz)
        df_res = df_res.append(sf.importance.reset_index(), sort=False)

        # --- saving results ---
        if params['outdir_modtmp'] is not None:
            feats.to_csv('%s/feats_%s.csv' % (params['outdir_modtmp'], gene2anlyz), index=True)

        return df_res
