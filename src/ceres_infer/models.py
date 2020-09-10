#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Models
@author: boyangzhao
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.svm import SVR
#from xgboost import XGBClassifier
from sklearn.dummy import DummyRegressor

from keras.layers import Dense, Input, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import regularizers
from keras.callbacks import EarlyStopping

from sklearn.metrics import roc_curve, auc, r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectKBest, f_classif, chi2, f_regression
from boruta import BorutaPy

from statsmodels.stats.multitest import multipletests
from sklearn.metrics import matthews_corrcoef
from scipy.stats import spearmanr

import matplotlib.pyplot as plt
import seaborn as sns


######################################################################
# Model class
######################################################################
class depmap_model:

    #----------------------------
    # Model building
    #----------------------------
    def __init__(self, mod_name, mod_params={}, param_grid={}, outdir='./'):
        self.mod_name = mod_name
        self.metric = None
        self.mod_params = mod_params
        self.outdir = outdir
        self.param_grid = param_grid

        # regression
        if self.mod_name == 'lm':
            self.model = LinearRegression()  # linear regression model
            self.metric = 'R2'
        elif self.mod_name == 'elasticNet':
            alpha = mod_params['alpha'] if 'alpha' in mod_params else 0.03
            l1_ratio = mod_params['l1_ratio'] if 'l1_ratio' in mod_params else 0.5
            self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)  #l1=lasso; l2=ridge; l1_ratio=1 means l1
            self.metric = 'R2'
        elif self.mod_name == 'rf':
            n_estimators = mod_params['n_estimators'] if 'n_estimators' in mod_params else 100
            max_depth = mod_params['max_depth'] if 'max_depth' in mod_params else 10
            min_samples_leaf = mod_params['min_samples_leaf'] if 'min_samples_leaf' in mod_params else 5
            max_features = mod_params['max_features'] if 'max_features' in mod_params else 'sqrt'
            self.model = RandomForestRegressor(n_estimators=n_estimators,
                                                max_depth=max_depth,
                                                min_samples_leaf=min_samples_leaf,
                                                max_features=max_features,
                                                oob_score=True,
                                                random_state = 42) #random forest
            self.metric = 'R2'
        elif self.mod_name == 'svr':
            self.model = SVR(kernel=mod_params['kernel'], C=mod_params['C']) #SVR linear kernel
            self.metric = 'R2'
        elif self.mod_name == 'dummy_reg':
            self.model = DummyRegressor(quantile=0.5)
            self.metric = 'R2'
        elif self.mod_name == 'mlp':
            # TODO note neural-net based models were tested separately, some of the code are integrated here, but \
            #  MLPs are not fully functional yet with the libraries written here
            K.clear_session()
            input_data = Input(shape=(mod_params['feat_size'],))

            for n in range(mod_params['num_layers']):
                if(n==0):
                    x = input_data
                #dense_name = 'Dense_%d' % n
                x = Dense(64*(2**(2*(mod_params['num_layers']-n-1))), kernel_regularizer=regularizers.l1(0.01))(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = Dropout(0.5)(x)
            encoded = Dense(1)(x)

            model = Model(input_data, encoded)
            model.compile(optimizer='adam', loss='mean_squared_error')

            self.model = model
            self.metric = 'R2'

        # classification
        elif self.mod_name == 'mlp_classify':
            K.clear_session()
            input_data = Input(shape=(mod_params['feat_size'],))

            for n in range(mod_params['num_layers']):
                if(n==0):
                    x = input_data
                # dense_name = 'Dense_%d' % n
                x = Dense(64*(2**(2*(mod_params['num_layers']-n-1))), kernel_regularizer=regularizers.l1(0.01))(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = Dropout(0.5)(x)
            encoded = Dense(1, activation='sigmoid')(x)

            model = Model(input_data, encoded)
            model.compile(optimizer='adam', loss='binary_crossentropy')

            self.model = model
            self.metric = 'AUC'
        elif self.mod_name == 'logit':
            self.model = LogisticRegression(penalty='l2', class_weight='balanced')
            self.metric = 'AUC'
        elif self.mod_name == 'rfc':
            n_estimators = mod_params['n_estimators'] if 'n_estimators' in mod_params else 100
            max_depth = mod_params['max_depth'] if 'max_depth' in mod_params else 10
            min_samples_leaf = mod_params['min_samples_leaf'] if 'min_samples_leaf' in mod_params else 5
            max_features = mod_params['max_features'] if 'max_features' in mod_params else 'sqrt'
            self.model = RandomForestClassifier(n_estimators=n_estimators,
                                                max_depth=max_depth,
                                                min_samples_leaf=min_samples_leaf,
                                                max_features=max_features,
                                                oob_score=True,
                                                class_weight='balanced',
                                                random_state = 42)
            self.metric = 'AUC'
        elif self.mod_name == 'xgboost':
            # Not in use - XGBClassifier has additional requirements that need to be configured
            # n_estimators = mod_params['n_estimators'] if 'n_estimators' in mod_params else 100
            # self.model = XGBClassifier(n_estimators=n_estimators)
            # self.metric = 'AUC'
            self.model = None
            self.metric = 'AUC'

    #----------------------------
    # Fitting methods
    #----------------------------
    def fit_tuned(self, x_train, y_train, x_test=None, y_test=None):
        # fit model with hyperparameter tuning
        if self.mod_name in ['lm','elasticNet','rf','svr','logit','rfc']:
            gsc = GridSearchCV(estimator=self.model,
                               param_grid=self.param_grid,
                               cv=5)
            gsc_res = gsc.fit(x_train, y_train)
            self.model = gsc_res.best_estimator_

        self.x_train = x_train
        self.y_train = y_train


    def fit(self, x_train, y_train, x_test=None, y_test=None):
        # fit model
        if len(self.param_grid) > 0:
            self.fit_tuned(x_train, y_train, x_test, y_test)
            return

        # regression
        if self.mod_name in ['lm', 'elasticNet', 'rf', 'svr', 'dummy_reg']:
            #sklearn regression models
            self.model.fit(x_train,y_train)
        elif self.mod_name in ['mlp']:
            epochs = self.mod_params['epochs'] if 'epochs' in self.mod_params else 100
            es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5,verbose=0, mode='auto')
            H = self.model.fit(x_train, y_train, epochs=epochs, batch_size=256, shuffle=True,
                               validation_data=(x_test, y_test),
                               callbacks=[es])

            N = np.arange(0, len(H.history['loss']))
            plt.figure()
            plt.plot(N, H.history['loss'], label='train_loss')
            plt.plot(N, H.history['val_loss'], label='val_loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.savefig('%s/model_train_history.png' % self.outdir)

        # classification
        elif self.mod_name in ['logit','rfc','xgboost']:
            #sklearn classification models
            self.model.fit(x_train,y_train)

        self.x_train = x_train
        self.y_train = y_train

    #----------------------------
    # Score/predict
    #----------------------------
    def predict(self, x):
        # regression
        if self.mod_name in ['lm', 'elasticNet', 'rf', 'svr', 'mlp', 'dummy_reg']:
            #sklearn regression models, return R2
            return self.model.predict(x)

        # classification
        elif self.mod_name in ['logit','rfc','xgboost']:
            return self.model.predict_proba(x)[:,1]

    def score(self, x, y):
        # regression
        if self.mod_name in ['lm', 'elasticNet', 'rf', 'svr', 'dummy_reg']:
            #sklearn regression models, return R2
            return self.model.score(x,y)
        elif self.mod_name in ['mlp']:
            y_pred = np.squeeze(self.model.predict(x))
            return r2_score(y, y_pred)


        # classification
        elif self.mod_name in ['logit','rfc','xgboost']:
            #sklearn classification models, return AUC
            if len(np.unique(y))<2:
                return np.nan
            y_pred = self.model.predict_proba(x)[:,1]
            return roc_auc_score(y, y_pred)

    def scoreCV(self, x, y, cv=3): #cross-validation scoring
        if self.mod_name in ['lm','elasticNet','rf','svr']:
            return cross_val_score(self.model, x, y, cv, scoring='r2')
        elif self.mod_name in ['logit','rfc','xgboost']:
            return cross_val_score(self.model, x, y, cv, scoring='roc_auc')


    #----------------------------
    # Evaluations - plots
    #----------------------------
    def plot_actualpred(self, x, y, title, fname, outdir):
        y_pred = self.predict(x)
        plt.figure()
        ax=sns.scatterplot(y_pred, y, alpha=0.4, s=100)
        ax.set(ylabel='y actual', xlabel='y predicted', title=title)
        plt.savefig('%s/%s.png' % (outdir,fname))
        plt.close()

    def plot_roc(self, x, y, title, fname, outdir):
        y_pred = self.predict(x)
        fpr, tpr, thresholds = roc_curve(y, y_pred)
        auc_val = auc(fpr, tpr)

        plt.figure()
        plt.plot([0,1],[0,1],'k--')
        plt.plot(fpr, tpr)
        plt.ylabel('True positive rate')
        plt.xlabel('False positive rate')
        plt.title('%s | AUC:%.2f' % (title,auc_val))
        plt.savefig('%s/%s.png' % (outdir,fname))
        plt.close()

    def eval_plot(self, x, y, title, fname, outdir):
        # regression
        if self.mod_name in ['lm', 'elasticNet', 'rf', 'svr', 'mlp', 'dummy_reg']:
            #sklearn regression models, return R2
            self.plot_actualpred(x, y, title, fname, outdir)

        # classification
        elif self.mod_name in ['logit', 'rfc', 'xgboost']:
            self.plot_roc(x, y, title, fname, outdir)


    #----------------------------
    # Results table
    #----------------------------
    def corr(self, y_actual, y_pred):
        if self.metric == 'R2':
            return spearmanr(y_actual, y_pred).correlation
        elif self.metric == 'AUC':
            return matthews_corrcoef(y_actual, y_pred>0.5)

    def corr_recall(self, y_actual, y_pred, y_null, perm=100):
        # returns the correlation value recall, based on null distribution
        corr_val = self.corr(y_actual, y_pred)
        if np.isnan(corr_val):
            return np.nan, np.nan

        rand_idx = np.random.choice(y_null.shape[1], perm)
        s_null = np.array([])
        for n in rand_idx:
            if (self.metric == 'AUC') & (len(np.unique(y_null[:,n]))<2):
                continue
            s_null = np.append(s_null, self.corr(y_null[:,n], y_pred))

        recall = sum(corr_val > s_null) / len(s_null)
        return corr_val, recall

    def evaluate(self, data, mod_descrip, feat_names, target_name, data_null=None, perm=100):
        # evaluate model with the given datasets, and returns a formatted results table
        # mod_descrip: model description (short)
        # feat_names: feature names, if array, will be concatenated with ','
        # target_name: target name
        # score_: score of given datasets

        feat_names_joined = feat_names if isinstance(feat_names,str) else feat_names.str.cat(sep=',')

        res_dict = {'model': mod_descrip,
                    'feature':feat_names_joined,
                    'target':target_name}

        for n,d in data.items():
            res_dict.update({ 'score_%s'%n:self.score(d['x'], d['y'])})

        if hasattr(self.model, 'oob_score_'):
            res_dict.update({'score_oob': self.model.oob_score_})

        if data_null is not None:
            for n,d_null in data_null.items():
                y_pred = self.predict(d_null['x'])
                corr_val, recall = self.corr_recall(d_null['y'], y_pred, d_null['y_null'], perm)
                res_dict.update({'corr_%s' % n: corr_val,
                                 'corr_%s_recall' % n: recall})

        return pd.DataFrame.from_dict(res_dict, orient='index').T

######################################################################
# Feature selection
######################################################################

#--------------------------------------------------------------------
# Base classes

class _featSelect_base:
    def __init__(self):
        self.importance = None #importance scores, e.g. coef or importance value from the model; the index values match the feature indices
        self.importance_sel = None #selected
        self.feat_names = None #a data frame

    def transform(self, x, feat_idx=None):
        #get only subset of the given dataset
        #feat_idx if defined, will only use a subset of the features, based on index
        if (feat_idx is None) & (self.importance_sel is not None):
            feat_idx = self.importance_sel.feat_idx

        if feat_idx is not None:
            if isinstance(feat_idx, (int, np.integer)):
                x = x.copy()[:,[feat_idx]]
            else:
                x = x.copy()[:,feat_idx]

        return x

    def transform_set(self, *x, feat_idx=None):
        return [self.transform(x_item, feat_idx) for x_item in x]


class _selectImpFeat(_featSelect_base):
    def __init__(self, dm_model, feat_names=None):
        # get the n most important features
        # returns the feature index and corresponding rank, sorted
        super().__init__()

        self.feat_names = feat_names
        feat_idx = []
        feat_imp = []
        topfeat_idx = []
        topfeat_imp = []

        # get feature importance
        if dm_model.mod_name in ['lm','elasticNet']:
            coef_abs = np.abs(dm_model.model.coef_)
            idx_coef_pair = sorted(enumerate(coef_abs), key=lambda x: x[1], reverse=True)
            feat_idx = [n[0] for n in idx_coef_pair]
            feat_imp = [n[1] for n in idx_coef_pair]
        elif dm_model.mod_name in ['rf','rfc','xgboost']:
            feat_idx = np.argsort(dm_model.model.feature_importances_)[::-1]
            feat_imp = dm_model.model.feature_importances_[feat_idx]
        elif dm_model.mod_name in ['logit']:
            coef_abs = np.abs(dm_model.model.coef_[0])
            idx_coef_pair = sorted(enumerate(coef_abs), key=lambda x: x[1], reverse=True)
            feat_idx = [n[0] for n in idx_coef_pair]
            feat_imp = [n[1] for n in idx_coef_pair]
        elif dm_model.mod_name is 'dummy_reg':
            feat_idx = self.feat_names.index if self.feat_names is not None else None
            feat_imp = [-1]*len(self.feat_names) if self.feat_names is not None else None

        # check lengths
        if self.feat_names is not None:
            if len(self.feat_names) != len(feat_idx):
                raise ValueError('Feature names size do not match the x input feature size')

        # save into data frames
        self.importance = pd.DataFrame({'feat_idx':feat_idx,
                                        'feature':self.feat_names.iloc[feat_idx] if self.feat_names is not None else None,
                                        'imp_val':feat_imp})


#--------------------------------------------------------------------
# Feature selection methods

class selectMixedDT(_featSelect_base):
    # univariate feature selection with mixed data type, based on statistical test
    # for y as categorical
    #   numeric features are selected based on ANOVA
    #   categorical features are selected based on chi-squared
    # for y as numeric
    #   numeric features are selected based on Pearson's correlation (regression)
    #   categorical features are selected based on ANOVA
    # results sorted based on qval in ascending order

    def __init__(self, alpha=0.05):
        # alpha is the corected p-value threshold (correction as BH FDR)
        super().__init__()

        self.alpha = alpha

    def fit(self, x, y, feat_idx_numeric, feat_idx_categorical, y_categorical, feat_names=None):
        # check length
        self.feat_names = feat_names

        if self.feat_names is not None:
            if len(self.feat_names) != x.shape[1]:
                raise ValueError('Feature names size do not match the x input feature size')

        # feature selection scoring calculations
        feat_labels_num = self.feat_names.iloc[feat_idx_numeric] if self.feat_names is not None else None
        feat_labels_cat = self.feat_names.iloc[feat_idx_categorical] if self.feat_names is not None else None
        x_num = x[:,feat_idx_numeric]
        x_cat = x[:,feat_idx_categorical]

        # univariate feature selection
        if y_categorical:
            sf_cat = SelectKBest(chi2, k=5).fit(x_cat, y) #for categorical y, categorical x; chi-squared
            sf_num = SelectKBest(f_classif, k=5).fit(x_num, y) #for categorical y, numeric x; ANOVA
        else:
            sf_cat = SelectKBest(f_classif, k=5).fit(x_cat, y) #for numeric y, categorical x; ANOVA
            sf_num = SelectKBest(f_regression, k=5).fit(x_num, y) #for numeric y, numeric x; linear regression

        # get the p-value, and perform multiple hypothesis correction
        df1 = pd.DataFrame({'feat_idx': feat_idx_numeric,
                            'feature':feat_labels_num,
                            'pval': sf_num.pvalues_})

        df2 = pd.DataFrame({'feat_idx': feat_idx_categorical,
                            'feature':feat_labels_cat,
                            'pval': sf_cat.pvalues_})

        df_pvals = pd.concat([df1,df2], axis=0)
        df_pvals['qval'] = np.nan
        df_pvals['reject'] = np.nan
        pval_notnan = ~df_pvals.pval.isnull()

        reject, pval_cor =  multipletests(df_pvals.pval[pval_notnan], alpha=self.alpha, method='fdr_bh')[:2]
        df_pvals.loc[pval_notnan, 'qval'] = pval_cor
        df_pvals.loc[pval_notnan, 'reject'] = reject

        # store the results in the importance data frame
        df_pvals.sort_values('qval', ascending=True, inplace=True)
        self.importance = df_pvals
        self.importance['imp_val'] = df_pvals.qval

        self.importance_sel = self.importance.loc[self.importance.reject==True,:]


class selectUnivariate(_featSelect_base):
    # univariate feature selection, based on univariate model and its performance metric
    # results sorted based on score_test in descending order

    def __init__(self, dm_model, threshold=0, sort=False):
        super().__init__()

        self.dm_model = dm_model
        self.threshold = threshold
        self.sort = sort

    def fit(self, x_train, y_train, x_test, y_test, feat_names=None, target_name=None):
        self.feat_names = feat_names

        # check length
        if x_train.shape[1] != x_test.shape[1]:
            raise ValueError('Feature size for train and test sets do not match')

        if self.feat_names is not None:
            if(len(self.feat_names) != x_train.shape[1]):
                raise ValueError('Feature names size do not match the x input feature size and/or feature index size')

        # run model for each feature
        df_res_sel = pd.DataFrame()
        for idx in range(0, x_train.shape[1]):
            x_tr, x_te = self.transform_set(x_train, x_test, feat_idx=idx)
            self.dm_model.fit(x_tr, y_train, x_te, y_test)
            df_res_sp = self.dm_model.evaluate({'train': {'x':x_tr, 'y':y_train},
                                                'test': {'x':x_te, 'y':y_test}},
                                                'univariate', self.feat_names.iloc[idx], target_name)
            df_res_sp['feat_idx'] = idx
            df_res_sp['feat_id'] = self.feat_names.index[idx]

            df_res_sel = df_res_sel.append(df_res_sp, sort=False)
        df_res_sel.set_index('feat_id',inplace=True,drop=True)
        if self.sort:
            df_res_sel.sort_values('score_test', ascending=False, inplace=True)

        self.importance = df_res_sel
        self.importance_sel = df_res_sel.loc[df_res_sel.score_test>self.threshold,:]


class selectKFeat(_selectImpFeat):
    def __init__(self, dm_model, k=10, feat_names=None):
        # get the n most important features, based on feature importance
        super().__init__(dm_model, feat_names)

        self.importance_sel = self.importance.iloc[0:k,:].copy()


class selectQuantile(_selectImpFeat):
    def __init__(self, dm_model, threshold=None, feat_names=None):
        # get the top quantile of features, based on feature importance
        super().__init__(dm_model, feat_names)

        df = self.importance.copy()
        df.sort_values('imp_val', ascending=False, inplace=True)
        df = df.loc[df.imp_val >= df.imp_val.quantile(q=threshold),:] #keep the top quartile
        df = df.loc[df.imp_val > 0, :] #remove any negative or zero importance

        self.importance_sel = df

######################################################################
# Model pipelines
######################################################################

def model_infer_iter(data, dm_model, feat_labels, target_name, df_res, y_categorical, data_null, perm=100):
    # iterative inference

    x_train, y_train = data['train'].values()
    x_test, y_test = data['test'].values()

    #-------
    # full model
    dm_model.fit(x_train, y_train, x_test, y_test)
    df_res_sp = dm_model.evaluate(data, 'all', 'all', target_name, data_null, perm)
    df_res = df_res.append(df_res_sp, sort=False)

    # round 1
    sf = selectQuantile(dm_model, threshold=0.75, feat_names=feat_labels.name)
    feat_names_sel = sf.importance_sel.feature
    if len(feat_names_sel) < 1: return df_res, None
    x_tr, x_te = sf.transform_set(x_train, x_test)
    dm_model.fit(x_tr, y_train, x_te, y_test)

    # round 2
    sf = selectQuantile(dm_model, threshold=0.75, feat_names=feat_names_sel)
    feat_names_sel = sf.importance_sel.feature
    if len(feat_names_sel) < 1: return df_res, None
    x_tr, x_te = sf.transform_set(x_tr, x_te)
    dm_model.fit(x_tr, y_train, x_te, y_test)

    # round 3
    sf = selectQuantile(dm_model, threshold=0.75, feat_names=feat_names_sel)
    feat_names_sel = sf.importance_sel.feature
    if len(feat_names_sel) < 1: return df_res, None
    x_tr, x_te = sf.transform_set(x_tr, x_te)

    # reduced model
    dm_model.fit(x_tr, y_train, x_te, y_test)

    data['train']['x'] = x_tr
    data['test']['x'] = x_te
    data_null['test']['x'] = x_te
    df_res_sp = dm_model.evaluate(data, 'topfeat', 'topfeat', target_name, data_null, perm)
    df_res = df_res.append(df_res_sp, sort=False)

    return df_res, sf


def model_univariate(data, dm_model, feat_labels, target_name, df_res, y_categorical, data_null, perm=100):
    # based on simple pairwise statistical test
    # y_categorical as True for if y are categorical values

    # approach- univariate as a pre-filter, use machine learning to reprioritize
    # classification: works
    # regression: maybe; works for elastic net

    x_train, y_train = data['train'].values()
    x_test, y_test = data['test'].values()

    #-------
    # full model
    dm_model.fit(x_train, y_train, x_test, y_test)
    df_res_sp = dm_model.evaluate(data, 'all', 'all', target_name, data_null, perm)
    df_res = df_res.append(df_res_sp, sort=False)

    # feature selection - univariate, filter by q-value
    sf = selectMixedDT(alpha=0.05)
    if y_categorical:
        sf.fit(x_train,y_train,
               np.where(feat_labels.source.isin(['RNA-seq','CN']))[0],
               np.where(feat_labels.source.isin(['CERES','Mut','Lineage']))[0],
               y_categorical,
               feat_labels.name)
    else:
        sf.fit(x_train,y_train,
               np.where(feat_labels.source.isin(['CERES','RNA-seq','CN']))[0],
               np.where(feat_labels.source.isin(['Mut','Lineage']))[0],
               y_categorical,
               feat_labels.name)
    x_tr, x_te = sf.transform_set(x_train,x_test)
    feat_names_sel = sf.importance_sel.feature
    if len(feat_names_sel) < 1: return df_res, None

    # reduced model
    dm_model.fit(x_tr, y_train, x_te, y_test)

    data['train']['x'] = x_tr
    data['test']['x'] = x_te
    data_null['test']['x'] = x_te
    df_res_sp = dm_model.evaluate(data, 'topfeat', 'topfeat', target_name, data_null, perm)
    df_res = df_res.append(df_res_sp, sort=False)

    return df_res, sf

def model_infer_iter_ens(data, dm_model, feat_labels, target_name, df_res, y_categorical, data_null, perm=100):
    # iterative inference, ensemble (random forest) methods, with Boruta feature selection
    # works on ensemble methods as boruta requires _feature_importance

    x_train, y_train = data['train'].values()
    x_test, y_test = data['test'].values()

    #-------
    # full model
    dm_model.fit(x_train, y_train, x_test, y_test)
    df_res_sp = dm_model.evaluate(data, 'all', 'all', target_name, data_null, perm)
    df_res = df_res.append(df_res_sp, sort=False)

    # round 1
    sf = selectQuantile(dm_model, threshold=0.75, feat_names=feat_labels.name)
    feat_names_sel = sf.importance_sel.feature
    if len(feat_names_sel) < 1: return df_res, None
    x_tr, x_te = sf.transform_set(x_train, x_test)
    dm_model.fit(x_tr, y_train, x_te, y_test)

    # round 2
    sf = selectQuantile(dm_model, threshold=0.75, feat_names=feat_names_sel)
    feat_names_sel = sf.importance_sel.feature
    if len(feat_names_sel) < 1: return df_res, None
    x_tr, x_te = sf.transform_set(x_tr, x_te)
    dm_model.fit(x_tr, y_train, x_te, y_test)

    # round 3
    sf = selectQuantile(dm_model, threshold=0.75, feat_names=feat_names_sel)
    feat_names_sel = sf.importance_sel.feature
    if len(feat_names_sel) < 1: return df_res, None
    x_tr, x_te = sf.transform_set(x_tr, x_te)

    # boruta feature selection
    dm_model.model.set_params(max_depth=7)
    feat_selector = BorutaPy(dm_model.model, n_estimators='auto', verbose=0)
    feat_selector.fit(x_tr, y_train)

    feat_names_sel = feat_names_sel[feat_selector.support_]
    if len(feat_names_sel) < 1: return df_res, None
    x_tr = feat_selector.transform(x_tr)
    x_te = feat_selector.transform(x_te)
    sf = _featSelect_base()
    sf.importance_sel = pd.DataFrame(feat_names_sel.copy())

    # reduced model
    dm_model.fit(x_tr, y_train, x_te, y_test)

    data['train']['x'] = x_tr
    data['test']['x'] = x_te
    data_null['test']['x'] = x_te
    df_res_sp = dm_model.evaluate(data, 'topfeat', 'topfeat', target_name, data_null, perm)
    df_res = df_res.append(df_res_sp, sort=False)

    return df_res, sf


def model_infer_ens(data, dm_model, feat_labels, target_name, df_res, y_categorical, data_null, perm=100):
    # iterative inference, ensemble (random forest) methods, with Boruta feature selection
    # works on ensemble methods as boruta requires _feature_importance
    # one round instead of three rounds of quantile feature elimination

    x_train, y_train = data['train'].values()
    x_test, y_test = data['test'].values()

    # -------
    # full model
    dm_model.fit(x_train, y_train, x_test, y_test)
    df_res_sp = dm_model.evaluate(data, 'all', 'all', target_name, data_null, perm)
    df_res = df_res.append(df_res_sp, sort=False)

    # a dummy selection just to get the feat_names_sel structure
    sf = selectQuantile(dm_model, threshold=0, feat_names=feat_labels.name)
    feat_names_sel = sf.importance_sel.feature
    if len(feat_names_sel) < 1: return df_res, None
    x_tr, x_te = sf.transform_set(x_train, x_test)
    dm_model.fit(x_tr, y_train, x_te, y_test)

    # boruta feature selection
    dm_model.model.set_params(max_depth=7)
    feat_selector = BorutaPy(dm_model.model, n_estimators='auto', verbose=0)
    feat_selector.fit(x_tr, y_train)

    feat_names_sel = feat_names_sel[feat_selector.support_]
    if len(feat_names_sel) < 1: return df_res, None
    x_tr = feat_selector.transform(x_tr)
    x_te = feat_selector.transform(x_te)
    sf = _featSelect_base()
    sf.importance_sel = pd.DataFrame(feat_names_sel.copy())

    # reduced model
    dm_model.fit(x_tr, y_train, x_te, y_test)

    data['train']['x'] = x_tr
    data['test']['x'] = x_te
    data_null['test']['x'] = x_te
    df_res_sp = dm_model.evaluate(data, 'topfeat', 'topfeat', target_name, data_null, perm)
    df_res = df_res.append(df_res_sp, sort=False)

    return df_res, sf

def model_infer(data, dm_model, feat_labels, target_name, df_res, y_categorical, data_null, perm=100):
    # simple inference

    x_train, y_train = data['train'].values()
    x_test, y_test = data['test'].values()

    #-------
    # full model
    dm_model.fit(x_train, y_train, x_test, y_test)
    df_res_sp = dm_model.evaluate(data, 'all', 'all', target_name, data_null, perm)
    df_res = df_res.append(df_res_sp, sort=False)

    # quantile
    sf = selectQuantile(dm_model, threshold=0.75, feat_names=feat_labels.name)
    feat_names_sel = sf.importance_sel.feature
    if len(feat_names_sel) < 1: return df_res, None
    x_tr, x_te = sf.transform_set(x_train, x_test)

    # reduced model
    dm_model.fit(x_tr, y_train, x_te, y_test)
    data['train']['x'] = x_tr
    data['test']['x'] = x_te
    data_null['test']['x'] = x_te
    df_res_sp = dm_model.evaluate(data, 'topfeat', 'topfeat', target_name, data_null, perm)
    df_res = df_res.append(df_res_sp, sort=False)

    return df_res, sf

