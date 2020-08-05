#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 23:19:20 2020

@author: boyangzhao
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#-----------------------------
# settings
#-----------------------------
outdir = './out/20.0122 test/newMut/'
model_name = 'elasticNet'

outdir_sub = '%s/cytoscape/' % outdir
if(not os.path.exists(outdir_sub)): os.mkdir(outdir_sub)

#-----------------------------
# analyze cytoscape results
#-----------------------------

df = pd.read_csv('%s/analyze_undirected/degree_target.csv' % outdir_sub)
plt.figure()
ax = sns.regplot(df.R2_rd, df.Degree, order=2, scatter_kws={'s':30, 'alpha':0.5})
ax.set(xlabel='R2 (reduced model; test set)', ylabel='Degree')
plt.savefig('%s/deg_undirected/degree_R2_rd.pdf' % outdir_sub)
plt.close()

df = pd.read_csv('%s/deg_directed/degree_target.csv' % outdir_sub)
plt.figure()
ax = sns.regplot(df.R2_rd, df.Indegree, order=2, scatter_kws={'s':30, 'alpha':0.5})
ax.set(xlabel='R2 (reduced model; test set)', ylabel='In-Degree')
plt.savefig('%s/deg_directed/degreeIn_R2_rd.pdf' % outdir_sub)
plt.close()

plt.figure()
ax = sns.regplot(df.R2_rd, df.Outdegree, order=2, scatter_kws={'s':30, 'alpha':0.5})
ax.set(xlabel='R2 (reduced model; test set)', ylabel='Out-Degree')
plt.savefig('%sdeg_directed/degreeOut_R2_rd.pdf' % outdir_sub)
plt.close()


