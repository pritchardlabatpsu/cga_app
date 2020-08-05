#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate networks
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import networkx as nx
import community

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

######################################################################
# Parameters
######################################################################
outdir = './out/20.0216 feat/reg_rf_boruta/'

min_gs_size = 4

######################################################################
# Set settings
######################################################################
plt.interactive(False)

outdir_sub = '%s/network/' % outdir
if(not os.path.exists(outdir_sub)): os.mkdir(outdir_sub)

#save settings
f = open('%s/run_settings.txt' % (outdir_sub), 'w')
f.write('Minimum gene set size: %d\n' % min_gs_size)
f.close()

#read in files
varExp_filtered = pd.read_csv('%s/anlyz_filtered/feat_summary_varExp_filtered.csv' % outdir, header=0)

######################################################################
# Derive networks
######################################################################
df = varExp_filtered[['feat_gene','target']]
df.to_csv('%s/varExp_filtered.adjlist' % outdir_sub, sep=' ', header=False, index=False)

#load into networkx
f = open('%s/varExp_filtered.adjlist' % outdir_sub, 'rb')
G=nx.read_adjlist(f)

#derive communities
communities = community.best_partition(G) #based on the Louvain method
nx.set_node_attributes(G, communities, 'modularity')

modularity = {}
for k,v in communities.items():
    if v not in modularity:
        modularity[v] = [k]
    else:
        modularity[v].append(k)

pickle.dump(modularity, open('%s/modularity.pkl' % outdir_sub,'wb'))

#write gene sets file
f = open('%s/gs.txt' % outdir_sub, 'w')
for n in modularity:
    if(len(modularity[n])>min_gs_size):
        f.write('gene_set_%d\t\t%s\n' % (n, '\t'.join(modularity[n])))
f.close()

#merge with varExp_filtered
df_communities = pd.DataFrame.from_dict(communities, orient='index', columns=['Class'])
varExp_filtered = varExp_filtered.merge(df_communities, left_on='feat_gene', right_index=True)
varExp_filtered.to_csv("%s/feat_summary_varExp_filtered_class.csv" % (outdir_sub), index=False)

######################################################################
# Stats
######################################################################
#-----------------------------
# gene set stats

#gene set sizes
gps_len = []
for k,v in modularity.items():
    gps_len.append(len(v))
    
plt.plot()
ax = sns.distplot(gps_len)
ax.set(xlabel='Gene set size', ylabel='Frequency', title='Median: %d' % np.median(gps_len))
plt.savefig('%s/gs_sizes.png' % (outdir_sub))
plt.close()

#write gene sets stats to file
pd.Series(gps_len).describe().to_csv('%s/gs_stats.csv' % (outdir_sub), header=False)


#-----------------------------
# network stats

#properties
f = open('%s/network_stats.txt' % outdir_sub, 'w')
f.write('Total number of nodes: %d\n' % len(G.nodes))
f.write('Total number of edges: %d\n' % len(G.edges))
f.close()

deg = sorted(d for n, d in G.degree())
plt.figure()
ax = sns.distplot(deg)
ax.set(xlabel='Degree', ylabel='Frequency')
plt.savefig('%s/network_degree_dist.png' % (outdir_sub))

#visualize network
pos = nx.spring_layout(G)  #compute layout

cmap = plt.cm.get_cmap('RdYlBu', len(modularity))

plt.figure(figsize=(9, 9))
plt.axis('off')
for k,v in modularity.items():
    if(len(v)>min_gs_size):
        val = np.random.randint(0,len(modularity)-1)
        nx.draw_networkx_nodes(G, pos, node_size=5, nodelist=v, node_color=cmap(val))
        nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='k', style='solid', width=1, edgelist=G.edges(v))
plt.savefig('%s/network.png' % (outdir_sub))
#plt.show(G)

plt.figure(figsize=(9, 9))
plt.axis('off')
for k,v in modularity.items():
    if(len(v)>30):
        val = np.random.randint(0,len(modularity)-1)
        nx.draw_networkx_nodes(G, pos, node_size=5, nodelist=v, node_color=cmap(val))
        nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='k', style='solid', width=1, edgelist=G.edges(v))
plt.savefig('%s/network_zoom.png' % (outdir_sub))
plt.show(G)



