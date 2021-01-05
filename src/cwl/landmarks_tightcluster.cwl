class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
id: landmarks_tightcluster
baseCommand:
  - python
  - get_landmarks.py
inputs:
  - id: ceres_indexed
    type: File
    inputBinding:
      position: 2
      shellQuote: false
  - id: gene_clusters
    type: File
    inputBinding:
      position: 1
      shellQuote: false
outputs:
  - id: gene_landmarks
    type: File?
    outputBinding:
      glob: landmarks.csv
label: landmarks_tightcluster
requirements:
  - class: ShellCommandRequirement
  - class: DockerRequirement
    dockerPull: frolvlad/alpine-python-machinelearning
  - class: InitialWorkDirRequirement
    listing:
      - entryname: get_landmarks.py
        entry: >
          import sys

          import os

          import numpy as np

          import pandas as pd


          fname_clusters = sys.argv[1]

          fname_crispr = sys.argv[2]


          df_crispr = pd.read_csv(fname_crispr, header=0, index_col=0)

          df_clusters = pd.read_csv(fname_clusters, header=0, index_col=0)


          #derive stats

          df_crispr_stats = pd.DataFrame({'min':df_crispr.apply(min),
                                          'max':df_crispr.apply(max),
                                          'avg':df_crispr.apply(np.mean),
                                          'std':df_crispr.apply(np.std)
                                          })
          df_crispr_stats['diff'] = df_crispr_stats['max'] -
          df_crispr_stats['min']


          #derive landmarks

          def getLandmark(df):
              #from the gene cluster, pick the one with the highest variation
              df.genes = df.genes.replace('..', ' (')
              df.genes = df.genes.replace('.', ')')
              gp_genes = df.genes.split(',')
              landmark_gene = df_crispr_stats[df_crispr_stats.index.isin(gp_genes)]['std'].idxmax()
              #landmark_gene = gp_genes[0]
              return landmark_gene

          df_clusters['landmark'] = df_clusters.apply(getLandmark, axis=1)


          df_clusters.to_csv('landmarks.csv')
        writable: false
