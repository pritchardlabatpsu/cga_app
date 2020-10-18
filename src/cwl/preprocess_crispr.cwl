class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
id: preprocess_crispr
baseCommand:
  - python
  - preprocess_crispr.py
inputs:
  - id: csv_info
    type: File
    inputBinding:
      position: 0
  - id: csv_crispr
    type: File
    inputBinding:
      position: 1
outputs:
  - id: ceres_indexed
    type: File?
    outputBinding:
      glob: df_crispr.csv
label: preprocess_crispr
requirements:
  - class: DockerRequirement
    dockerPull: frolvlad/alpine-python-machinelearning
  - class: InitialWorkDirRequirement
    listing:
      - entryname: preprocess_crispr.py
        entry: |-
          import sys
          import os
          import numpy as np
          import pandas as pd

          fname_info = sys.argv[1]
          fname_crispr = sys.argv[2]
          outdir = './'
          df_info = pd.read_csv(fname_info, header=0)
          df_crispr = pd.read_csv(fname_crispr, header=0)

          #preprocess data
          def reindex(df):
              if(df.index.name == 'DepMap_ID'):
                return df
              #make the first column as the index
              df = df.rename(columns={df.columns[0]: 'DepMap_ID'})
              df.set_index('DepMap_ID', inplace=True)
              return df


          df_crispr = reindex(df_crispr)

          df_crispr.to_csv('%s/df_crispr.csv' % outdir)
        writable: false
