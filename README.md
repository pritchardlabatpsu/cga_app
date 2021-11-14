# CERES inference

This project aims to make predictions of CERES based on functional and genomic features. 

The file structure is as follows,
```
.
├── data               # where raw data is stored
├── notebooks          # jupyter/Rmd notebooks for analyses
├── out                # outputs of analyses
├── src                # source codes
│   ├── ceres_infer    # source codes for ceres_infer package
│   ├── cwl            # CWL files for deriving lossy gene sets
│   ├── figures.R      # R script for generating figures in manuscript
│   ├── figures.py     # python script for generating figures in manuscript
│   ├── data_source.py # python script consolidate all data source csv's into a single Excel
├── README.md
├── LICENSE            # License info
├── environment.yml    # conda environment
├── requirements.txt   # package requirements
├── setup.py           # build script for ceres_infer package
```

## Installation
After cloning the repository, run the following commands to set up a new virtual environment and install the required packages.

```bash
conda env create -f environment.yml
conda activate cnp
pip install -Ue . # installs the ceres_infer package
```

## Data
The repo here contains all notebooks and source codes needed to run the pipeline, build models, and analyze results. 
The raw data need to be downloaded separately. For DepMap, the 19Q3 and 19Q4 releases were downloaded from the DepMap Portal (https://depmap.org/portal/download/). These were placed into the `data/DepMap/19Q3/` and `data/DepMap/19Q4/` folders.
For gene sets, Panther gene sets were downloaded from Enrichr libaries and paralog gene lists were retrieved from Ensembl gene trees using _biomaRt_ in R. These gene sets were placed into the `data/gene_sets/` folder.

Intermediate and final processed files and models are too big for storage on github. They are available upon request.

## Data preprocessing

Refer to `run01-preprocess_data.ipynb` in notebooks for preprocessing of data.

## Run pipeline

The pipeline for model building and inference can be accessed with the `worfklow` class. Before calling this method, the data need to be preprocessed (see above), and the parameters need to be defined (see below).

```python
from ceres_infer.session import workflow

params = {} # Define all the params here

wf = workflow(params)
wf.run_pipe()
```

Refer to `run02-template.ipynb` in notebooks for templates. Including customizing the pipeline.

## Additional analyses

Additional analyses are in the `notebooks` folder.

## Manuscript

Codes for the generation of all figures in manuscript are in `src/figures.py` and `src/figures.R`. Data source Excel file is generated with `src/data_source.py`.
