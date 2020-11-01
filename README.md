# CERES inference

This project aims to make predictions of CERES based on functional and genomic features. 

The file structure is as follows,
```
.
├── _data               # directory where raw data is stored
├── _notebooks          # jupyter/Rmd notebooks for analyses
├── _out                # outputs of analyses
├── _src                # source codes
│   ├── _ceres_infer    # source for ceres_infer package
│   ├── _cwl            # CWL files for deriving lossy gene sets
│   ├── figures.R       # R scripts for generating figures in manuscript
│   ├── figures.py      # python scripts for generating figures in manuscript
├── README.md
├── environment.yml     # conda environment
├── requirements.txt    # package requirements
├── setup.py            # build script for ceres_infer package
```

## Installation
After cloning the repository, run the following commands to set up a new virtual environment and install the required packages.

```bash
conda env create -f environment.yml
conda activate cnp
pip install -Ue . # installs the ceres_infer package
```

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

Codes for the generation of all figures in manuscript are in `/src/figures.py` and `/src/figures.R`.
