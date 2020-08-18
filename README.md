# CERES inference

See workflow.md for more info on the workflow of preprocessing, inference, and analyses of data.

## Installation
After cloning the repository, run the following commands to set up a new virtual environment and install the required packages.

```bash
conda env create -f environment.yml
pip install -U -e .
```

## Data preprocessing

Refer to `run01-preprocess_data.ipynb` in notebooks for preprocessing of data.

## Run pipeline

The pipeline for model building and inference can be accessed with the `worfklow` class. Before calling this method, the data need to be preprocessed (see above), and the parameters need to be defined (see `TODO` below).

```python
from ceres_infer.session import workflow

params = {} # TODO: define all the params here

wf = workflow(params)
wf.run_pipe()
```

## Additional analyses

Additional analyses are in the `notebooks` folder.
