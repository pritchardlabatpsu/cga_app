### Run models
Primary files for parsing the datasets and building the models.

| Name | Description | 
| :--- | :--- |
| infer_rdmodel.py | Run and build models |

### Analyze results pipeline
Workflow on analyzing the results after building the models. The files are run sequentially.

| Name | Description | 
| :--- | :--- |
| anlyz_rdmodel.py | analyze model results | 
| anlyz_rdmodel_filter.py | filter and reanalyze the results |
| derive_genesets.py | Derive gene sets | 
| anlyz_cytoscape.py | Analyze the outputs from cytoscape |

### Independent analyses
Individual analyses run separately, some as exploratory analyses on the input datasets, other as additional analyses that compares across model results.

| Name | Description | 
| :--- | :--- |
| anlyz_baseline.py | Baseline analyses of the datasets |
| anlyz_baseline_genesets.py | Baseline analyses of gene sets |
| compr_modresults.py | Compare model results, overlap of features, etc |
