class: Workflow
cwlVersion: v1.0
id: derive_tight_landmarks
label: derive_tight_landmarks
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
inputs:
  - id: csv_info
    type: File
    'sbg:x': -422.39886474609375
    'sbg:y': -163.5
  - id: csv_crispr
    type: File
    'sbg:x': -412.39886474609375
    'sbg:y': -17.5
  - id: clusters_n
    type: int
    'sbg:exposed': true
  - id: min_cluster_size
    type: int
    'sbg:exposed': true
outputs:
  - id: ceres_indexed
    outputSource:
      - preprocess_crispr/ceres_indexed
    type: File?
    'sbg:x': -70.4765625
    'sbg:y': -216.5
  - id: gene_landmarks
    outputSource:
      - landmarks_tightcluster/gene_landmarks
    type: File?
    'sbg:x': 273
    'sbg:y': -146
  - id: gene_clusters
    outputSource:
      - tight_cluster/gene_clusters
    type: File?
    'sbg:x': 115
    'sbg:y': -213
steps:
  - id: preprocess_crispr
    in:
      - id: csv_info
        source: csv_info
      - id: csv_crispr
        source: csv_crispr
    out:
      - id: ceres_indexed
    run: ./preprocess_crispr.cwl
    label: preprocess_crispr
    'sbg:x': -263.3984375
    'sbg:y': -90.5
  - id: tight_cluster
    in:
      - id: ceres
        source: preprocess_crispr/ceres_indexed
      - id: clusters_n
        default: 20
        source: clusters_n
      - id: min_cluster_size
        default: 25
        source: min_cluster_size
    out:
      - id: gene_clusters
    run: ./tight_cluster.cwl
    label: tight_cluster
    'sbg:x': -65
    'sbg:y': -67
  - id: landmarks_tightcluster
    in:
      - id: ceres_indexed
        source: preprocess_crispr/ceres_indexed
      - id: gene_clusters
        source: tight_cluster/gene_clusters
    out:
      - id: gene_landmarks
    run: ./landmarks_tightcluster.cwl
    label: landmarks_tightcluster
    'sbg:x': 115
    'sbg:y': -29
requirements: []
