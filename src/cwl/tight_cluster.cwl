class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
id: tight_cluster
baseCommand:
  - Rscript
  - tight_cluster.R
inputs:
  - id: ceres
    type: File
    inputBinding:
      position: 0
  - 'sbg:toolDefaultValue': '20'
    id: clusters_n
    type: int
    inputBinding:
      position: 1
  - 'sbg:toolDefaultValue': '25'
    id: min_cluster_size
    type: int
    inputBinding:
      position: 2
outputs:
  - id: gene_clusters
    type: File?
    outputBinding:
      glob: clusters.csv
label: tight_cluster
requirements:
  - class: ResourceRequirement
    ramMin: 64000
  - class: DockerRequirement
    dockerPull: zhaob1/r_bio
  - class: InitialWorkDirRequirement
    listing:
      - entryname: tight_cluster.R
        entry: >
          #tight clustering

          library(tightClust)

          library(dplyr)

          library(argparse)


          args <- commandArgs(trailingOnly=TRUE)

          ceres_filename <- args[1] #ceres filename

          clustersN <- as.integer(args[2]) #number of clusters

          min_cluster_size <- as.integer(args[3]) #minimum cluster size


          df.ceres <- read.csv(ceres_filename, row.names=1, header=TRUE)

          df.ceres <- t(df.ceres)


          #test

          #data(tclust.test.data) #rows are genes (300), and columns are samples
          (75)

          #tclust1 <- tight.clust(tclust.test.data$Data, target=1, k.min=25,
          random.seed=12345)


          cluster <- tight.clust(df.ceres, target=clustersN,
          k.min=min_cluster_size, random.seed=12345)

          cluster_info <- data.frame(cluster=seq(1,length(cluster$size)),
          size=cluster$size)

          df_cluster <- data.frame(genes=rownames(cluster$data), 
                                   cluster=cluster$cluster)

          df_cluster <- df_cluster %>% group_by(cluster) %>% summarize(genes =
          paste(genes, collapse = ',')) %>% as.data.frame

          df_cluster <- df_cluster[df_cluster$cluster > 0,]

          df_cluster <- merge(df_cluster, cluster_info, by=c("cluster"))

          write.csv(df_cluster, 'clusters.csv', row.names=FALSE)
        writable: false
