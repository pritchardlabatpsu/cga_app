#analyze CERES baseline results
setwd('/Users/boyangzhao/Dropbox/Industry/Quantalarity/client Penn/proj_ceres/')

out_dir <- './out/bio_8_3/stats/'
df.stats <- read.csv(sprintf('%s/stats_ceres.csv',out_dir), row.names=1, header=TRUE)

#---
#gene set GO terms
#https://bioinformatics-core-shared-training.github.io/cruk-summer-school-2018/RNASeq2018/html/06_Gene_set_testing.nb.html
#https://stephenturner.github.io/deseq-to-fgsea/
#https://github.com/ctlab/fgsea
library(fgsea)
library(tidyverse)
library(ggplot2)
library(dplyr)

#load the pathways into a named list
pathways.hallmark <- gmtPathways("../datasets/GSEA/MSigDB/h.all.v7.0.symbols.gmt")

analyzeGSEA <- function(ranks, anlyz_prefix){
  fgseaRes <- fgsea(pathways=pathways.hallmark, stats=ranks, nperm=1000, minSize=15, maxSize=500)
  
  png(sprintf("%s/gsea_%s_bar_all.png", out_dir,anlyz_prefix))
  par(mar=c(5,5,3,3))
  ggplot(fgseaRes, aes(reorder(pathway, NES), NES)) +
    geom_col(aes(fill=padj<0.05)) +
    coord_flip() +
    labs(x="Pathway", y="Normalized Enrichment Score",
         title="Hallmark pathways NES from GSEA") + 
    theme_minimal()
  dev.off()
  
  topPathwaysUp <- fgseaRes[ES > 0, ][head(order(pval), n=10), pathway]
  topPathwaysDown <- fgseaRes[ES < 0, ][head(order(pval), n=10), pathway]
  topPathways <- c(topPathwaysUp, rev(topPathwaysDown))
  
  png(sprintf("%s/gsea_%s_high.png", out_dir,anlyz_prefix))
  par(mar=c(5,5,3,3))
  plotGseaTable(pathways.hallmark[topPathwaysUp], ranks, fgseaRes, gseaParam = 0.5)
  dev.off()
  
  png(sprintf("%s/gsea_%s_low.png", out_dir,anlyz_prefix))
  par(mar=c(5,5,3,3))
  plotGseaTable(pathways.hallmark[topPathwaysDown], ranks, fgseaRes, gseaParam = 0.5)
  dev.off()
  
  png(sprintf("%s/gsea_%s_all.png", out_dir,anlyz_prefix))
  par(mar=c(5,5,3,3))
  plotGseaTable(pathways.hallmark[topPathways], ranks, fgseaRes, gseaParam = 0.5)
  dev.off()
  
}

#gene set enrichment - sd
ranks <- df.stats$std #ranked list, with higher val higher ranked
names(ranks) <- sub('.\\(.*','',rownames(df.stats))
anlyz_prefix <- 'sd'
analyzeGSEA(ranks, anlyz_prefix)

#gene set enrichment - mean
ranks <- df.stats$avg #ranked list, with higher val higher ranked
names(ranks) <- sub('.\\(.*','',rownames(df.stats))
anlyz_prefix <- 'mean'
analyzeGSEA(ranks, anlyz_prefix)

