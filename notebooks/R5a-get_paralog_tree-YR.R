#if (!requireNamespace("BiocManager", quietly = TRUE))
#install.packages("BiocManager")
#BiocManager::install("biomaRt")

### Get gene list
dir_in_res = '../out/20.0216 feat/reg_rf_boruta'
dir_in_anlyz = file.path(dir_in_res, 'anlyz_filtered')
f_featsumm = file.path(dir_in_anlyz,'agg_summary_filtered.csv')
df_aggRes = read.csv(f_featsumm)  #aggregated feat summary
div_genes = df_aggRes[,1]

### BioMart get paralogs
library(biomaRt)
human = useMart("ensembl", dataset = "hsapiens_gene_ensembl", host= "grch37.ensembl.org")
res = getBM(attributes = c("external_gene_name", "hsapiens_paralog_associated_gene_name"),
            filters = "hgnc_symbol",
            values = div_genes,
            mart = human)

res.notna = res[!res$hsapiens_paralog_associated_gene_name=="", ]

### Format result
agg.res = res.notna[!duplicated(res.notna$external_gene_name),]
agg.res[,"hsapiens_paralog_associated_gene_name"] = aggregate(hsapiens_paralog_associated_gene_name~external_gene_name, data=res.notna, toString)[,2]
agg.res$com = paste(agg.res$external_gene_name, agg.res$hsapiens_paralog_associated_gene_name)
agg.res$com = gsub(", ", "\t", agg.res$com)
agg.res$com = gsub(" ", "\t", agg.res$com)
agg.res$com = gsub(" $", "", agg.res$com)
agg.res$com = paste0(agg.res$com, "\t")

### Write output
write.table(agg.res$com, '../out/paralog.txt', append = FALSE, sep = '\t\t',
            row.names = TRUE, col.names = FALSE, quote = FALSE)
