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
human = useMart("ensembl", dataset = "hsapiens_gene_ensembl")
res = getBM(attributes = c("external_gene_name", "hsapiens_paralog_associated_gene_name"),
            filters = "hgnc_symbol",
            values = 'ADSS',
            mart = human)

### Format result
agg.res = res[!duplicated(res$external_gene_name),]
agg.res[,"hsapiens_paralog_associated_gene_name"] = aggregate(hsapiens_paralog_associated_gene_name~external_gene_name, data=res, toString)[,2]
agg.res[,"hsapiens_paralog_associated_gene_name"] = gsub(",", "\t", agg.res$hsapiens_paralog_associated_gene_name)
agg.res[,"hsapiens_paralog_associated_gene_name"] = gsub("^\t ", "", agg.res$hsapiens_paralog_associated_gene_name)

### Write output
write.table(agg.res, 'paralog.txt', append = FALSE, sep = "\t", dec = "\t",
            row.names = FALSE, col.names = FALSE, quote = FALSE)