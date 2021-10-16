### This script is the gprofiler analysis -- but for integration into figures.R
rm(list = ls())
setwd('~/git/cnp_dev/notebooks/')
library(gprofiler2)
set_base_url("http://biit.cs.ut.ee/gprofiler_archive3/e100_eg47_p14")


dir_out <- '../manuscript/figures/'
dir.res.out = '../manuscript/figures_manual/gprofiler/raw/'
# Read in data (as in figures.R)
dir_in <- '../out/20.0216 feat/reg_rf_boruta/anlyz_filtered/'
df.stats <- read.csv(sprintf('%s/%s', dir_in, 'agg_summary_filtered.csv'), header=TRUE)
df.aggRes = read.csv(sprintf('%s/%s', dir_in, 'feat_summary_varExp_filtered.csv'), header=TRUE)

dir.lx = '../out/19.1013 tight cluster/'
df.lx = read.csv(sprintf('%s/%s', dir.lx,'landmarks_n200_k200.csv'))

######################################################################
# gprofiler
######################################################################
# -- figure 1 gprofiler (target genes) --
#read in data
div.genes = df.stats[,1]

#gprofiler analysis for figure1
gostres.f1 = gost(query = div.genes, organism = "hsapiens", ordered_query = FALSE,
                  multi_query = FALSE, significant = TRUE, exclude_iea = FALSE, 
                  measure_underrepresentation = FALSE, evcodes = FALSE, 
                  user_threshold = 0.05, correction_method = "g_SCS", 
                  domain_scope = "annotated", custom_bg = NULL, 
                  numeric_ns = "", sources = c('GO:BP', 'GO:CC', 'GO:MF'))
p.f1 = gostplot(gostres.f1, capped =FALSE, interactive = F)
write.csv(apply(gostres.f1$result,2,as.character),sprintf('%s/%s', dir.res.out,'target_gores.csv' )
          ,row.names = FALSE)

#order result dataframe by p-values, only analyze significant p-values
gostres.f1.sig = gostres.f1$result[gostres.f1$result$p_value<5e-2,]
gostres.f1.sig = gostres.f1.sig[order(gostres.f1.sig$p_value),]

#get the dataframe of GO:BP, GO:CC, GO:MF individual dataframes
res.f1.gobp = gostres.f1.sig[grep('GO:BP',gostres.f1.sig$source),]
res.f1.gocc = gostres.f1.sig[grep('GO:CC',gostres.f1.sig$source),]
res.f1.gomf = gostres.f1.sig[grep('GO:MF',gostres.f1.sig$source),]

#tables for top 5 GO:BP, GO:CC, GO:MF terms
table.f1.gobp = publish_gosttable(gostres.f1, res.f1.gobp[c(1:5),],use_colors = T,
                  show_columns = c("source", "term_id", "term_name","term_size","intersection_size"),
                  filename = sprintf('%s/%s', dir_out,'fig1_gprofiler_bp.pdf' ))
table.f1.gocc = publish_gosttable(gostres.f1, res.f1.gocc[c(1:5),],use_colors = T,
                  show_columns = c("source", "term_id", "term_name","term_size","intersection_size"),
                  filename = sprintf('%s/%s', dir_out,'fig1_gprofiler_cc.pdf' ))
table.f1.gomf = publish_gosttable(gostres.f1, res.f1.gomf[c(1:5),],use_colors = T,
                  show_columns = c("source", "term_id", "term_name","term_size","intersection_size"),
                  filename = sprintf('%s/%s', dir_out,'fig1_gprofiler_mf.pdf' ))

#scatter plot for all 3 features
p.f1.all = publish_gostplot(p.f1, width = 9, height = 9,filename = sprintf('%s/%s', dir_out,'fig1_gprofiler_all.pdf' ))

#-------------------------------------------------------------------------
# -- figure 3 supplemental gprofiler (predictor genes) --
#read in data, get predictor only genes(all predictors - targets)
feats = sapply(strsplit(as.character(df.aggRes$feature), " "), "[[", 1)
targets = unique(as.character(df.aggRes$target))
all_genes = unique(c(targets,feats))
feats_only = all_genes[!all_genes %in% targets]

#gprofiler analysis
gostres.f3 = gost(query = feats_only, organism = "hsapiens", ordered_query = FALSE,
                          multi_query = FALSE, significant = TRUE, exclude_iea = FALSE, 
                          measure_underrepresentation = FALSE, evcodes = FALSE, 
                          user_threshold = 0.05, correction_method = "g_SCS", 
                          domain_scope = "annotated", custom_bg = NULL, 
                          numeric_ns = "", sources =  c('GO:BP', 'GO:CC', 'GO:MF'), as_short_link = FALSE)
p.f3 = gostplot(gostres.f3, capped =FALSE, interactive = F)
write.csv(apply(gostres.f3$result,2,as.character),sprintf('%s/%s', dir.res.out,'predictor_gores.csv' )
          ,row.names = FALSE)


#get significant terms and order by pvalues
gostres.f3.sig = gostres.f3$result[gostres.f3$result$p_value<5e-2,] # Significant terms
gostres.f3.sig = gostres.f3.sig[order(gostres.f3.sig$p_value),]

#get the dataframe for GO:BP, GO:CC, GO:MF 
res.f3.gobp = gostres.f3.sig[grep('GO:BP',gostres.f3.sig$source),]
res.f3.gocc = gostres.f3.sig[grep('GO:CC',gostres.f3.sig$source),]
res.f3.gomf = gostres.f3.sig[grep('GO:MF',gostres.f3.sig$source),]

#tables for top 5 GO:BP, GO:CC, GO:MF terms
table.f3.gobp = publish_gosttable(gostres.f3, res.f3.gobp[c(1:5),],use_colors = T,
                  show_columns = c("source", "term_id", "term_name","term_size","intersection_size"),
                  filename = sprintf('%s/%s', dir_out,'fig3_gprofiler_bp.pdf' ))
table.f3.gocc = publish_gosttable(gostres.f3, res.f3.gocc[c(1:5),],use_colors = T,
                  show_columns = c("source", "term_id", "term_name","term_size","intersection_size"),
                  filename = sprintf('%s/%s', dir_out,'fig3_gprofiler_cc.pdf' ))
table.f3.gomf = publish_gosttable(gostres.f3, res.f3/ginf[c(1:5),],use_colors = T,
                  show_columns = c("source", "term_id", "term_name","term_size","intersection_size"),
                  filename = sprintf('%s/%s', dir_out,'fig3_gprofiler_mf.pdf' ))

#scatter plot for all 3 features
p.f3.all = publish_gostplot(p.f3, width = 9, height = 9, filename = sprintf('%s/%s', dir_out,'fig3_gprofiler_all.pdf' ))


#-------------------------------------------------------------------------
# -- figure 4 supplemental gprofiler (predictor genes) --
#read in data, get predictor only genes(all predictors - targets)
lx.gene = gsub("\\s*\\([^\\)]+\\)", "", df.lx$landmark)

#gprofiler analysis
gostres.f4 = gost(query = lx.gene, organism = "hsapiens", ordered_query = FALSE,
                  multi_query = FALSE, significant = T, exclude_iea = FALSE, 
                  measure_underrepresentation = FALSE, evcodes = FALSE, 
                  user_threshold = 0.05, correction_method = "g_SCS", 
                  domain_scope = "annotated", custom_bg = NULL, 
                  numeric_ns = "", sources =  c('GO:BP', 'GO:CC', 'GO:MF'))
p.f4 = gostplot(gostres.f4, capped =FALSE, interactive = F)
write.csv(apply(gostres.f4$result,2,as.character),sprintf('%s/%s', dir.res.out,'lx200_gores.csv' )
          ,row.names = FALSE)

#get significant terms and order by pvalues
gostres.f4.sig = gostres.f4$result[gostres.f4$result$p_value<5e-2,] # Significant terms
gostres.f4.sig = gostres.f4.sig[order(gostres.f4.sig$p_value),]
# Get the dataframe of only ordered p-values and GO:BP, GO:CC, GO:MF individual dataframes
res.f4.gobp = gostres.f4.sig[grep('GO:BP',gostres.f4.sig$source),]
res.f4.gocc = gostres.f4.sig[grep('GO:CC',gostres.f4.sig$source),]
res.f4.gomf = gostres.f4.sig[grep('GO:MF',gostres.f4.sig$source),]

#tables for top 5 GO:BP, GO:CC, GO:MF terms
table.f4.gobp = publish_gosttable(gostres.f4, res.f4.gobp[c(1:5),], use_colors = T,
                            show_columns = c("source", "term_id", "term_name","term_size","intersection_size"),
                            filename = sprintf('%s/%s', dir_out,'fig4_gprofiler_bp.pdf' ))
table.f4.gomf = publish_gosttable(gostres.f4, res.f4.gomf[c(1:5),], use_colors = T,
                            show_columns = c("source", "term_id", "term_name","term_size","intersection_size"), 
                            filename = sprintf('%s/%s', dir_out,'fig4_gprofiler_mf.pdf' ))
table.f4.gocc = publish_gosttable(gostres.f4, res.f4.gocc[c(1:5),],use_colors = T,
                            show_columns = c("source", "term_id", "term_name","term_size","intersection_size"),
                            filename = sprintf('%s/%s', dir_out,'fig4_gprofiler_cc.pdf' ))
#scatter plot for all 3 features
p.f4.all = publish_gostplot(p.f4, width = 12, height = 9,filename = sprintf('%s/%s', dir_out,'fig4_gprofiler_all.pdf' ))







