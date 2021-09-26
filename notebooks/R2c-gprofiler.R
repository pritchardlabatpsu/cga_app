### This script is the gprofiler analysis -- but for integration into figures.R
rm(list = ls())
setwd('~/cnp_dev/notebooks/')
library(gprofiler2)
set_base_url("http://biit.cs.ut.ee/gprofiler_archive3/e100_eg47_p14")

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
                  numeric_ns = "", sources = sources = c('GO:BP', 'GO:CC', 'GO:MF'))
p.f1 = gostplot(gostres.f1, capped =FALSE, interactive = F)

#order result dataframe by p-values
gostres.f1 = gostres.f1[order(gostres.f1$p_value),]

#get the dataframe of GO:BP, GO:CC, GO:MF individual dataframes
res.f1.gobp = gostres.f1[grep('GO:BP',gostres.f1$source),]
res.f1.gocc = gostres.f1[grep('GO:CC',gostres.f1$source),]
res.f1.gomf = gostres.f1[grep('GO:MF',gostres.f1$source),]

table.f1.gobp = publish_gostplot(p.f1, res.f1.gobp[c(1:5),], width = 18, height = 10, filename = sprintf('%s/%s', dir_out,'fig1_gprofiler_cellcycle.png' ))

### Tables for top 5 GO:BP, GO:CC, GO:MF terms
publish_gosttable(gostres, res.gobp[c(1:5),],use_colors = T,
                  show_columns = c("source", "term_id", "term_name","term_size","intersection_size"),
                  filename = 'target_bp.png')
publish_gosttable(gostres, res.gocc[c(1:5),],use_colors = T,
                  show_columns = c("source", "term_id", "term_name","term_size","intersection_size"),
                  filename = 'target_cc.png')
publish_gosttable(gostres, res.gomf[c(1:5),],use_colors = T,
                  show_columns = c("source", "term_id", "term_name","term_size","intersection_size"),
                  filename = 'target_mf.png')

# Plot highlighting top5 terms
p.target = publish_gostplot(p, c(res.gobp$term_id[1:5],res.gocc$term_id[1:5],res.gomf$term_id[1:5]), width = 15, height = 10,filename = 'target_only.png')