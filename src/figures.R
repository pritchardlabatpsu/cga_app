setwd('./') #this is the root directory of project

# output directory
dir_out <- './manuscript/figures/'

# read in data
dir_in <- './out/20.0216 feat/reg_rf_boruta/anlyz_filtered/'
df.stats <- read.csv(sprintf('%s/%s', dir_in, 'agg_summary_filtered.csv'), header=TRUE)

# -- figure 1 supplemental --
pdf(sprintf("%s/fig1supp_smooth_compr_score_scatter_q3_q4.pdf", dir_out))
smoothScatter(df.stats$score_rd10, df.stats$p19q4_score_rd10,
              xlab="Score (Q3)", ylab="Score (Q4)",
              xlim=c(0,1), cex.axis=1.5, cex.lab=1.5)
lines(c(0,1), c(0,1), lty=2, type='l')
dev.off()

pdf(sprintf("%s/fig1supp_smooth_corr_recall_q4.pdf", dir_out))
smoothScatter(df.stats$p19q4_corr_rd10, df.stats$p19q4_recall_rd10,
              xlab="Correlation", ylab="Recall",
              xlim=c(-1,1), ylim=c(0,1), cex.axis=1.5, cex.lab=1.5)
lines(-1:1, c(0.95,0.95,0.95), lty=2, type='l')
dev.off()

pdf(sprintf("%s/fig1supp_smooth_corr_recall.pdf", dir_out))
smoothScatter(df.stats$corr_rd10, df.stats$recall_rd10,
              xlab="Correlation", ylab="Recall",
              xlim=c(-1,1), ylim=c(0,1), cex.axis=1.5, cex.lab=1.5)
lines(-1:1, c(0.95,0.95,0.95), lty=2, type='l')
dev.off()

# -- figure 1 supplemental gprofiler --
library(gprofiler2)
# read in data
div.genes = df.stats[,1]
# gprofiler analysis for fig1
gostres.f1 = gost(query = div.genes, organism = "hsapiens", ordered_query = FALSE,
               multi_query = FALSE, significant = TRUE, exclude_iea = FALSE, 
               measure_underrepresentation = FALSE, evcodes = FALSE, 
               user_threshold = 0.05, correction_method = "g_SCS", 
               domain_scope = "annotated", custom_bg = NULL, 
               numeric_ns = "", sources = NULL)
p.f1 = gostplot(gostres.f1, capped =FALSE, interactive = F)
# Highlight top cell cycle related terms in GO:BP
res.f1.gobp = gostres.f1$result[grep('GO:BP',gostres.f1$result$source),]
p.f1.cellcycle = publish_gostplot(p.f1, res.f1.gobp[c(1:5),], width = 15, height = 10, filename = sprintf('%s/%s', dir_out,'f1supp_gprofiler_cellcycle.png' ))
# Highlight top mitochondrial terms
res.f1.mito = gostres.f1$result[grep('^mitochon',gostres.f1$result$term_name),]
res.f1.mito = res.f1.mito[order(res.f1.mito$p_value), ]
row.names(res.f1.mito) = NULL
p.f1.mito = publish_gostplot(p.f1, res.f1.mito,width = 15, height = 10, filename = sprintf('%s/%s', dir_out,'f1supp_gprofiler_mito.png'))

# -- figure 4 supplemental gprofiler --
# read in data
dir.lx = './out/19.1013 tight cluster/'
df.lx = read.csv(sprintf('%s/%s', dir.lx,'landmarks_n100_k100.csv'))
lx.gene = gsub("\\s*\\([^\\)]+\\)", "", df.lx$landmark)
# gprofiler analysis for fig4
gostres.f4 = gost(query = lx.gene, organism = "hsapiens", ordered_query = FALSE,
               multi_query = FALSE, significant = T, exclude_iea = FALSE, 
               measure_underrepresentation = FALSE, evcodes = FALSE, 
               user_threshold = 0.05, correction_method = "g_SCS", 
               domain_scope = "annotated", custom_bg = NULL, 
               numeric_ns = "", sources = NULL)
p.f4 = gostplot(gostres.f4, capped =FALSE, interactive = F)
# Highlight top cell cycle related terms
res.f4.cellcycle = gostres.f4$result[grep('cycl',gostres.f4$result$term_name),]
p.f4.cellcycle = publish_gostplot(p.f4, res.f4.cellcycle[c(1:5),], width = 15, height = 10, filename = sprintf('%s/%s', dir_out,'f4supp_gprofiler_cellcycle.png'))
# Highlight multiple protein binding related terms in GO:MF
res.f4.mf = gostres.f4$result[grep('GO:MF',gostres.f4$result$source),]
p.f4.mf = publish_gostplot(p.f4, res.f4.mf[c(1:10),],width = 15, height = 10, filename = sprintf('%s/%s', dir_out, 'f4supp_gprofiler_proteinbinding.png'))
# Highlight nucleus and lumen related terms in GOCC
res.f4.cc = gostres.f4$result[grep('GO:CC',gostres.f4$result$source),]
p.f4.cc = publish_gostplot(p.f4, res.f4.cc[c(1:8),],width = 15, height = 10, filename = sprintf('%s/%s', dir_out, 'f4_supp_gprofiler_GOCC.png'))



