setwd('./') #this is the root directory of project
library(gprofiler2)
library(ggplot2)
library(gridExtra)

# output directory
dir_out <- './manuscript/figures/'
dir_out_srcdata <- paste0(dir_out, 'source_data')

######################################################################
# Figure 1
######################################################################
dir_in <- './out/20.0817 proc_data_baseline/distr_permutations/'  # input directory

plot_corr_sd <- function(df, colorName){
  # plots correlation versus SD
  p <- ggplot(df, aes(CERES_SD,MaxCorr))
  p <- p + geom_point(aes(size=CERES_Range), color=colorName, alpha=0.2) + theme_bw() +
    theme(legend.position="none") + geom_line(aes(CERES_SD,pred), size=1.5, color="red") +
    labs(x="CERES SD", y="CERES Max Corr")
  return(p)
}

process_corrmatrix <- function(sum_data){
  sum_data_ess <- sum_data[sum_data$Essentiality=="common_essential",]
  sum_data_seless <- sum_data[sum_data$Essentiality=="selective_essential",]
  sum_data_non <- sum_data[sum_data$Essentiality=="common_nonessential",]
  lmmod <- lm(MaxCorr~CERES_SD,sum_data_seless)
  sum_data_seless$pred <- predict.lm(lmmod)
  lmmod2 <- lm(MaxCorr~CERES_SD,sum_data_ess)
  sum_data_ess$pred <- predict.lm(lmmod2)
  lmmod3 <- lm(MaxCorr~CERES_SD,sum_data_non)
  sum_data_non$pred <- predict.lm(lmmod3)

  pe <- plot_corr_sd(sum_data_ess, "purple4")
  pse <- plot_corr_sd(sum_data_seless, "magenta3")
  pne <- plot_corr_sd(sum_data_non, "mediumpurple3")

  return(list(pe, pse, pne))
}

# -- Figures of correlations/SD --
# -- Fig 1 B --
bootpval <- read.csv(paste0(dir_in, "bootpvals_combined.csv"))

# correlations without permutation
sum_data <- read.csv(paste0(dir_in, "sumdata1.csv"), header=T, stringsAsFactors = F)
r <- process_corrmatrix(sum_data); pe <- r[[1]]; pse <- r[[2]]; pne <- r[[3]]

df <- sum_data[c('CERES_SD', 'MaxCorr', 'Essentiality')]
write.csv(df, paste0(dir_out_srcdata, '/fig_1b_no_permutation.csv'), row.names = FALSE)

# correlations with permutation, one example
sum_data <- read.csv(paste0(dir_in, "sumdata2.csv"),header=T, stringsAsFactors =F)
r <- process_corrmatrix(sum_data); pe2 <- r[[1]]; pse2 <- r[[2]]; pne2 <- r[[3]]

df <- sum_data[c('CERES_SD', 'MaxCorr', 'Essentiality')]
write.csv(df, paste0(dir_out_srcdata, '/fig_1b_with_permutation.csv'), row.names = FALSE)

## plot histogram of linear model pvalues for ceres_sd
gen_hist <- function(bootpval, ess_class, colorName){
  p <- ggplot(bootpval,aes(log10(.data[[ess_class]])))
  p <- p + geom_histogram(bins=20,fill=colorName) +
    theme_bw() + scale_x_continuous('log10 [p-values of linear fit Max Corr vs SD]') + labs( y='Count')

  return(p)
}

nonesshist <- gen_hist(bootpval, "noness", "mediumpurple3")
selesshist <- gen_hist(bootpval, "seless", "magenta3")
esshist <- gen_hist(bootpval, "ess", "purple4")

lay <- rbind(c(1,1,2,2,3,3,3),
             c(4,4,5,5,6,6,6),
             c(7,7,8,8,9,9,9))
gx <- grid.arrange(pe,pe2,esshist,pse,pse2,selesshist,pne,pne2,nonesshist,ncol=5,layout_matrix=lay)
ggsave(
  paste0(dir_out, "fig1_SD_maxcorr.png"), plot = gx, device = NULL, path = NULL, scale = 1,
  width = 10, height = 5, units = c("in"), dpi = 300, limitsize = F,
)

df <- bootpval[c('ess', 'seless', 'noness')]
colnames(df) <- c('Common essential', 'Selective essential',  'Non-essential')
write.csv(df, paste0(dir_out_srcdata, '/fig_1b_pval.csv'), row.names = FALSE)

# -- Figures of distributions of select genes --
# -- Fig 1 A --
ceresdata <- read.csv(paste0(dir_in, "/ceres_processed.csv"), stringsAsFactors = F, header=T)

plotGene <- function(geneName, colorName, x_breaks, x_lim, binsize=0.01){
  p <- ggplot(ceresdata,aes(x=.data[[geneName]]))
  p <- p + geom_histogram(binwidth=binsize, fill=colorName) +
    scale_x_continuous(breaks = x_breaks, lim = x_lim) +
    scale_y_continuous(breaks = seq(0, 15, 5), lim = c(0, 15), "Count") + theme_bw() +
    labs(x=geneName, y="Count")
  return(p)
}

plotGeneCorr <- function(geneName1, geneName2, colorName){
  p <- ggplot(ceresdata,aes(x=.data[[geneName1]], y=.data[[geneName2]]))
  p <- p + geom_point(size=4,colour=colorName,alpha=0.2) + theme_bw() +
    labs(x=geneName1, y=geneName2)
}

pTK1 <- plotGene("TAOK1", "mediumpurple3", seq(-1, 2, 1), c(-1, 2))
pm4k4 <- plotGene("MAP4K4", "mediumpurple3", seq(-1, 2, 1), c(-1, 2))
ptmcorr <- plotGeneCorr("TAOK1", "MAP4K4", "mediumpurple3")

pm23 <- plotGene("MED23", "magenta3", seq(-2, 2, 1), c(-2, 1))
pm24 <- plotGene("MED24", "magenta3", seq(-2, 2, 1), c(-2, 1))
pm234 <- plotGeneCorr("MED23", "MED24", "magenta3")

pr6 <- plotGene("RAB6A", "purple4", seq(-3, 0, 1), c(-3, 0))
prc1 <- plotGene("RIC1", "purple4", seq(-3, 0, 1), c(-3, 0))
pr6rc1 <- plotGeneCorr("RAB6A", "RIC1", "purple4")

gx <- grid.arrange(pr6,prc1,pr6rc1,pm23,pm24,pm234,pTK1,pm4k4,ptmcorr,ncol=3)
ggsave(
  paste0(dir_out, "fig1_ceres_distr_genes.png"), plot = gx, device = NULL, path = NULL, scale = 1,
  width = 10, height = 5, units = c("in"), dpi = 300, limitsize = F,
)

df <- ceresdata[c('TAOK1', 'MAP4K4', 'MED23', 'MED24', 'RAB6A', 'RIC1')]
write.csv(df, paste0(dir_out_srcdata, '/fig_1a.csv'), row.names = FALSE)

# -----------------------------------------------------
# -- figure 1 supplemental --

# read in data
dir_in <- './out/20.0216 feat/reg_rf_boruta/anlyz_filtered/'
df.stats <- read.csv(sprintf('%s/%s', dir_in, 'agg_summary_filtered.csv'), header=TRUE)
df.aggRes = read.csv(sprintf('%s/%s', dir_in, 'feat_summary_varExp_filtered.csv'), header=TRUE)

dir.lx = './out/19.1013 tight cluster/'
df.lx = read.csv(sprintf('%s/%s', dir.lx,'landmarks_n200_k200.csv'))

# Supp Fig S5 A
pdf(sprintf("%s/fig1supp_smooth_compr_score_scatter_q3_q4.pdf", dir_out))
smoothScatter(df.stats$score_rd10, df.stats$p19q4_score_rd10,
              xlab="Score (Q3)", ylab="Score (Q4)",
              xlim=c(0,1), cex.axis=1.5, cex.lab=1.5)
lines(c(0,1), c(0,1), lty=2, type='l')
dev.off()

df <- df.stats[, c('score_rd10', 'p19q4_score_rd10')]
colnames(df) <- c('Score (P19Q3)', 'Score (P19Q4)')
write.csv(df, paste0(dir_out_srcdata, '/fig_S5a.csv'), row.names = FALSE)

# Supp Fig S5 B
pdf(sprintf("%s/fig1supp_smooth_corr_recall.pdf", dir_out))
smoothScatter(df.stats$corr_rd10, df.stats$recall_rd10,
              xlab="Correlation", ylab="Recall",
              xlim=c(-1,1), ylim=c(0,1), cex.axis=1.5, cex.lab=1.5)
lines(-1:1, c(0.95,0.95,0.95), lty=2, type='l')
dev.off()

df <- df.stats[, c('corr_rd10', 'recall_rd10')]
colnames(df) <- c('Correlation (P19Q3)', 'Recall (P19Q3)')
write.csv(df, paste0(dir_out_srcdata, '/fig_S5b.csv'), row.names = FALSE)

# Supp Fig S5 C
pdf(sprintf("%s/fig1supp_smooth_corr_recall_q4.pdf", dir_out))
smoothScatter(df.stats$p19q4_corr_rd10, df.stats$p19q4_recall_rd10,
              xlab="Correlation", ylab="Recall",
              xlim=c(-1,1), ylim=c(0,1), cex.axis=1.5, cex.lab=1.5)
lines(-1:1, c(0.95,0.95,0.95), lty=2, type='l')
dev.off()

df <- df.stats[, c('p19q4_corr_rd10', 'p19q4_recall_rd10')]
colnames(df) <- c('Correlation (P19Q4)', 'Recall (P19Q4)')
write.csv(df, paste0(dir_out_srcdata, '/fig_S5c.csv'), row.names = FALSE)

######################################################################
# GProfiler
######################################################################
set_base_url("http://biit.cs.ut.ee/gprofiler_archive3/e100_eg47_p14")
dir_out_gprofiler = './manuscript/figures_manual/gprofiler/raw'
if (!file.exists(dir_out_gprofiler)){
  dir.create(dir_out_gprofiler, recursive=TRUE)
}

analyze_gprofiler <- function(gostres, prefix){
  prefix <- paste0(dir_out_gprofiler, '/', prefix)

  # save results to csv
  write.csv(apply(gostres$result, 2, as.character), paste0(prefix, '_gores.csv'), row.names = FALSE)

  # get GO:BP dataframe
  p = gostplot(gostres, capped = FALSE, interactive = FALSE)

  # filter dataframe of only ordered p-values and GO:BP, GO:CC, GO:MF individual dataframes
  gostres.sig = gostres$result[gostres$result$p_value<5e-2,]
  gostres.sig = gostres.sig[order(gostres.sig$p_value),]
  res.gobp = gostres.sig[grep('GO:BP',gostres.sig$source),]
  res.gocc = gostres.sig[grep('GO:CC',gostres.sig$source),]
  res.gomf = gostres.sig[grep('GO:MF',gostres.sig$source),]

  # tables for top 5 GO:BP, GO:CC, GO:MF terms
  cols = c("source", "term_id", "term_name", "term_size")
  publish_gosttable(gostres, res.gobp[c(1:5),], use_colors = T, show_columns = cols, filename = paste0(prefix, '_bp.pdf'))
  publish_gosttable(gostres, res.gocc[c(1:5),], use_colors = T, show_columns = cols, filename = paste0(prefix, '_cc.pdf'))
  publish_gosttable(gostres, res.gomf[c(1:5),], use_colors = T, show_columns = cols, filename = paste0(prefix, '_mf.pdf'))

  # plot highlighting top5 terms
  p.target = publish_gostplot(p, c(res.gobp$term_id[1:5], res.gocc$term_id[1:5], res.gomf$term_id[1:5]), width = 15, height = 10, filename = paste0(prefix, '.png'))

  # plot, saved as pdf
  p.target.all = publish_gostplot(p, width = 10, height = 6, filename = paste0(prefix, '.pdf'))
}

# ------- Target/predictor -----------
dir_in <- './out/20.0216 feat/reg_rf_boruta/anlyz_filtered/'
df.stats <- read.csv(sprintf('%s/%s', dir_in, 'agg_summary_filtered.csv'), header=TRUE)
div.genes = df.stats[,1]

df.varstats <- read.csv(sprintf('%s/%s', dir_in, 'feat_summary_varExp_filtered.csv'), header=TRUE)
feats = sapply(strsplit(as.character(df.varstats$feature), " "), "[[", 1)
targets = unique(as.character(df.stats$target))
all_genes = unique(c(targets, feats))
feats_only = all_genes[!all_genes %in% targets]

# target
# Fig 1 E
gostres.t = gost(query = div.genes, organism = "hsapiens", ordered_query = FALSE,
                 multi_query = FALSE, significant = TRUE, exclude_iea = FALSE, 
                 measure_underrepresentation = FALSE, evcodes = TRUE, 
                 user_threshold = 0.05, correction_method = "g_SCS", 
                 domain_scope = "annotated", custom_bg = NULL, 
                 numeric_ns = "", sources = c('GO:BP', 'GO:CC', 'GO:MF'))
analyze_gprofiler(gostres.t, 'target')

df <- data.frame(div.genes)
colnames(df) <- 'Target genes'
write.csv(df, paste0(dir_out_srcdata, '/fig_1e.csv'), row.names = FALSE)

# predictor
# Fig 3 C
gostres.p = gost(query = feats_only, organism = "hsapiens", ordered_query = FALSE,
                multi_query = FALSE, significant = TRUE, exclude_iea = FALSE, 
                measure_underrepresentation = FALSE, evcodes = FALSE, 
                user_threshold = 0.05, correction_method = "g_SCS", 
                domain_scope = "annotated", custom_bg = NULL, 
                numeric_ns = "", sources =  c('GO:BP', 'GO:CC', 'GO:MF'), as_short_link = FALSE)
analyze_gprofiler(gostres.p, 'predictor')

df <- data.frame(feats_only)
colnames(df) <- 'Predictor genes'
write.csv(df, paste0(dir_out_srcdata, '/fig_3c.csv'), row.names = FALSE)

# ------- L200 -----------
dir.lx = './out/19.1013 tight cluster/'
df.lx = read.csv(sprintf('%s/%s', dir.lx,'landmarks_n200_k200.csv'))
lx.gene = gsub("\\s*\\([^\\)]+\\)", "", df.lx$landmark)

# Fig 5 E
gostres.lx = gost(query = lx.gene, organism = "hsapiens", ordered_query = FALSE,
                  multi_query = FALSE, significant = T, exclude_iea = FALSE, 
                measure_underrepresentation = FALSE, evcodes = FALSE, 
                user_threshold = 0.05, correction_method = "g_SCS", 
                domain_scope = "annotated", custom_bg = NULL, 
                 numeric_ns = "", sources =  c('GO:BP', 'GO:CC', 'GO:MF'))
analyze_gprofiler(gostres.lx, 'L200')

df <- data.frame(lx.gene)
colnames(df) <- 'L200 genes'
write.csv(df, paste0(dir_out_srcdata, '/fig_5e.csv'), row.names = FALSE)
