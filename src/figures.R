setwd('./') #this is the root directory of project
library(gprofiler2)
library(ggplot2)
library(gridExtra)

# output directory
dir_out <- './manuscript/figures/'

# -----------------------------------------------------
dir_in <- './out/20.0817 proc_data_baseline/distr_permutations/'  # input directory

# -- figure 1 CERES distributions --
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
bootpval <- read.csv(paste0(dir_in, "bootpvals_combined.csv"))

# correlations without permutation
sum_data <- read.csv(paste0(dir_in, "sumdata1.csv"),header=T, stringsAsFactors =F)
r <- process_corrmatrix(sum_data); pe <- r[[1]]; pse <- r[[2]]; pne <- r[[3]]

# correlations with permutation, one example
sum_data <- read.csv(paste0(dir_in, "sumdata2.csv"),header=T, stringsAsFactors =F)
r <- process_corrmatrix(sum_data); pe2 <- r[[1]]; pse2 <- r[[2]]; pne2 <- r[[3]]

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

# -- Figures of distributions of select genes --
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

# -----------------------------------------------------
# -- figure 1 supplemental --

# read in data
dir_in <- './out/20.0216 feat/reg_rf_boruta/anlyz_filtered/'
df.stats <- read.csv(sprintf('%s/%s', dir_in, 'agg_summary_filtered.csv'), header=TRUE)

dir.lx = './out/19.1013 tight cluster/'
df.lx = read.csv(sprintf('%s/%s', dir.lx,'landmarks_n200_k200.csv'))

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
p.f1.cellcycle = publish_gostplot(p.f1, res.f1.gobp[c(1:5),], width = 18, height = 10, filename = sprintf('%s/%s', dir_out,'fig1_gprofiler_cellcycle.png' ))

# Highlight top mitochondrial terms
res.f1.mito = gostres.f1$result[grep('^mitochon',gostres.f1$result$term_name),]
res.f1.mito = res.f1.mito[order(res.f1.mito$p_value), ]
row.names(res.f1.mito) = NULL
p.f1.mito = publish_gostplot(p.f1, res.f1.mito, width = 18, height = 10, filename = sprintf('%s/%s', dir_out,'fig1_gprofiler_mito.png'))

# -- figure 4 supplemental gprofiler --
# read in data
lx.gene = gsub("\\s*\\([^\\)]+\\)", "", df.lx$landmark)

# gprofiler analysis for fig4
gostres.f4 = gost(query = lx.gene, organism = "hsapiens", ordered_query = FALSE,
               multi_query = FALSE, significant = T, exclude_iea = FALSE, 
               measure_underrepresentation = FALSE, evcodes = FALSE, 
               user_threshold = 0.05, correction_method = "g_SCS", 
               domain_scope = "annotated", custom_bg = NULL, 
               numeric_ns = "", sources = NULL)
p.f4 = gostplot(gostres.f4, capped =FALSE, interactive = F)

# Highlight top terms in GO:BP
res.f4.bp = gostres.f4$result[grep('GO:BP',gostres.f4$result$source),]
p.f4.bp = publish_gostplot(p.f4, res.f4.bp[c(1:5),], width = 15, height = 10, filename = sprintf('%s/%s', dir_out,'fig4_gprofiler_GOBP.png'))

# Highlight multiple protein binding related terms in GO:MF
res.f4.mf = gostres.f4$result[grep('GO:MF',gostres.f4$result$source),]
p.f4.mf = publish_gostplot(p.f4, res.f4.mf[c(1:5),], width = 15, height = 10, filename = sprintf('%s/%s', dir_out, 'fig4_gprofiler_GOMF.png'))

# Highlight nucleus and lumen related terms in GO:CC
res.f4.cc = gostres.f4$result[grep('GO:CC',gostres.f4$result$source),]
p.f4.cc = publish_gostplot(p.f4, res.f4.cc[c(1:5),], width = 15, height = 10, filename = sprintf('%s/%s', dir_out, 'fig4_gprofiler_GOCC.png'))

