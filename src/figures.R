setwd('/Users/boyangzhao/Dropbox/Industry/Quantalarity/client Penn/proj_ceres/')
# output directory
dir_out <- './manuscript/figures/'

# read in data
dir_in <- './out/20.0216 feat/reg_rf_boruta/anlyz_filtered/'
df.stats <- read.csv(sprintf('%s/%s', dir_in, 'agg_summary_filtered.csv'), header=TRUE)

# -- figure 1 --
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
