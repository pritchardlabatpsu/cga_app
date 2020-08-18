#analyze CERES baseline results

#----------- analyze methods --------------
genPlotsStats <- function(out_dir, agg_summary_fname){
  df.stats <- read.csv(sprintf('%s/%s',out_dir,agg_summary_fname), row.names=1, header=TRUE)
  
  pdf(sprintf("%s/stats_score_aggRes/compr_recall_scatter_q3_q4_smooth.pdf", out_dir)) 
  smoothScatter(df.stats$recall_rd10, df.stats$p19q4_recall_rd10, 
                xlab="Recall (Q3)", ylab="Recall (Q4)", 
                xlim=c(0,1), ylim=c(0,1), cex.axis=1.5, cex.lab=1.5)
  dev.off()
  
  
  pdf(sprintf("%s/stats_score_aggRes/compr_score_scatter_q3_q4_smooth.pdf", out_dir)) 
  smoothScatter(df.stats$score_rd10, df.stats$p19q4_score_rd10, 
                xlab="Score (Q3)", ylab="Score (Q4)", 
                xlim=c(0,1), cex.axis=1.5, cex.lab=1.5)
  lines(c(0,1), c(0,1), lty=2, type='l')
  dev.off()
  
  
  pdf(sprintf("%s/stats_score_aggRes/score_recall_q4_smooth.pdf", out_dir))
  smoothScatter(df.stats$p19q4_score_rd10, df.stats$p19q4_recall_rd10, 
                xlab="Score", ylab="Recall", 
                xlim=c(0,1), cex.axis=1.5, cex.lab=1.5)
  dev.off()
  
  pdf(sprintf("%s/stats_score_aggRes/score_recall_smooth.pdf", out_dir))
  smoothScatter(df.stats$score_rd10, df.stats$recall_rd10, 
                xlab="Score", ylab="Recall", 
                xlim=c(0,1), cex.axis=1.5, cex.lab=1.5)
  dev.off()
  
  
  pdf(sprintf("%s/stats_score_aggRes/corr_recall_q4_smooth.pdf", out_dir))
  smoothScatter(df.stats$p19q4_corr_rd10, df.stats$p19q4_recall_rd10, 
                xlab="Correlation", ylab="Recall", 
                xlim=c(-1,1), ylim=c(0,1), cex.axis=1.5, cex.lab=1.5)
  lines(-1:1, c(0.95,0.95,0.95), lty=2, type='l')
  dev.off()
  
  pdf(sprintf("%s/stats_score_aggRes/corr_recall_smooth.pdf", out_dir))
  smoothScatter(df.stats$corr_rd10, df.stats$recall_rd10, 
                xlab="Correlation", ylab="Recall", 
                xlim=c(-1,1), ylim=c(0,1), cex.axis=1.5, cex.lab=1.5)
  lines(-1:1, c(0.95,0.95,0.95), lty=2, type='l')
  dev.off()
}


#----------- analyze  --------------
args <- commandArgs(trailingOnly = TRUE)

# setwd('/Users/boyangzhao/Dropbox/Industry/Quantalarity/client Penn/proj_ceres/github/cnp_dev/')
# out_dir <- './out/20.0216 feat/reg_rf_boruta/anlyz/'
# out_dir_filtered <- './out/20.0216 feat/reg_rf_boruta/anlyz_filtered/'

fname_anlyz <- args[1]
fname_anlyz_filtered <- args[2]

print(fname_anlyz)
print(fname_anlyz_filtered)
#genPlotsStats(fname_anlyz, 'agg_summary.csv')
#genPlotsStats(fname_anlyz_filtered, 'agg_summary_filtered.csv')
