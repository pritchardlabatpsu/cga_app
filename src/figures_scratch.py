# Supp Figs
dir_in_multi = './out/20.0216 feat/reg_rf_boruta/anlyz_filtered/'
dir_in_uni = './out/20.0216 feat/reg_univariate_rf/anlyz_filtered/'
varExp_multi = pd.read_csv(os.path.join(dir_in_multi, 'feat_summary_varExp_filtered.csv'), header=0)
varExp_uni = pd.read_csv(os.path.join(dir_in_uni, 'feat_summary_varExp_filtered.csv'), header=0)

# targets
a = set(varExp_multi.target.unique())
b = set(varExp_uni.target.unique())
plt.figure()
venn2(subsets = (len(a-b), len(b-a), len(a.intersection(b))), set_labels = ('Multivariate', 'Univariate'))
plt.title('Targets')
plt.savefig("%s/fig3supp_venn_targets.pdf" % dir_out)
plt.close()

# features
a = set(varExp_multi.feat_gene.unique())
b = set(varExp_uni.feat_gene.unique())
plt.figure()
venn2(subsets = (len(a-b), len(b-a), len(a.intersection(b))), set_labels = ('Multivariate', 'Univariate'))
plt.title('Features')
plt.savefig("%s/fig3supp_venn_feats.pdf" % dir_out)
plt.close()

# concordance
plt.figure()
ax = sns.violinplot(df_conc_tr.concordance)
ax.set(xlim=[0.4,1.05], xlabel='Concordance', ylabel='Count')
plt.tight_layout()
plt.savefig("%s/fig4_concordance_tr.pdf" % dir_out)
plt.close()

plt.figure()
ax = sns.violinplot(df_conc_te.concordance)
ax.set(xlim=[0.4,1.05], xlabel='Concordance', ylabel='Count')
plt.tight_layout()
plt.savefig("%s/fig4_concordance_te.pdf" % dir_out)
plt.close()