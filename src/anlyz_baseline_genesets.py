
#analyze gene sets

def parseGenesets(fname):
    genesets = dict()
    f = open(fname)
    for x in f:
        gs = f.readline()
        gs_name = re.sub('\\t\\t.*\\n','',gs)
        genes = re.sub('.*\\t\\t','',gs).replace('\t\n','').split(sep='\t')
        genes = np.hstack(genes)
        genesets[gs_name] = genes
    f.close()
    
    return genesets

genesets = parseGenesets('../datasets/enrichr/KEGG_2019_Human.txt')
gps_len = pd.Series(genesets).apply(len)
gps_len.describe()

#count    154.000000
#mean      90.980519
#std       71.910663
#min        5.000000
#25%       41.000000
#50%       73.000000
#75%      124.000000
#max      530.000000


genesets = parseGenesets('../datasets/enrichr/Reactome_2016.txt')
gps_len = pd.Series(genesets).apply(len)
gps_len.describe()

#count     765.000000
#mean       61.874510
#std       120.476301
#min         5.000000
#25%        12.000000
#50%        25.000000
#75%        60.000000
#max      1631.000000

genesets = parseGenesets('../datasets/enrichr/Panther_2016.txt')
gps_len = pd.Series(genesets).apply(len)
gps_len.describe()

#count     56.0000
#mean      45.7500
#std       52.7151
#min        5.0000
#25%       10.7500
#50%       28.0000
#75%       51.5000
#max      278.0000
