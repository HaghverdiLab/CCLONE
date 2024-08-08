import pandas as pd
import numpy as np
from scipy.io import mmread
import os


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def get_meta(meta_file, annotation_file, chrom):

    meta_0 = pd.read_csv(meta_file, index_col=False, header=None,
                         sep="\t", comment="#")
    meta_0.columns = ["chr", "pos", "ID", "ref", "mut", "qual", "filter", "info"]
    meta_0 = meta_0[["chr", "pos", "ref", "mut"]]
    ann_0 = pd.read_csv(annotation_file, sep="\t", index_col=False, header=None, comment="#")
    ann_0.insert(0, 'chr', np.repeat(chrom, ann_0.shape[0]))
    ann_0.columns = ["chr", "pos", "allele", "gene", "feature", "feature_type", "consequence", "AA", "Codons",
                     "Existing_variation", "effect", "REDIdb", "AF", "dbSNP-common"]
    ann_0["pos"] = [int(i.split(":")[1]) for i in ann_0["pos"]]
    ann_0["cons"] = ann_0["gene"] + ":" + ann_0["consequence"]
    # filter
    REDIdb = np.ones(meta_0.shape[0]).astype(bool)
    dbSNP = np.ones(meta_0.shape[0]).astype(bool)
    gene = []
    for i, pos in enumerate(meta_0.pos):
        idx = np.where(ann_0["pos"] == pos)[0]
        REDIdb[i] = np.any(ann_0.iloc[idx]["REDIdb"] != "-")
        dbSNP[i] = np.any(ann_0.iloc[idx]["dbSNP-common"] != "-")
        gene.append(",".join(np.unique(ann_0.iloc[idx]["cons"])))

    meta_0["REDIdb"], meta_0["dbSNP"], meta_0["gene"] = REDIdb, dbSNP, gene

    return meta_0


def get_variant_measurement_data(path,
                                 all_chroms=None,
                                 cell_names=None,
                                 perc=10,
                                 filter_hela=True):
    meta, ann, ALT, REF = None, None, None, None
    all_chroms = ["chr" + str(i) for i in range(1, 23)] if all_chroms is None else all_chroms

    for chrom in all_chroms:
        print(chrom)

        f = open(path + '/' + chrom + '/cellSNP.tag.AD.mtx', 'r')
        ALT_0 = pd.DataFrame(mmread(f).A)

        f = open(path + '/' + chrom + '/cellSNP.tag.DP.mtx', 'r')
        DP = pd.DataFrame(mmread(f).A)

        REF_0 = DP - ALT_0

        meta_0 = get_meta(path + '/' + chrom + '/cellSNP.base.vcf',
                          path + '/' + chrom + '/annotations.tsv',
                          chrom)

        # variants need to be covered in at least 10% of the cells
        sub = np.sum((ALT_0 + REF_0) >= 2, axis=1) > (ALT_0.shape[1] / perc)
        ALT_0, REF_0, meta_0 = ALT_0[sub], REF_0[sub], meta_0[sub]

        ALT = ALT_0 if ALT is None else pd.concat((ALT, ALT_0))
        REF = REF_0 if REF is None else pd.concat((REF, REF_0))
        meta = meta_0 if meta is None else pd.concat((meta, meta_0))

    # reset indices
    REF.index = np.arange(0, REF.shape[0])
    ALT.index = np.arange(0, REF.shape[0])
    meta.index = np.arange(0, REF.shape[0])

    if cell_names is not None:
        REF.columns, ALT.columns = cell_names, cell_names

    # filter vars in repeat regions
    sub = meta.ref != "N"
    if filter_hela:
        int_pos = np.array([int(i) for i in meta.pos.tolist()])
        in_region = (meta.chr == "chr6") & (int_pos > 28510120) & (int_pos < 33480577)
        sub &= ~in_region
    REF, ALT = REF.loc[sub], ALT.loc[sub]
    meta = meta.loc[sub]
    return REF, ALT, meta
