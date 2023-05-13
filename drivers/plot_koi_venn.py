import numpy as np, matplotlib.pyplot as plt, pandas as pd
import os, pickle

from astropy.table import Table
from astropy.io import fits

from agetools.paths import DATADIR, RESULTSDIR, LOCALDIR

# pip install aesthetic
from aesthetic.plot import savefig

from agetools import venn

def plot_venn(outdir):

    df = pd.read_csv(os.path.join(
        DATADIR, "interim", "koi_table_X_GDR3_B20_S19_S21_M14_M15.csv")
    )

    s19_set = set(df[~pd.isnull(df['s19_Prot'])].kepid)
    s21_set = set(df[~pd.isnull(df['s21_Prot'])].kepid)
    m14_set = set(df[~pd.isnull(df['m14_Prot'])].kepid)
    m15_set = set(df[~pd.isnull(df['m15_Prot'])].kepid)

    labels = venn.get_labels([s19_set, s21_set, m14_set, m15_set], fill=['number', 'logic'])
    fig, ax = venn.venn4(labels, names=['S19', 'S21', 'M14', 'M15'])

    outpath = os.path.join(outdir, f'venn.png')
    savefig(fig, outpath)


if __name__ == "__main__":

    outdir = os.path.join(RESULTSDIR, 'koi_venn')
    if not os.path.exists(outdir): os.mkdir(outdir)
    plot_venn(outdir)
