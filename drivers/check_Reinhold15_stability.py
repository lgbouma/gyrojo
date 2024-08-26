"""
count to see how Reinhold+2015's stability outliers compare to our flags
"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from astropy.io import fits, ascii
from astropy.table import Table
import os
from os.path import join
from gyrojo.paths import LOCALDIR, PAPERDIR, DATADIR, TABLEDIR, RESULTSDIR
from gyrojo.getters import select_by_quality_bits
from gyrointerp.helpers import prepend_colstr, left_merge
from aesthetic.plot import set_style, savefig
from collections import Counter
from operator import itemgetter
from numpy import array as nparr

# Bouma 2024 gyro ages...
csvpath = join(TABLEDIR, "table_star_gyro_allcols.csv")
bdf = pd.read_csv(csvpath)

hl = fits.open(
    join(DATADIR, "literature", "Reinhold_2015_20934_stars.fits")
)
rdf = Table(hl[1].data).to_pandas()

rdf.KIC = rdf.KIC.astype(str)
bdf.KIC = bdf.KIC.astype(str)

mdf = rdf.merge(bdf, how='inner', on='KIC')

for stype in ['ss', 'vs']:
    sel = (mdf.Flag.str.rstrip() == stype)
    N0 = len(mdf[sel])
    N1 = mdf[sel].flag_is_gyro_applicable.sum()
    print(stype, N0, N1)

print(len(bdf), len(bdf[bdf.flag_is_gyro_applicable]))

import IPython; IPython.embed()
