"""
count to see how Simonian+2019's photometric outliers compare to our flags
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
    join(DATADIR, "literature", "Simonian_2019_t1.fits")
)
df0 = Table(hl[1].data).to_pandas()
hl.close()
hl = fits.open(
    join(DATADIR, "literature", "Simonian_2019_t2.fits")
)
df1 = Table(hl[1].data).to_pandas()
hl.close()

sdf = pd.concat((df0, df1))

sdf.KIC = sdf.KIC.astype(str)
bdf.KIC = bdf.KIC.astype(str)

mdf = sdf.merge(bdf, how='left', on='KIC')

for dmag in [-0.5, -0.3]:
    sel = (mdf.dK < dmag)
    N0 = len(mdf[sel])
    N1 = mdf[sel].flag_is_gyro_applicable.sum()
    print(dmag, N0, N1)

print(len(bdf), len(bdf[bdf.flag_is_gyro_applicable]))
