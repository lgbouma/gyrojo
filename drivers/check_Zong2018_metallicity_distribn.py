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

datadir = '/Users/luke/Dropbox/proj/gyrojo/secret_data'

hl = fits.open(
    join(datadir, "Zong_2018_LAMOST_Kepler_2012_to_2017_227870rows.fits")
)
zdf = Table(hl[1].data).to_pandas()

mdf = bdf.merge(zdf, on='KIC', how='inner')
mdf = mdf[mdf.gyro_median <= 4000]

set_style('science')
plt.close("all")
fig, ax = plt.subplots()
bins = np.arange(-3.1, 3.1, 0.1)
ax.hist(mdf.Fe_H, bins=bins)
ax.set_xlabel('Zong+2018 Fe/H')
ax.set_ylabel('Count')
ax.set_xlim([-1.2, 1.2])
outpath = join(RESULTSDIR, "Zong2018_metallicity_distribution",
               "zong2018_hist.png")
savefig(fig, outpath)

print(42*'-')
frac = np.sum((mdf.Fe_H > 0.5) | (mdf.Fe_H < -0.5)) / len(mdf)
print(f'based on {len(mdf)} stars in the xmatch...')
print(f'{frac*100:.2f}% of the sample has |Fe/H|>0.5')

frac = np.sum((mdf.Fe_H > 0.3) | (mdf.Fe_H < -0.3)) / len(mdf)
print(f'based on {len(mdf)} stars in the xmatch...')
print(f'{frac*100:.2f}% of the sample has |Fe/H|>0.3')

smdf = mdf[mdf.flag_is_gyro_applicable]

print(42*'-')
frac = np.sum((smdf.Fe_H > 0.5) | (smdf.Fe_H < -0.5)) / len(smdf)
print(f'based on {len(smdf)} stars in the xmatch w/ gyro applicable...')
print(f'{frac*100:.2f}% of the sample has |Fe/H|>0.5')

frac = np.sum((smdf.Fe_H > 0.3) | (smdf.Fe_H < -0.3)) / len(smdf)
print(f'based on {len(smdf)} stars in the xmatch...')
print(f'{frac*100:.2f}% of the sample has |Fe/H|>0.3')
