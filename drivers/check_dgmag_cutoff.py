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

from cdips.utils.gaiaqueries import given_votable_get_df
votablepath = join(DATADIR, 'interim', 'allkic_nbhr_counts_dG5-result.vot.gz')
gdf = given_votable_get_df(votablepath, assert_equal=None)

bdf['flag_dr3_crowding_dG5'] = (
    bdf.dr3_source_id.isin(gdf.source_id)
)

sel = ( (bdf.flag_is_gyro_applicable) & ~(bdf.flag_dr3_crowding_dG5) )

print(bdf['flag_is_gyro_applicable'].sum())
print(sel.sum())
