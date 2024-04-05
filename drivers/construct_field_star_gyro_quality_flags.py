"""
Build flags for
- subgiants
- photometric binaries
- ruwe outliers
- crowding
- non-single-stars
- "CP/CB candidates"

all in service of the one flag to rule them all:
    "flag_is_gyro_applicable"

(NOTE 2023/06/05: photometric binaries need reconstruction using the Green19
reddening maps)
"""

import pandas as pd, numpy as np
import os
from os.path import join
from glob import glob
from gyrojo.paths import DATADIR, RESULTSDIR, LOCALDIR, TABLEDIR

datestr = '20230529'
datestr = '20240405'

csvpath = join(RESULTSDIR, f"field_gyro_posteriors_{datestr}",
               f"field_gyro_posteriors_{datestr}_gyro_ages_X_GDR3_S19_S21_B20.csv")

df = pd.read_csv(
    csvpath, dtype={'dr3_source_id':str, 'KIC':str, 'kepid':str }
)

# drop 3 stars with nan DR3 source id's
df = df[~(df.dr3_source_id.astype(str) == 'nan')]

df['M_G'] = (
    df['dr3_phot_g_mean_mag'] + 5*np.log10(df['dr3_parallax']/1e3) + 5
)

###########################
# build the quality flags #
###########################

df['flag_ruwe_outlier'] = df['dr3_ruwe'] > 1.4

df['flag_logg'] = df['adopted_logg'] < 4.2

df['flag_not_CP_CB'] = (
    pd.isnull(df.s21_flag1)
    &
    pd.isnull(df.s19_flag1)
)

#################################
# check if in Kepler EB catalog #
#################################

from astropy.io import fits
from astropy.table import Table

fitspath = join(DATADIR, 'literature', 'Kirk_2016_KEBC_2876_rows.fits')
hdul = fits.open(fitspath)
kebc_df = Table(hdul[1].data).to_pandas()
kebc_df['KIC'] = kebc_df['KIC'].astype(str)
df['KIC'] = df['KIC'].astype(str)

df['flag_in_KEBC'] = (
    df.KIC.isin(kebc_df.KIC)
)


###################################################
# get the nonsingle star flag via Gaia DR3 xmatch #
###################################################
from cdips.utils.gaiaqueries import given_source_ids_get_gaia_data
source_ids = np.array(df.dr3_source_id).astype(np.int64)
assert pd.isnull(source_ids).sum() == 0
groupname = f'field_gyro_{datestr}'

gdf = given_source_ids_get_gaia_data(
    source_ids, groupname, n_max=int(6e4), overwrite=False,
    which_columns='g.source_id, g.radial_velocity_error, g.non_single_star',
    gaia_datarelease='gaiadr3'
)

# 0: "single".
# 1,2,3:
#    • bit 1 (least-significant bit) is set to 1 in case of an astrometric binary
#    • bit 2 is set to 1 in case of a spectroscopic binary
#    • bit 3 is set to 1 in case of an eclisping binary
# Counter({0: 52914, 1: 974, 2: 151, 3: 109})
df['dr3_non_single_star'] = np.array(gdf.non_single_star)
df['flag_dr3_non_single_star'] = (df.dr3_non_single_star > 0)

############################################################
# get the neighbor count via Gaia DR3 source catalog query #
############################################################
from cdips.utils.gaiaqueries import given_source_ids_get_neighbor_counts
dGmag = 2.5
sep_arcsec = 4
runid = f'field_gyro_{datestr}_neighbors'
n_max = 60000
count_df, ndf = given_source_ids_get_neighbor_counts(
    source_ids, dGmag, sep_arcsec, runid, n_max=n_max, overwrite=False,
    enforce_all_sourceids_viable=True, gaia_datarelease='gaiadr3'
)

df['nbhr_count'] = np.array(count_df.nbhr_count)
df['flag_nbhr_count'] = df['nbhr_count'] >= 1

#############
# CAMD flag #
#############

#TODO: need CAMD flag constructed via Green19 map
# this is a hacky selection in M_G vs (BP-RP), no extinction.  made in
# "session_{datestr}_phot_single_selection_and_other_fun_subsets.glu"
# and 20240405 is just a copy of 20230529
manual_path = join(
    RESULTSDIR,
    "glue_interactive_viz",
    f"field_gyro_posteriors_{datestr}_GDR3_S19_S21_B20_phot_single.csv"
)
manual_phot_single_df = pd.read_csv(manual_path, dtype={'KIC':str})

df['KIC'] = df.KIC.astype(str)

df['flag_camd_outlier'] = ~(df.KIC.isin(manual_phot_single_df.KIC))

################################
# finally, is gyro applicable? #
################################

df['flag_is_gyro_applicable'] = (
    (~df['flag_logg'])
    &
    (~df['flag_ruwe_outlier'])
    &
    (~df['flag_dr3_non_single_star'])
    &
    (~df['flag_camd_outlier'])
    #&
    #(df['flag_not_CP_CB'])
    &
    (~df['flag_in_KEBC'])
    &
    (df['adopted_Teff'] > 3800)
    &
    (df['adopted_Teff'] < 6200)
)

outcsv = join(
    TABLEDIR,
    f"field_gyro_posteriors_{datestr}_gyro_ages_X_GDR3_S19_S21_B20_with_qualityflags.csv"
)

df.to_csv(outcsv, index=False)
print(f"Wrote {outcsv}")
