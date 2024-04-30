"""
Build flags for...
bit 0: T if adopted_Teff outside of 3800-6200K
bit 1: T if adopted_logg>4.2 (dwarf stars only)
bit 2: T if M_G<3.9 (require Main Sequence < F8V)
bit 3: T if in KEBC (no known EBs)
bit 4: T if Kepler<->Gaia xmatch ang dist > 3 arcsec
bit 5: T if Kepler<->Gaia xmatch yielded multiple stars within ΔG=0.5 mag at <4"
bit 6: T if Gaia DR3 non_single_star
bit 7: T if RUWE>1.4
bit 8: T if nbhr_count >= 1 (crowding, at least one 1/10th Gmag brightness within 4 arcsec)
bit 9: T if above logg/Teff locus (iso age precision, cutting subgiant FGKs & photometric outliers)

plus, not relevant for gyro:

bit 10: T if in CP/CB from Santos [lossy...]
bit 11: T if period was not reported in the Santos19/Santos21 samples
(i.e. it was from the David21 meta-analysis, or Santos+priv comm -- both of which are KOI-only samples)

...all in service of the one flag that can be used to rule them all:
    "flag_is_gyro_applicable"
"""

import pandas as pd, numpy as np
import os
from os.path import join
from glob import glob
from gyrojo.paths import DATADIR, RESULTSDIR, LOCALDIR, TABLEDIR
from numpy import array as nparr


def build_gyro_quality_flag(sample='gyro', datestr='20240430'):

    if sample == 'gyro':
        csvpath = join(
            RESULTSDIR, f"field_gyro_posteriors_{datestr}",
            f"field_gyro_posteriors_{datestr}_gyro_ages_X_GDR3_S19_S21_B20.csv"
        )
    elif sample == 'allKIC':
        # made by get_cleaned_gaiadr3_X_kepler_supplemented_dataframe
        csvpath = join(
            DATADIR, "interim", f"kic_X_dr3_supp.csv"
        )

    df = pd.read_csv(
        csvpath, dtype={'dr3_source_id':str, 'KIC':str, 'kepid':str }
    )
    if sample == 'allKIC':
        df['KIC'] = df.kepid.astype(str)

    # drop 3 stars with nan DR3 source id's
    df = df[~(df.dr3_source_id.astype(str) == 'nan')]

    df['M_G'] = (
        df['dr3_phot_g_mean_mag'] + 5*np.log10(df['dr3_parallax']/1e3) + 5
    )

    ###########################
    # build the quality flags #
    ###########################

    df['flag_dr3_ruwe_outlier'] = df['dr3_ruwe'] > 1.4

    df['flag_kepler_gaia_ang_dist'] = df['dr3_kepler_gaia_ang_dist'] > 3

    df['flag_Teffrange'] = (
        (df['adopted_Teff'] < 3800)
        |
        (df['adopted_Teff'] > 6200)
    )

    df['flag_logg'] = df['adopted_logg'] < 4.2

    df['flag_dr3_M_G'] = (df['M_G'] < 3.9) | (df['M_G'] > 8.5)

    if sample == 'gyro':
        df['flag_is_CP_CB'] = ~(
            pd.isnull(df.s21_flag1)
            &
            pd.isnull(df.s19_flag1)
        )
    elif sample == 'allKIC':
        df['flag_is_CP_CB'] = np.zeros(len(df)).astype(bool)

    df['flag_kepler_gaia_xmambiguous'] = (
        df.count_n_gaia_nbhr >= 1
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
    groupname = f'field_{sample}_{datestr}'

    n_max = int(6e4)
    if sample == 'allKIC':
        n_max = int(2e5)

    gdf = given_source_ids_get_gaia_data(
        source_ids, groupname, n_max=n_max, overwrite=False,
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
    runid = f'field_{sample}_{datestr}_neighbors'

    count_df, ndf = given_source_ids_get_neighbor_counts(
        source_ids, dGmag, sep_arcsec, runid, n_max=n_max, overwrite=False,
        enforce_all_sourceids_viable=True, gaia_datarelease='gaiadr3',
        impose_nmax=0
    )

    # NOTE this "nbhr_count" number may double-count at times, based on 
    # Kepler-1975 (KOI 7913) == KIC 8873450...  nonetheless, seems to be a specific
    # glitch.
    df['nbhr_count'] = np.array(count_df.nbhr_count)
    df['flag_dr3_crowding'] = df['nbhr_count'] >= 1

    #####################
    # RELATIVE AGE FLAG #
    #####################
    from gyrojo.locus_definer import constrained_polynomial_function
    # made by tests/test_logg_teff_locus.py ; if you want to redefine this
    # particular quality flag, you need to rerun that rescript.
    selfn = 'manual'
    csvpath = join(DATADIR, "interim", f"logg_teff_locus_coeffs_{selfn}.csv")
    coeffs = pd.read_csv(csvpath).values.flatten()
    _teff = nparr(df['adopted_Teff'])
    logg_locus = constrained_polynomial_function(
        _teff, coeffs, selfn=selfn
    )
    if selfn in ['simple', 'complex']:
        df['flag_farfrommainsequence'] = df['adopted_logg'] < logg_locus
    elif selfn == 'manual':
        # the logic behind this line is mixed.  it's basically "iso age
        # precision no better than a factor of three below around ~5400K, and
        # just not too far from where most KOIs are above it".
        csvpath = join(DATADIR, "interim", f"logg_teff_locus_coeffs_simple.csv")
        _coeffs = pd.read_csv(csvpath).values.flatten()
        _y = constrained_polynomial_function(_teff, _coeffs, selfn='simple')
        p = constrained_polynomial_function(_teff, coeffs, selfn=selfn)
        _y0 = p - 0.12
        # 'top'
        y0 = np.maximum(_y, _y0)
        df['flag_farfrommainsequence'] = df['adopted_logg'] < y0

    # calculate for ater
    df['b20t2_rel_E_Age'] = np.abs(df['b20t2_E_Age'])/df['b20t2_Age']
    df['b20t2_rel_e_Age'] = np.abs(df['b20t2_e_Age'])/df['b20t2_Age']

    ##############################
    # ROTATION PERIOD PROVENANCE #
    ##############################
    df['flag_Prot_provenance'] = ~(
        (df.Prot_provenance == 'Santos2019')
        |
        (df.Prot_provenance == 'Santos2021')
    )

    ################################
    # finally, is gyro applicable? #
    ################################

    # see description of flags in docstring
    flag_bits = {
        'flag_Teffrange': 0,
        'flag_logg': 1,
        'flag_dr3_M_G': 2,
        'flag_in_KEBC': 3,
        'flag_kepler_gaia_ang_dist': 4,
        'flag_kepler_gaia_xmambiguous': 5,
        'flag_dr3_non_single_star': 6,
        'flag_dr3_ruwe_outlier': 7,
        'flag_dr3_crowding': 8,
        'flag_farfrommainsequence': 9,
        'flag_is_CP_CB': 10,
        'flag_Prot_provenance': 11,
    }

    # Iterate over the flag columns and update the flag_gyro_quality column
    df['flag_gyro_quality'] = 0
    for flag, bit_pos in flag_bits.items():
        # Convert the flag column to NumPy array and perform left-shift operation
        shifted_values = np.left_shift(df[flag].astype(int).values, bit_pos)
        # Update the 'flag_gyro_quality' column using the shifted values
        df['flag_gyro_quality'] |= shifted_values

    # Define the mask to check bits 0 through 9 inclusive
    mask = 0b0000001111111111

    # Create the 'flag_is_gyro_applicable' column - uses bits 0 to 9, but ignores
    # bit 10.
    df['flag_is_gyro_applicable'] = ((df['flag_gyro_quality'] & mask) == 0)

    if sample == 'gyro':
        outcsv = join(
            TABLEDIR,
            f"field_gyro_posteriors_{datestr}_gyro_ages_X_GDR3_S19_S21_B20_with_qualityflags.csv"
        )
    elif sample == 'allKIC':
        outcsv = join(
            TABLEDIR,
            f"{sample}_{datestr}_X_GDR3_B20_with_qualityflags.csv"
        )

    df.to_csv(outcsv, index=False)
    print(f"Wrote {outcsv}")

if __name__ == "__main__":
    datestr = '20240430'
    build_gyro_quality_flag(sample='allKIC', datestr=datestr)
    build_gyro_quality_flag(sample='gyro', datestr=datestr)  # ie with rotation
