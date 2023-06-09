"""
This script takes the cumulative KOI table, and left-joins it against Gaia DR3,
the Berger+20 stellar parameter catalog, and all relevant rotation period
catalogs (McQuillan+14, Mazeh+15, Santos+19, Santos+21).

The output is written to /data/interim/koi_table_X_GDR3_B20_S19_S21_M14_M15.csv

Usage:
    $ python build_koi_table.py
"""
import numpy as np, matplotlib.pyplot as plt, pandas as pd
import os, pickle

from astropy.table import Table
from astropy.io import fits

from agetools.paths import DATADIR, LOCALDIR, RESULTSDIR
from agetools.plotting import plot_sub_praesepe_selection_cut

from gyrointerp.helpers import prepend_colstr, left_merge
from gyrojo.getters import get_cleaned_gaiadr3_X_kepler_dataframe

from astroquery.vizier import Vizier

def build_koi_table(overwrite=0):

    outcsvpath = os.path.join(DATADIR, "interim",
                              "koi_table_X_GDR3_B20_S19_S21_M14_M15.csv")

    if os.path.exists(outcsvpath) and not overwrite:
        print(f"Found {outcsvpath}, returning it and not overwriting.")
        return pd.read_csv(outcsvpath)

    # Kepler Q1-Q17 DR 25 table (Thompson+2018)
    # https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative
    # 2022/06/27
    koipath = os.path.join(
        DATADIR, 'raw',
        'cumulative_2022.06.27_15.31.02.csv'
    )
    koi_df = pd.read_csv(koipath, comment='#', sep=',')

    cgk_df = get_cleaned_gaiadr3_X_kepler_dataframe()

    Vizier.ROW_LIMIT = -1

    # Berger+2020: Gaia-Kepler stellar properties catalog.
    _v = Vizier(columns=["**"])
    _v.ROW_LIMIT = -1
    catalogs = _v.get_catalogs("J/AJ/159/280")

    # Table 1: input parameters
    # 'KIC', 'gmag', 'e_gmag', 'Ksmag', 'e_Ksmag', 'plx', 'e_plx', '__Fe_H_',
    # 'e__Fe_H_', 'RUWE', 'Ncomp', 'KsCorr', 'State', 'output', 'KSPC', '_RA',
    # '_DE'
    b20t1_df = catalogs[0].to_pandas()
    b20t1_df = prepend_colstr('b20t1_', b20t1_df)

    # Table 2: output parameters
    # "E_" means upper err, "e_" means lower.  Note that "e_" is signed, so
    # that all entries in these columns are negative.
	# ['recno', 'KIC', 'Mass', 'E_Mass', 'e_Mass', 'Teff', 'E_Teff', 'e_Teff',
    #  'logg', 'E_logg', 'e_logg', '__Fe_H_', 'E__Fe_H_', 'e__Fe_H_', 'Rad',
    #  'E_Rad', 'e_Rad', 'rho', 'E_rho', 'e_rho', 'Lum', 'E_Lum', 'e_Lum',
    #  'Age', 'f_Age', 'E_Age', 'e_Age', 'Dist', 'E_Dist', 'e_Dist', 'Avmag',
    #  'GOF', 'TAMS']
    b20t2_df = catalogs[1].to_pandas() # output parameters
    b20t2_df = prepend_colstr('b20t2_', b20t2_df)

    # Santos+2019: M+K stars
    # https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJS/244/21
    # 'KIC', 'Kpmag', 'Q', 'Teff', 'E_Teff', 'e_Teff', 'logg', 'E_logg',
    # 'e_logg', 'Mass', 'E_Mass', 'e_Mass', 'Prot', 'E_Prot', 'Sph', 'E_Sph',
    # 'Fl1', 'DMK', 'Fl2', 'Fl3', 'Fl4', 'Fl5', 'M17', 'Simbad', '_RA', '_DE'
    catalogs = Vizier.get_catalogs("J/ApJS/244/21")
    s19_df = catalogs[0].to_pandas()
    s19_df = prepend_colstr('s19_', s19_df)

    # Santos+2021: G+F stars
    # https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJS/255/17
    # 'KIC', 'Kpmag', 'Q', 'Teff', 'E_Teff', 'e_Teff', 'logg', 'E_logg',
    # 'e_logg', 'Mass', 'E_Mass', 'e_Mass', 'Prot', 'E_Prot', 'Sph', 'E_Sph',
    # 'flag1', 'flag2', 'flag3', 'flag4', 'flag5', 'KCat', 'Simbad', '_RA', '_DE'
    catalogs = Vizier.get_catalogs("J/ApJS/255/17")
    s21_df = catalogs[0].to_pandas()
    s21_df = prepend_colstr('s21_', s21_df)

    # McQuillan+2014
    # https://vizier.u-strasbg.fr/viz-bin/VizieR-3?-source=J/ApJS/211/24/table1
    # 'KIC', 'Teff', 'log_g_', 'Mass', 'Prot', 'e_Prot', 'n_Prot', 'Rper',
    # 'LPH', 'w', 'Ref', '_RA', '_DE'
    catalogs = Vizier.get_catalogs("J/ApJS/211/24")
    m14_df = catalogs[0].to_pandas()
    m14_df = prepend_colstr('m14_', m14_df)

    # Mazeh+2015
    # https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/ApJ/801/3
    # columns (single letters being flags):
    # 'KOI', 'KIC', 'Teff', 'log_g_', 'Prot', 'e_Prot', 'Rvar', 'LPH', 'w', 'D',
    # 'N', 'C', 'G', 'T', 'F', 'R', 'M1', 'M2', 'Simbad', '_RA', '_DE'
    # .
    catalogs = Vizier.get_catalogs("J/ApJ/801/3")
    m15_df = catalogs[0].to_pandas()
    m15_df = prepend_colstr('m15_', m15_df)

    #
    # BEGIN MERGING
    #
    print('begin merging')
    mdf0 = left_merge(koi_df, cgk_df, 'kepid', 'kepid')
    mdf0.loc[pd.isnull(mdf0.dr3_source_id), 'dr3_source_id'] = 'XMATCH NOT FOUND'
    assert np.all(~pd.isnull(mdf0.dr3_source_id))

    mdf1 = left_merge(mdf0, b20t2_df, 'kepid', 'b20t2_KIC')
    mdf2 = left_merge(mdf1, s19_df, 'kepid', 's19_KIC')
    mdf3 = left_merge(mdf2, s21_df, 'kepid', 's21_KIC')
    mdf4 = left_merge(mdf3, m14_df, 'kepid', 'm14_KIC')
    mdf5 = left_merge(mdf4, m15_df, 'kepid', 'm15_KIC')

    period_cols = ['s19_Prot', 's21_Prot', 'm14_Prot', 'm15_Prot']

    N_reported_periods = np.zeros(len(mdf5))
    for period_col in period_cols:
        N_reported_periods += (
            ~pd.isnull(mdf5[period_col])
            &
            (mdf5[period_col] > 0)
        ).astype(int)

    # N_reported_periods: the number of reported rotation period measurements,
    # from Santos+19, Santos+21, McQuillan+14, Mazeh+15.  Max 4, min 0.
    mdf5['N_reported_periods'] = N_reported_periods.astype(int)

    # Whether Berger+2020 parameters are reported.  Their sample selection is
    # described in their Section 2.1.  They use the Gaia-Kepler crossmatch
    # "detailed in Berger+2018b", which inclouded 195,710 stars.  They removed
    # stars lacking "AAA" 2MASS photometry, and stars lacking measured
    # parallaxes in Gaia DR2.  This means they removed the brightest stars due
    # to saturation and the faintest stars due to photon noise.  These cuts
    # reduced the sample to 190,213 and then to 186,672 stars, respectively.
    # Requiring g-band photometry from either the KIC or the Kepler-INT survey
    # reduced the catalog to 186,548 stars.  (Note here "Berger+2018b" refers
    # to https://ui.adsabs.harvard.edu/abs/2018ApJ...866...99B/abstract).  The
    # crossmatch used the CDS X-match service, and made subsequent cleaning
    # based on position and magnitude differences of the sources.
    mdf5['has_b20t2_params'] = ~(pd.isnull(mdf5['b20t2_KIC']).astype(bool))

    # Construct Teff and logg columns as: B+20 if available, else Gaia DR3.
    # From 9564 KOIs, this yields Teff and logg from Counter({'Berger+2020':
    # 8870, 'GaiaDR3-gspphot': 420, '': 274}).  The Gaia DR3 inclusion helps
    # with cases such as KOI-7913, which otherwise would be missed.
    # For the small GaiaDR3-gspphot subset, assign fixed 250K and 0.2dex
    # uncertainties on Teff and logg.
    mdf5['adopted_Teff'] = np.ones(len(mdf5))*np.nan
    mdf5['provenance_Teff'] = ''
    mdf5['adopted_logg'] = np.ones(len(mdf5))*np.nan
    mdf5['provenance_logg'] = ''

    _sel = ~pd.isnull(mdf5['b20t2_Teff'])
    mdf5.loc[_sel, 'adopted_Teff'] = mdf5.loc[_sel, 'b20t2_Teff']
    mdf5.loc[_sel, 'provenance_Teff'] = 'Berger+2020'
    for colstr in ['E_Teff', 'e_Teff']:
        mdf5.loc[_sel, f'adopted_{colstr}'] = mdf5.loc[_sel, f'b20t2_{colstr}']

    _sel = pd.isnull(mdf5['b20t2_Teff']) & ~pd.isnull(mdf5['dr3_teff_gspphot'])
    mdf5.loc[_sel, 'adopted_Teff'] = mdf5.loc[_sel, 'dr3_teff_gspphot']
    mdf5.loc[_sel, 'provenance_Teff'] = 'GaiaDR3-gspphot'
    mdf5.loc[_sel, f'adopted_E_Teff'] = 200
    mdf5.loc[_sel, f'adopted_e_Teff'] = -200

    # gyro-interp requires symmetric uncertainties
    mdf5['adopted_Teff_err'] = np.nanmean([
        np.array(mdf5['adopted_E_Teff']),
        np.array(np.abs(mdf5['adopted_e_Teff']))
    ], axis=0)

    _sel = ~pd.isnull(mdf5['b20t2_logg'])
    mdf5.loc[_sel, 'adopted_logg'] = mdf5.loc[_sel, 'b20t2_logg']
    mdf5.loc[_sel, 'provenance_logg'] = 'Berger+2020'
    for colstr in ['E_logg', 'e_logg']:
        mdf5.loc[_sel, f'adopted_{colstr}'] = mdf5.loc[_sel, f'b20t2_{colstr}']

    _sel = pd.isnull(mdf5['b20t2_logg']) & ~pd.isnull(mdf5['dr3_teff_gspphot'])
    mdf5.loc[_sel, 'adopted_logg'] = mdf5.loc[_sel, 'dr3_teff_gspphot']
    mdf5.loc[_sel, 'provenance_logg'] = 'GaiaDR3-gspphot'
    mdf5.loc[_sel, f'adopted_E_logg'] = 0.2
    mdf5.loc[_sel, f'adopted_e_logg'] = -0.2

    from collections import Counter
    _r = Counter(mdf5['provenance_Teff'])
    print(42*'-')
    print(_r)
    print(42*'-')

    # Get the mean period, for cases in which it was reported.
    period_arr = np.array([mdf5[period_col] for period_col in period_cols])
    period_arr[period_arr == 0] = np.nan
    mean_period = np.nanmean(period_arr, axis=0)
    mdf5['mean_period'] = mean_period

    # Do all reported finite periods agree within 10% of the mean period?
    eps = 0.1
    periods_agree = (
        ( np.abs(period_arr - mean_period[None,:])/period_arr < eps )
        |
        (np.isnan(period_arr))
    )
    mdf5['flag_reported_periods_agree'] = np.all(periods_agree, axis=0)

    #
    # Begin stellar quality flags
    #

    # Flag to account for Mazeh+2015 quality flag specifics.  Santos+19,21 and
    # McQuillan+2014 did not have any analogous flags.  See
    # doc/20220902_quality_flag_accounting.txt for explanation.
    mdf5['flag_m15_combined'] = (
        (mdf5['m15_C'] == 1.)
        |
        (mdf5['m15_G'] == 1.)
        |
        (mdf5['m15_F'] == 1.)
        |
        (mdf5['m15_R'] == 1.)
        |
        (mdf5['m15_M1'] == 1.)
        |
        (mdf5['m15_M2'] == 1.)
    )

    # Require finite Teff and logg
    mdf5['flag_nanteffloggperiod'] = (
        pd.isnull(mdf5.mean_period)
        |
        pd.isnull(mdf5.adopted_Teff)
        |
        pd.isnull(mdf5.adopted_logg)
    )

    # Flag for whether the star's median Teff falls within our allowed range.
    mdf5['flag_st_toohot_toocold'] = (
        (mdf5.adopted_Teff > 6200)
        |
        (mdf5.adopted_Teff < 3800)
    ).astype(bool)

    # Based on berger18_berger20_spectral_classification_dwarfsontop.png
    # At <6200K, the overwhelming majority of stars are subgiants.
    mdf5['flag_st_is_low_logg'] = (
        (mdf5.adopted_logg < 4.0)
    ).astype(bool)

    # Optional flags, that are important for interpreting whether
    # the gyro age might be treated with suspicion.
    mdf5['flag_ruwe'] = (
        (mdf5.dr3_ruwe > 1.2)
    ).astype(bool)

    # NOTE: the neighboring star count flags are calculated done _AFTER_ the
    # application of the rotation vs Teff cut.

    #
    # Begin planet quality flags
    #

    mdf5['flag_koi_is_fp'] = (
        (mdf5.koi_disposition == "FALSE POSITIVE")
    ).astype(bool)

    mdf5['flag_koi_is_grazing'] = (
        mdf5.koi_impact > 0.9
    ).astype(bool)

    mdf5['flag_koi_is_low_snr'] = (
        (mdf5.koi_max_mult_ev < 10)
        |
        (pd.isnull(mdf5.koi_max_mult_ev))
    ).astype(bool)

    #
    # Combined flag
    #

    # "Step 0" is KOIs that have 3800-6200K, logg>4, not FP, not grazing,
    # finite MES, MES>10, and that were not flagged by Mazeh+15 (as
    # C/G/F/R/M1/M2).  That is the adopted output of this routine.

    # A deprecated (i.e. no longer included!) "Step 1" was the same sample, but
    # applying the rotation period and amplitude cuts to require sub-Praesepe
    # (according to at least one of M14,M15,S19,S21).  This yielded the lithium
    # targets for HIRES selection.
    #
    # ------------------------------------------ 
    # [WHEN WE DIDN'T INCLUDE M1/M2 IN STEP1] 
    # We find that 2363 KOIs meet "step 0", while 110 meet "step 1" (4.7%).  By
    # way of comparison, assuming a constant SFR in the Galaxy over the past 10
    # billion years, we would expect 7% of stars to be below Praesepe (700 Myr)
    # in age.  This puts us within a factor of two of expectations -- the
    # origin of the ~few-percent disagreement remains to be explored.
    #
    # Of the 110 stars meeting step 1,
    # two have conflicting rotation periods from the reported literature
    # values.  The first is KIC 10140122 (KOI-4545), with a period of 7.2 days
    # reported by S+19, and 14.3 days reported by M+15.  The second is KIC
    # 4476123 (KOI-814, aka. Kepler-689b), with a period of 12.5 days from
    # S+21, and 5.0 days from M+15.  Visual inspection of each KOI-4545 shows
    # what Mazeh+2015 already noted: both stars show marginal rotation signals
    # that are inconsistent between adjacent quarters, suggesting that they
    # originate from neighboring stars blended in the Kepler images.  Both
    # systems are excluded from further consideration.
    # ------------------------------------------ 
    #
    # [INCLUDING M1/M2 IN STEP1] 
    # We find that 1018 KOIs meet "step 0", while 86 meet "step 1" (8.4%).  By
    # way of comparison, assuming a constant SFR in the Galaxy over the past 10
    # billion years, we would expect 7% of stars to be below Praesepe (700 Myr)
    # in age.  This puts us reasonably close to expectations.
    #
    # Of the 82 stars meeting step 1, all have rotation periods that agree
    # within the reported literature.
    # ------------------------------------------ 

    mdf5['flag_selected_step0'] = (
        (~mdf5['flag_m15_combined'])
        &
        (~mdf5['flag_nanteffloggperiod'])
        &
        (~mdf5['flag_st_toohot_toocold'])
        &
        (~mdf5['flag_st_is_low_logg'])
        &
        (~mdf5['flag_koi_is_fp'])
        &
        (~mdf5['flag_koi_is_grazing'])
        &
        (~mdf5['flag_koi_is_low_snr'])
    )

    N_KOIs_step0 = mdf5.flag_selected_step0.sum()
    print(f"{N_KOIs_step0} KOIs meet 'step 0' "
          "(3800-6200K, logg>4, not FP, not grazing, finite MES, MES>10, "
          "not flagged by Mazeh+15)")

    print(f"Writing {outcsvpath}")
    mdf5['dr3_source_id'] = mdf5['dr3_source_id'].astype(str)
    mdf5.to_csv(outcsvpath, index=False)

    return mdf5


if __name__ == "__main__":
    mdf5 = build_koi_table(overwrite=1)
    import IPython; IPython.embed()
