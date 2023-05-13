"""
This script takes the cumulative KOI table, and left-joins it against Gaia DR3,
the Berger+20 stellar parameter catalog, and all relevant rotation period
catalogs (McQuillan+14, Mazeh+15, Santos+19, Santos+21).

The output is written to /data/interim/koi_table_X_GDR3_B20_S19_S21_M14_M15.csv

Usage:
    $ python build_koi_table.py

TODO:
    2023.01.10: When cleaning up, can just remove all the DO_PRAESEPE_FLAGS
    stuff (since it is totally deprecated).
"""
import numpy as np, matplotlib.pyplot as plt, pandas as pd
import os, pickle

from astropy.table import Table
from astropy.io import fits

from agetools.paths import DATADIR, LOCALDIR, RESULTSDIR
from agetools.plotting import plot_sub_praesepe_selection_cut

from gyrointerp.helpers import prepend_colstr, left_merge

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

    # Kepler-Gaia DR3 crossmatch from https://gaia-kepler.fun/, downloaded to
    # the ~/local/ directory.  Use the 4 arcsecond match (will go by brightest
    # match below).
    # For KIC stars with multiple potential Gaia matches within the 4 arcsecond
    # search radius, we adopted the brightest star as the actual match.  This
    # is typically unambiguous, since in most such cases there is a large
    # brightness difference between the primary and any apparent neighbors.  We
    # mark cases where the $G$-band difference in magnitudes is less than $0.5$
    # with a quality flag, $\texttt{n\_gaia\_nbhr}$, which denotes the number
    # of sources within 4 arcseconds and 0.5 $G$-band magnitudes of the
    # primary.  NOTE: narrower search radii (e.g., the 1 arcsecond match)
    # exclude a larger fraction of stars -- e.g., 91 of KOI's are omitted in
    # the 1 arcsecond match, and 23 of these have either "candidate" or
    # "confirmed" status.  The overwhelming majority of these 23 are M-dwarfs
    # with high proper motions.  Performing the same crossmatch at 4
    # arcseconds, only two KOIs (KOI-3993 and KOI-6531), both known false
    # positives, fail to yield Gaia DR3 crossmatches.

    kepler_dr3_path = os.path.join(LOCALDIR, "kepler_dr3_4arcsec.fits")
    hdul = fits.open(kepler_dr3_path)
    gk_df = Table(hdul[1].data).to_pandas()
    hdul.close()

    gk_df = gk_df.sort_values(by=['kepid','phot_g_mean_mag'])
    udupkepids = np.unique(
        gk_df[gk_df.duplicated('kepid', keep=False)]['kepid']
    )
    idcache = {}
    for ukepid in udupkepids:
        g_mags = np.array(gk_df.loc[gk_df.kepid==ukepid, 'phot_g_mean_mag'])
        n_gaia_nbhr = np.sum(g_mags - min(g_mags) < 0.5) - 1
        idcache[ukepid] = n_gaia_nbhr

    cgk_df = gk_df[~gk_df.duplicated('kepid', keep='first')]
    n_gaia_nbhr = np.zeros(len(cgk_df))
    for ix, kepid in enumerate(cgk_df.kepid):
        if kepid in idcache:
            if idcache[kepid] >= 1:
                n_gaia_nbhr[ix] = idcache[kepid]

    cgk_df['flag_n_gaia_nbhr'] = n_gaia_nbhr.astype(int)

    okcols = ['kepid', 'nconfp', 'nkoi', 'ntce', 'jmag', 'hmag', 'kmag',
              'planet?', 'flag_n_gaia_nbhr']
    colstr = 'dr3_'
    cgk_df = cgk_df.rename({c:colstr+c for c in cgk_df.columns
                            if c not in okcols}, axis='columns')
    selcols = ['kepid', 'dr3_source_id', 'dr3_ra', 'dr3_dec', 'dr3_parallax',
               'dr3_parallax_error', 'dr3_parallax_over_error', 'dr3_pmra',
               'dr3_pmra_error', 'dr3_pmdec', 'dr3_pmdec_error', 'dr3_ruwe',
               'dr3_phot_g_mean_flux_over_error', 'dr3_phot_g_mean_mag',
               'dr3_phot_bp_mean_flux_over_error', 'dr3_phot_bp_mean_mag',
               'dr3_phot_rp_mean_flux_over_error', 'dr3_phot_rp_mean_mag',
               'dr3_phot_bp_rp_excess_factor', 'dr3_bp_rp',
               'dr3_radial_velocity', 'dr3_radial_velocity_error',
               'dr3_rv_nb_transits', 'dr3_rv_renormalised_gof',
               'dr3_rv_chisq_pvalue', 'dr3_phot_variable_flag', 'dr3_l',
               'dr3_b', 'dr3_non_single_star', 'dr3_teff_gspphot',
               'dr3_logg_gspphot', 'dr3_mh_gspphot', 'dr3_distance_gspphot',
               'dr3_ag_gspphot', 'dr3_ebpminrp_gspphot',
               'dr3_kepler_gaia_ang_dist', 'dr3_pm_corrected', 'nconfp',
               'nkoi', 'ntce', 'planet?', 'flag_n_gaia_nbhr']
    # "cleaned" Gaia-Kepler dataframe.  duplicates accounted for; dud columns
    # dropped (most columns kept).
    cgk_df = cgk_df[selcols]
    cgk_df['dr3_source_id'] = cgk_df['dr3_source_id'].astype(str)
    assert np.all(~pd.isnull(cgk_df.dr3_source_id))

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

    DO_PRAESEPE_FLAGS = False

    if DO_PRAESEPE_FLAGS:

        # Evaluate whether the quoted rotation periods are below the Praesepe
        # sequence.  This requires them to have 3800 >= Teff >= 6200.  (otherwise,
        # NaN is returned)
        period_cols = ['s19_Prot', 's21_Prot', 'm14_Prot', 'm15_Prot']
        for period_col in period_cols:

            period_val = mdf5[period_col]
            teff_val = mdf5["adopted_Teff"]

            sel_below_Praesepe = period_val < PraesepeInterpModel(
                teff_val, bounds_error=False, polynominal_order=5
            )

            flag_key = f"flag_{period_col}_below_Praesepe"
            mdf5[flag_key] = sel_below_Praesepe

        mdf5['flag_any_Prot_below_Praesepe'] = (
            mdf5['flag_s19_Prot_below_Praesepe']
            |
            mdf5['flag_s21_Prot_below_Praesepe']
            |
            mdf5['flag_m14_Prot_below_Praesepe']
            |
            mdf5['flag_m15_Prot_below_Praesepe']
        )

        N_KOIs = len(mdf5[(mdf5["adopted_Teff"]>=3800) & (mdf5["adopted_Teff"]<=6200)])
        N_KOIs_below_Praesepe = mdf5.flag_any_Prot_below_Praesepe.sum()
        print(f"{N_KOIs_below_Praesepe}/{N_KOIs} KOIs in 3800-6200K range below Praesepe")
        print(f"(not accounting for other flags!)")

        DEBUG = 1
        if DEBUG:
            for poly_order in [5,6,7]:
                plot_sub_praesepe_selection_cut(mdf5, poly_order=poly_order)

        # Evaluate whether the rotation amplitudes are consistent with the Praesepe
        # floor from Rebull+2020, Figure 10: 0.001 mag ~= 0.1%.  Convert the scales
        # of "S_ph", "R_var", "R_per" appropriately to match Luisa Rebull's scale,
        # per doc/20220902_amplitude_measurements.txt
        def Sph_to_Rebull(Sph):
            return Sph * 2.56

        def Rvar_to_Rebull(Rvar):
            return Rvar * (2.56/3.28)

        def Rper_to_Rebull(Rper):
            return Rper * (2.56/3.28)

        amplitude_cols = ["s19_Sph", "s21_Sph", "m14_Rper", "m15_Rvar"]
        CUTOFF = 1e-3

        mdf5["flag_s19_Ampl_below_cutoff"] = (
            (Sph_to_Rebull(mdf5["s19_Sph"]/1e6) < CUTOFF)
        )
        mdf5["flag_s21_Ampl_below_cutoff"] = (
            (Sph_to_Rebull(mdf5["s21_Sph"]/1e6) < CUTOFF)
        )
        mdf5["flag_m14_Ampl_below_cutoff"] = (
            (Rper_to_Rebull(mdf5["m14_Rper"]/1e6) < CUTOFF)
        )
        mdf5["flag_m15_Ampl_below_cutoff"] = (
            (Rvar_to_Rebull(mdf5["m15_Rvar"]/1e6) < CUTOFF)
        )

        # This flag is the generalization of "flag_any_Prot_below_Praesepe" to
        # include the condition that the paper-specific amplitudes must also be
        # _above_ the Praesepe amplitude floor of 0.1%.
        mdf5['flag_any_Prot_Ampl_below_Praesepe'] = (
            (mdf5['flag_s19_Prot_below_Praesepe'] & (~mdf5["flag_s19_Ampl_below_cutoff"]))
            |
            (mdf5['flag_s21_Prot_below_Praesepe'] & (~mdf5["flag_s21_Ampl_below_cutoff"]))
            |
            (mdf5['flag_m14_Prot_below_Praesepe'] & (~mdf5["flag_m14_Ampl_below_cutoff"]))
            |
            (mdf5['flag_m15_Prot_below_Praesepe'] & (~mdf5["flag_m15_Ampl_below_cutoff"]))
        )

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

    if DO_PRAESEPE_FLAGS:
        raise AssertionError("DEPRECATED!") # but here for posterity
        mdf5['flag_selected_step1'] = (
            (mdf5['flag_any_Prot_Ampl_below_Praesepe'])
            &
            (~mdf5['flag_m15_combined'])
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
        N_KOIs_step1 = mdf5.flag_selected_step1.sum()
        assert np.all(mdf5[mdf5['flag_selected_step1']]['flag_selected_step0'])

    N_KOIs_step0 = mdf5.flag_selected_step0.sum()
    print(f"{N_KOIs_step0} KOIs meet 'step 0' "
          "(3800-6200K, logg>4, not FP, not grazing, finite MES, MES>10, "
          "not flagged by Mazeh+15)")
    if DO_PRAESEPE_FLAGS:
        print(f"{N_KOIs_step1} KOIs meet 'step 1' "
              "(below Praesepe in Prot/ampl, 3800-6200K, logg>4, not FP, "
              "not grazing, finite MES, MES>10, not flagged by Mazeh+15)")

    print(f"Writing {outcsvpath}")
    mdf5['dr3_source_id'] = mdf5['dr3_source_id'].astype(str)
    mdf5.to_csv(outcsvpath, index=False)

    return mdf5


if __name__ == "__main__":
    mdf5 = build_koi_table(overwrite=1)
    import IPython; IPython.embed()
