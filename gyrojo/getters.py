"""
Getters:
    | get_gyro_data
    | get_li_data
    | get_joint_results
    | get_kicstar_data
    | get_cleaned_gaiadr3_X_kepler_dataframe
"""
import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from os.path import join
from glob import glob
from numpy import array as nparr

from astropy.io import fits
from astropy.table import Table

from gyrointerp.helpers import prepend_colstr, left_merge

from gyrojo.paths import LOCALDIR, DATADIR, RESULTSDIR


def get_gyro_data(sampleid):
    """
    sample_id (str): in ['all', 'sel_2s']
    """

    # 864 rows: step0, +requiring consisting rotation period measurements
    csvpath = join(
        RESULTSDIR, "koi_gyro_posteriors_20230110",
        "step0_koi_gyro_ages_X_GDR3_B20_S19_S21_M14_M15.csv"
    )

    # merge against the sel_2s sample.
    # NOTE TODO: might want to just check the entire sample once this is fully
    # automated.  but for now, this is where the good science is at.
    kdf = pd.read_csv(csvpath)
    if sampleid == 'all':
        pass
    elif sampleid == 'sel_2s':
        sel_2s = kdf['median'] + kdf['+2sigma'] < 1000
        kdf = kdf[sel_2s]

    Prots = kdf['mean_period']
    Prot_errs = np.ones(len(Prots))
    Prot_errs[Prots<=15] = 0.01*Prots[Prots<=15]
    Prot_errs[(Prots>15) & (Prots<=20)] = 0.02*Prots[(Prots>15) & (Prots<=20)]
    Prot_errs[(Prots>20) & (Prots<=25)] = 0.03*Prots[(Prots>20) & (Prots<=25)]
    Prot_errs[(Prots>25) & (Prots<=30)] = 0.04*Prots[(Prots>25) & (Prots<=30)]
    Prot_errs[Prots>30] = 0.05*Prots[Prots>30]

    kdf['Prot_err'] = Prot_errs

    # manual hard drop.
    bad = (
        (kdf.kepler_name == 'Kepler-1563 b') # Teff=5873, Prot=46.4 days. >5sigma outlier.
        |
        (kdf.kepler_name == 'Kepler-1676 b') # Teff=6165, Prot=32.4 days. >5sigma outlier.
    )
    kdf = kdf[~bad]

    # yields 862 planets, for 638 stars.

    return kdf


def get_li_data(sampleid):

    from cdips.utils.mamajek import get_interp_BmV_from_Teff

    # Get B-V colors for stars with HIRES data available.  Take statistical
    # uncertainties to be propagated from the uncertainties in Teff.
    # (Systematic uncertainties with the conversion currently unknown, but
    # certainly important for any mildly evolved stars.)
    csvpath = join(DATADIR, "interim", f"koi_jump_getter_{sampleid}.csv")
    mjdf = pd.read_csv(csvpath)

    BmV = get_interp_BmV_from_Teff(nparr(mjdf.adopted_Teff))

    # hotter star --> lower B-V (and this is "minus 1 sigma" -- keep signed)
    BmV_merr = np.abs(BmV - get_interp_BmV_from_Teff(
        nparr(mjdf.adopted_Teff + mjdf.adopted_Teff_err)
    ))
    BmV_perr = np.abs(BmV - get_interp_BmV_from_Teff(
        nparr(mjdf.adopted_Teff - mjdf.adopted_Teff_err)
    ))

    mjdf['B-V'] = BmV
    mjdf['B-V_err'] = np.mean([BmV_merr, BmV_perr], axis=0)

    #
    # Collect Li EW measurements, assuming the 5 angstrom window.
    #
    names = nparr(mjdf.name)
    kepoi_names = nparr(mjdf.kepoi_name) # actual unique row identifier
    fn = lambda x: x.replace(".fits", "")
    spec_ids = nparr(mjdf.filename.apply(os.path.basename).apply(fn))
    li_dir = join(DATADIR, "interim", f"Li_EW_HIRES_{sampleid}")
    assert os.path.exists(li_dir)

    globstrs = [join(
        li_dir, name,
        f"{name}_{spec_id}_Li_EW_deltawav5.0_xshift*_results.csv")
        for name, spec_id in zip(names, spec_ids)
    ]
    li_paths = [glob(globstr)[0] for globstr in globstrs ]
    for li_path in li_paths:
        assert os.path.exists(li_path)
    li_df = pd.concat([pd.read_csv(f) for f in li_paths])
    li_df['kepoi_name'] = kepoi_names

    assert len(li_df) == len(mjdf)

    mldf = li_df.merge(mjdf, on='kepoi_name', how='left')

    return mldf



def get_joint_results(COMPARE_AGE_UNCS=0):

    csvpath = join(RESULTSDIR, 'koi_gyro_X_lithium_posteriors_20230208',
                   'all_merged_joint_age_posteriors.csv')
    df = pd.read_csv(csvpath)

    # Take the adopted age as the joint (gyro+li) age.
    # If the join age is NaN (only the case for one interesting system,
    # Kepler-1939/800myr/1Rearth), take the adopted age as the gyro age.
    df['adopted_age_median'] = df['joint_median']
    df['adopted_age_+1sigma'] = df['joint_+1sigma']
    df['adopted_age_-1sigma'] = df['joint_-1sigma']

    _sel = pd.isnull(df['adopted_age_median'])
    df.loc[_sel, 'adopted_age_median'] = df.loc[_sel, 'gyro_median']
    df.loc[_sel, 'adopted_age_+1sigma'] = df.loc[_sel, 'gyro_+1sigma']
    df.loc[_sel, 'adopted_age_-1sigma'] = df.loc[_sel, 'gyro_-1sigma']

    GET_BERGER20_RADII = 0
    GET_BEST_RADII = 1
    assert np.sum([
        GET_BEST_RADII, GET_BERGER20_RADII
    ]) == 1

    if GET_BERGER20_RADII:
        # as a quick-look -- pull the Berger+20 radii.
        d0 = join(DATADIR, 'literature')
        p0 = 'Berger_2020_AJ_160_108_table1_planet_radii.fits'
        fitspath = join(d0, p0)
        hl = fits.open(fitspath)
        bdf = Table(hl[1].data).to_pandas()

        bdf['kepoi_name'] = (
            bdf.KOI.astype(str).apply(
                lambda x: "K"+str(x[:-2]).zfill(6)+str(x[-2:])
            )
        )
        bdf = prepend_colstr("B20_", bdf)

        df = df.merge(bdf, how='left', left_on='kepoi_name',
                      right_on="B20_kepoi_name")

        # if berger radius is null, use the KOI radius TODO fix this!!! 
        # better to start with the CKS X radii.  then berger.  then this.
        _sel = pd.isnull(df['B20_Radius'])
        N_null_0 = len(df[_sel])
        df.loc[_sel, 'B20_Radius'] = df.loc[_sel, 'koi_prad']
        df.loc[_sel, 'B20_E_Radius'] = df.loc[_sel, 'koi_prad_err1']
        df.loc[_sel, 'B20_e_radius_lc'] = np.abs(df.loc[_sel, 'koi_prad_err2'])
        N_null_1 = len(df[pd.isnull(df['B20_Radius'])])
        print(N_null_0)
        print(N_null_1)

        a_rp = df['B20_Radius']
        a_rp_err1 = df['B20_E_Radius']
        a_rp_err2 = df['B20_e_radius_lc']

    elif GET_BEST_RADII:

        # Petigura+2022 radii
        d1 = join(DATADIR, 'literature')
        p1 = 'Petigura_2022_CKS_X_t2_CXM_planets.fits'
        fitspath = join(d1, p1)
        hl = fits.open(fitspath)
        pdf = Table(hl[1].data).to_pandas()
        pdf = prepend_colstr("P22_", pdf)
        hl.close()

        # Berger+2020 radii
        d0 = join(DATADIR, 'literature')
        p0 = 'Berger_2020_AJ_160_108_table1_planet_radii.fits'
        fitspath = join(d0, p0)
        hl = fits.open(fitspath)
        bdf = Table(hl[1].data).to_pandas()

        bdf['kepoi_name'] = (
            bdf.KOI.astype(str).apply(
                lambda x: "K"+str(x[:-2]).zfill(6)+str(x[-2:])
            )
        )
        bdf = prepend_colstr("B20_", bdf)

        df = df.merge(pdf, how='left', left_on='kepoi_name',
                      right_on="P22_Planet")
        df = df.merge(bdf, how='left', left_on='kepoi_name',
                      right_on="B20_kepoi_name")

        # by default, adopt Petigura+22
        # big E is upper
        df['Rp'] = df['P22_Rp']
        df['E_Rp'] = df['P22_E_Rp']
        df['e_Rp'] = np.abs(df['P22_e_rp_lc'])

        # else, take Berger+20
        _sel = pd.isnull(df['Rp'])
        df.loc[_sel, 'Rp'] = df.loc[_sel, 'B20_Radius']
        df.loc[_sel, 'E_Rp'] = df.loc[_sel, 'B20_E_Radius']
        df.loc[_sel, 'e_Rp'] = np.abs(df.loc[_sel, 'B20_e_radius_lc'])

        # else, take KOI radii(?)
        _sel = pd.isnull(df['Rp'])
        df.loc[_sel, 'Rp'] = df.loc[_sel, 'koi_prad']
        df.loc[_sel, 'E_Rp'] = df.loc[_sel, 'koi_prad_err1']
        df.loc[_sel, 'e_Rp'] = np.abs(df.loc[_sel, 'koi_prad_err2'])

        a_rp = df['Rp']
        a_rp_err1 = df['E_Rp']
        a_rp_err2 = df['e_Rp']


    else:
        # just the KOI radii.  crappy uncertainties.
        a_rp = df['koi_prad']
        a_rp_err1 = df['koi_prad_err1'] # upper
        a_rp_err2 = np.abs(df['koi_prad_err2']) # lower

    if GET_BEST_RADII and COMPARE_AGE_UNCS:

        # Petigura+2022 star info
        d1 = join(DATADIR, 'literature')
        p1 = 'Petigura_2022_CKS_X_t1_stars.txt'
        txtpath = join(d1, p1)
        t = Table.read(txtpath, format='cds')
        pdf = t.to_pandas()
        pdf = prepend_colstr("P22S_", pdf)

        df['kepoi_number_str'] = df['kepoi_name'].apply(
            lambda x: x[:-3]
        )
        pdf['kepoi_number_str'] = pdf['P22S_KOI'].apply(
            lambda x: "K"+str(x).zfill(5)
        )

        mdf = df.merge(pdf, how='left', on='kepoi_number_str')

        return mdf


    a_period = df['koi_period']

    a_age = 1e6*(df['adopted_age_median'])
    a_age_err1 = 1e6*(df['adopted_age_+1sigma'])
    a_age_err2 = 1e6*(df['adopted_age_-1sigma'])

    df['pl_name'] = df['kepler_name']
    _sel = pd.isnull(df['pl_name'])
    df.loc[_sel, 'pl_name'] = df.loc[_sel, 'kepoi_name']
    a_pl_name = df['pl_name']

    mes = df['koi_max_mult_ev']

    paramdict = {}
    paramdict['rp'] = nparr(a_rp)
    paramdict['rp_err1'] = nparr(a_rp_err1)
    paramdict['rp_err2'] = nparr(a_rp_err2)
    paramdict['period'] = nparr(a_period)
    paramdict['age'] = nparr(a_age)
    paramdict['age_err1'] = nparr(a_age_err1)
    paramdict['age_err2'] = nparr(a_age_err2)
    paramdict['pl_name'] = nparr(a_pl_name)
    paramdict['mes'] = nparr(mes)

    return df, paramdict


def get_kicstar_data(sampleid):

    assert sampleid in ['Santos19_Santos21_all', 'Santos19_Santos21_clean0',
                        'Santos19_Santos21_logg']

    csvpath = join(DATADIR, 'interim', 'S19_S21_merged_X_GDR3_X_B20.csv')

    if not os.path.exists(csvpath):
        from astroquery.vizier import Vizier
        Vizier.ROW_LIMIT = -1

        # Santos+2019: M+K stars
        # https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJS/244/21
        # 'KIC', 'Kpmag', 'Q', 'Teff', 'E_Teff', 'e_Teff', 'logg', 'E_logg',
        # 'e_logg', 'Mass', 'E_Mass', 'e_Mass', 'Prot', 'E_Prot', 'Sph', 'E_Sph',
        # 'Fl1', 'DMK', 'Fl2', 'Fl3', 'Fl4', 'Fl5', 'M17', 'Simbad', '_RA', '_DE'
        catalogs = Vizier.get_catalogs("J/ApJS/244/21")
        s19_df = catalogs[0].to_pandas()
        s19_df = s19_df.rename(
            {'Fl1':'s19_flag1', 'Fl2':'s19_flag2', 'Fl3':'s19_flag3', 'Fl4':'s19_flag4',
             'Fl5':'s19_flag5'},
            axis='columns'
        )
        s19_df['Provenance'] = 'Santos2019'
        for ix in range(1,6):
            s19_df[f"s21_flag{ix}"] = np.nan

        # Santos+2021: G+F stars
        # https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJS/255/17
        # 'KIC', 'Kpmag', 'Q', 'Teff', 'E_Teff', 'e_Teff', 'logg', 'E_logg',
        # 'e_logg', 'Mass', 'E_Mass', 'e_Mass', 'Prot', 'E_Prot', 'Sph', 'E_Sph',
        # 'flag1', 'flag2', 'flag3', 'flag4', 'flag5', 'KCat', 'Simbad', '_RA', '_DE'
        catalogs = Vizier.get_catalogs("J/ApJS/255/17")
        s21_df = catalogs[0].to_pandas()
        s21_df['Provenance'] = 'Santos2021'
        s21_df = s21_df.rename(
            {'flag1':'s21_flag1', 'flag2':'s21_flag2', 'flag3':'s21_flag3',
             'flag4':'s21_flag4', 'flag5':'s21_flag5'},
            axis='columns'
        )
        for ix in range(1,6):
            s21_df[f"s19_flag{ix}"] = np.nan

        selcols = [
            'KIC', 'Kpmag', 'Q', 'Teff', 'E_Teff', 'e_Teff', 'logg', 'E_logg',
            'e_logg', 'Mass', 'E_Mass', 'e_Mass', 'Prot', 'E_Prot', 'Sph', 'E_Sph',
            'Provenance',
            's19_flag1', 's19_flag2', 's19_flag3', 's19_flag4', 's19_flag5',
            's21_flag1', 's21_flag2', 's21_flag3', 's21_flag4', 's21_flag5'
        ]

        _df = pd.concat((s19_df[selcols], s21_df[selcols]))

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

        cgk_df = get_cleaned_gaiadr3_X_kepler_dataframe()

        # start merging!
        mdf0 = left_merge(_df, b20t1_df, 'KIC', 'b20t1_KIC')
        mdf1 = left_merge(mdf0, b20t2_df, 'KIC', 'b20t2_KIC')
        mdf2 = left_merge(mdf1, cgk_df, 'KIC', 'kepid')

        from copy import deepcopy
        df = deepcopy(mdf2)

        df.to_csv(csvpath, index=False)
        print(f"Wrote {csvpath}")

    else:
        df = pd.read_csv(csvpath)

    # default Teffs
    df['adopted_Teff'] = df['b20t2_Teff']
    df['adopted_Teff_provenance'] = 'Berger2020_table2'
    df['adopted_Teff_err'] = np.nanmean([
        np.array(df['b20t2_E_Teff']),
        np.array(np.abs(df['b20t2_e_Teff']))
    ], axis=0)

    # else, take Santos 19/21 Teffs
    _sel = pd.isnull(df['adopted_Teff'])
    df.loc[_sel, 'adopted_Teff'] = df.loc[_sel, 'Teff']
    df.loc[_sel, 'adopted_Teff_err'] = df.loc[_sel, 'e_Teff']
    df.loc[_sel, 'adopted_Teff_provenance'] = df.loc[_sel, 'Provenance']

    assert np.sum(pd.isnull(df['adopted_Teff'])) == 0
    assert np.sum(pd.isnull(df['adopted_Teff_err'])) == 0

    Prots = df['Prot']
    Prot_errs = np.ones(len(Prots))
    Prot_errs[Prots<=15] = 0.01*Prots[Prots<=15]
    Prot_errs[(Prots>15) & (Prots<=20)] = 0.02*Prots[(Prots>15) & (Prots<=20)]
    Prot_errs[(Prots>20) & (Prots<=25)] = 0.03*Prots[(Prots>20) & (Prots<=25)]
    Prot_errs[(Prots>25) & (Prots<=30)] = 0.04*Prots[(Prots>25) & (Prots<=30)]
    Prot_errs[Prots>30] = 0.05*Prots[Prots>30]

    df['Prot_err'] = Prot_errs

    assert np.sum(pd.isnull(df['Prot'])) == 0
    assert np.sum(pd.isnull(df['Prot_err'])) == 0

    sel = df.Prot < 45
    if sampleid == 'Santos19_Santos21_logg':
        sel &= df.logg > 4.2
    if sampleid == 'Santos19_Santos21_clean0':
        # S21 flags:
        #
        # s21_flag1: CP/CB candidate flag: 
        #   [0 == "no rotation modulation" (6 occurrences)
        #   [1 == "type 1 CP/CB classical pulsator / close-in binary) candidate (2251 occurrences)
        #
        # s21_flag2: Subsample
        #
        #   	Note (G1): Flag as follows:
        #   			1 = main-sequence or subgiant solar-like targets in
        #   					DR25 Mathur+ (2017, J/ApJS/229/30) and Berger+ (2020, J/AJ/159/280);
        #   			2 = main-sequence or subgiant solar-like targets only in
        #   					DR25 Mathur et al. (2017, J/ApJS/229/30);
        #   			3 = main-sequence or subgiant solar-like targets only in
        #   					DR25 Berger et al. (2020, J/AJ/159/280).
        #
        # s21_flag3: binarity flag
        #
        #   0 = single stars in Berger+ (2018, J/ApJ/866/99) and/or in
        #       Simonian+ (2019, J/ApJ/871/174);
        #   1 = binary candidates in Berger+ (2018, J/ApJ/866/99);
        #   2 = binary candidates in Simonian+ (2019, J/ApJ/871/174)
        #   3 = binary candidates in Berger+ (2018, J/ApJ/866/99) and in
        #       Simonian+ (2019, J/ApJ/871/174).
        #
        # s21_flag4: KOI flag
        #
        #   	Flag as follows:
        #   		0 = confirmed;
        #   		1 = candidate;
        #   		2 = false positive.	
        #
        # s21_flag5: stellar property source flag
        #   
        #   	Flag as follows:
        #   0 = Berger et al. (2020, J/AJ/159/280);
        #   1 = Mathur et al. (2017, J/ApJS/229/30)
        #
        # S19 flags:
        #
        # s19_flag1: CP/CB candidate flag: 
        #    (1): We distinguish between three types of classical pulsator (CP)
        #        candidates. Type-1 candidates show a behavior somewhat similar to RR
        #        Lyrae and Cepheids: high-amplitude and stable flux variations, beating
        #        patterns, and a large number of harmonics. Interestingly, a
        #        significant fraction of these targets were identified as Gaia binary
        #        candidates. Therefore, it is possible that these targets are not CPs
        #        but close-in binaries (CB). If that is the case, the signal may still
        #        be related to rotation, but may be distinct from the rotational
        #        behavior of single stars. We refer to these targets (350) as Type-1
        #        CP/CB candidates.
        #
        # s19_flag2: Gaia binary flag
        #
        # s19_flag3: Gaia subgiant flag
        #   Gaia binary (Gaia Bin.) and subgiant (Gaia Subg.) candidate flags from Berger
        #   et al. (2018), where 0 corresponds to single star and main-sequence star,
        #   respectively, and 1 corresponds to binary system and subgiant star,
        #   respectively
        #
        # s19_flag4: KOI flag
        #
        # s19_flag5: FlipPer Class flag
        #   FliPerClass (FPC) indicates targets that are possibly solar-type stars (0),
        #   classical pulsators (1), and binary systems/photometric pollution (2).

        sel &= df.logg > 4.2

        sel &= (
            df.Sph >= 500
        )

        not_CP_CB = pd.isnull(df.s21_flag1) & pd.isnull(df.s19_flag1)

        sel &= not_CP_CB

    df = df[sel]

    return df


def get_cleaned_gaiadr3_X_kepler_dataframe():
    """
    Kepler-Gaia DR3 crossmatch from https://gaia-kepler.fun/, downloaded to
    the ~/local/ directory.  Use the 4 arcsecond match (will go by brightest
    match below).
    For KIC stars with multiple potential Gaia matches within the 4 arcsecond
    search radius, we adopted the brightest star as the actual match.  This
    is typically unambiguous, since in most such cases there is a large
    brightness difference between the primary and any apparent neighbors.  We
    mark cases where the $G$-band difference in magnitudes is less than $0.5$
    with a quality flag, $\texttt{n\_gaia\_nbhr}$, which denotes the number
    of sources within 4 arcseconds and 0.5 $G$-band magnitudes of the
    primary.  NOTE: narrower search radii (e.g., the 1 arcsecond match)
    exclude a larger fraction of stars -- e.g., 91 of KOI's are omitted in
    the 1 arcsecond match, and 23 of these have either "candidate" or
    "confirmed" status.  The overwhelming majority of these 23 are M-dwarfs
    with high proper motions.  Performing the same crossmatch at 4
    arcseconds, only two KOIs (KOI-3993 and KOI-6531), both known false
    positives, fail to yield Gaia DR3 crossmatches.
    """

    kepler_dr3_path = join(LOCALDIR, "kepler_dr3_4arcsec.fits")
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

    return cgk_df