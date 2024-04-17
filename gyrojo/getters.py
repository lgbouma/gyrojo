"""
Getters:
    | get_kicstar_data
    | get_gyro_data
    | get_li_data
    | get_age_results
    | get_cleaned_gaiadr3_X_kepler_dataframe
    | get_cleaned_gaiadr3_X_kepler_supplemented_dataframe
    | get_koi_data
Selector:
    | select_by_quality_bits
"""
import os
from copy import deepcopy
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from os.path import join
from glob import glob
from numpy import array as nparr

from astropy.io import fits
from astropy.table import Table

from gyrointerp.helpers import prepend_colstr, left_merge

from gyrojo.paths import LOCALDIR, DATADIR, RESULTSDIR, TABLEDIR


def get_gyro_data(sampleid, koisampleid='cumulative-KOI',
                  drop_grazing=1, drop_highruwe=1):
    """
    Args:
        sampleid (str): the dataframe of either stars or planets to return.
        Specified as:

            "Santos19_Santos21_dquality": stars from KIC, x Santos19 and Santos21,
            with gyro ages calculated.  "flag_is_gyro_applicable" has been
            calculated.

            "koi_X_S19S21dquality": planets, as above.

        koisampleid (str): if you're getting planets, which slice of the KOI
        tables do you want?  "cumulative-KOI" or "DR25-KOI".

        drop_grazing (bool): whether you want "flag_is_ok_planetcand" to drop
        grazing planets, or not.  The "flag_koi_is_grazing" will be true no
        matter what.
    """

    assert sampleid in [
        'Santos19_Santos21_dquality',
        'koi_X_S19S21dquality',
        'deprecated_all',
        'deprecated_sel_2s'
    ]

    if sampleid in ['Santos19_Santos21_dquality', 'koi_X_S19S21dquality']:
        # made by construct_field_star_gyro_quality_flags.py driver
        csvpath = join(
            TABLEDIR,
            'field_gyro_posteriors_20240405_gyro_ages_X_GDR3_S19_S21_B20_with_qualityflags.csv'
        )
        fdf = pd.read_csv(
            csvpath, dtype={
                'dr3_source_id':str, 'KIC':str, 'kepid':str
            }
        )

        if sampleid == 'Santos19_Santos21_dquality':
            # return dataframe of stars only
            return fdf

        else:
            assert sampleid == 'koi_X_S19S21dquality'

        df, paramdict, st_ages = get_age_results(
            whichtype='gyro', COMPARE_AGE_UNCS=0,
            drop_grazing=drop_grazing,
            drop_highruwe=drop_highruwe
        )
        return df


    if sampleid in ['deprecated_all', 'deprecated_sel_2s']:
        # 864 rows: step0, +requiring consisting rotation period measurements
        csvpath = join(
            RESULTSDIR, "koi_gyro_posteriors_20230110",
            "step0_koi_gyro_ages_X_GDR3_B20_S19_S21_M14_M15.csv"
        )

        # merge against the sel_2s sample.
        # NOTE TODO: might want to just check the entire sample once this is fully
        # automated.  but for now, this is where the good science is at.
        kdf = pd.read_csv(csvpath)
        if sampleid == 'deprecated_all':
            pass
        elif sampleid == 'deprecated_sel_2s':
            sel_2s = kdf['median'] + kdf['+2sigma'] < 1000
            kdf = kdf[sel_2s]

        Prots = kdf['mean_period']
        from gyrojo.prot_uncertainties import get_empirical_prot_uncertainties
        Prot_errs = get_empirical_prot_uncertainties(np.array(Prots))

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

    # odd edge case; the "primary" was observed for 730sec, this one is
    # only 60sec and causes the fitter to fail.  (and i think it's the same
    # star)
    mjdf = mjdf[mjdf['name'] != 'CK03377B']

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
    li_paths = []
    for globstr in globstrs:
        glob_result = glob(globstr)
        if len(glob_result) > 0:
            li_paths.append(glob_result[0])
        else:
            raise NotImplementedError(f'failing for {globstr}')
    for li_path in li_paths:
        assert os.path.exists(li_path)
    li_df = pd.concat([pd.read_csv(f) for f in li_paths])
    li_df['kepoi_name'] = kepoi_names

    assert len(li_df) == len(mjdf)

    mldf = li_df.merge(mjdf, on='kepoi_name', how='left')

    return mldf



def get_age_results(whichtype='gyro', COMPARE_AGE_UNCS=0, drop_grazing=1,
                    drop_highruwe=1, manual_includes=None):
    """
    Get age results for the planet hosts.

    "gyro" results are from Santos19_Santos21_dquality, stars for
    which gyro is applicable, and planets (by default) from the
    cumulative-KOI table, which are "OK" planet candidates.

    "gyro_li" includes both gyro and lithium results, separately.

    "joint" results are currently defunct
    """

    assert whichtype in ['joint', 'gyro', 'li', 'gyro_li']

    if whichtype == 'joint':

        raise DeprecationWarning(
            'joint age posteriors need to be assessed start-to-finish'
        )
        raise NotImplementedError

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

    elif 'gyro' in whichtype:

        kic_df = get_gyro_data('Santos19_Santos21_dquality')
        # REQUIRE "flag_is_gyro_applicable"; change what this means
        # based on the kwargs.
        if drop_highruwe:
            sel = (kic_df['flag_is_gyro_applicable'])
            skic_df = kic_df[sel]
        else:
            sel = select_by_quality_bits(
                kic_df,
                [0, 1, 2, 3, 4, 5, 6, 8, 9],  # leaving high ruwe...
                [0, 0, 0, 0, 0, 0, 0, 0, 0]
            )
            kic_df['flag_is_gyro_applicable'] = sel
            sel = (kic_df['flag_is_gyro_applicable'])

        if isinstance(manual_includes, list):
            for m in manual_includes:
                sel |= kic_df.KIC.astype(str).str.contains(m)

        skic_df = kic_df[sel]

        skic_df['KIC'] = skic_df['KIC'].astype(str)

        # parent sample age distribution
        st_ages = 1e6*nparr(skic_df['gyro_median'])

        koi_df = get_koi_data('cumulative-KOI', drop_grazing=drop_grazing)
        koi_df['kepid'] = koi_df['kepid'].astype(str)
        # REQUIRE "flag_is_ok_planetcand"
        skoi_df = koi_df[koi_df['flag_is_ok_planetcand']]

        df = skoi_df.merge(skic_df, how='inner', left_on='kepid',
                           right_on='KIC', suffixes=('','_KIC'))

        for N in [1,2,3]:
            if N == 1:
                df[f'adopted_age_+{N}sigmapct'] = df[f'gyro_+{N}sigmapct']
                df[f'adopted_age_-{N}sigmapct'] = df[f'gyro_-{N}sigmapct']
            df['adopted_age_median'] = df['gyro_median']
            df[f'adopted_age_+{N}sigma'] = df[f'gyro_+{N}sigma']
            df[f'adopted_age_-{N}sigma'] = df[f'gyro_-{N}sigma']

    if whichtype == 'gyro_li':

        # made by plot_process_koi_li_posteriors.py
        li_method = 'eagles'
        datestr = '20240405'
        outdir = join(
            RESULTSDIR, f"koi_lithium_posteriors_{li_method}_{datestr}"
        )
        csvpath = join(
            outdir, f"{li_method}_koi_lithium_ages_X_S19S21_dquality.csv"
        )

        # here, if eagles, take single lithium ages for all finite cases.
        li_df = pd.read_csv(csvpath)
        li_df = li_df.sort_values(by='li_median')
        li_df['kepid'] = li_df.kepid.astype(str)
        li_df = li_df.drop_duplicates(subset='kepid', keep='first')
        df['kepid'] = df.kepid.astype(str)
        selcols = [
            c for c in li_df if 'li_' in c or c == 'kepid'
        ]
        _df = df.merge(
            li_df[selcols], how='left', on='kepid'
        )
        assert len(_df) == len(df)
        df = deepcopy(_df)


    GET_BERGER20_RADII = 0
    GET_PETIGURA22_RADII = 0
    GET_BESTGUESS_RADII = 1
    assert np.sum([
        GET_BESTGUESS_RADII, GET_BERGER20_RADII, GET_PETIGURA22_RADII
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

    elif GET_BESTGUESS_RADII or GET_PETIGURA22_RADII:

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

        if GET_BESTGUESS_RADII:
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

    if GET_BESTGUESS_RADII and COMPARE_AGE_UNCS:

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
    a_age_pcterr1 = df['adopted_age_+1sigmapct']
    a_age_pcterr2 = df['adopted_age_-1sigmapct']

    df['pl_name'] = df['kepler_name']
    _sel = pd.isnull(df['pl_name'])
    df.loc[_sel, 'pl_name'] = df.loc[_sel, 'kepoi_name']
    a_pl_name = df['pl_name']

    mes = df['koi_max_mult_ev']

    adopted_Teff = df['adopted_Teff']

    paramdict = {}
    paramdict['adopted_Teff'] = nparr(adopted_Teff)
    paramdict['rp'] = nparr(a_rp)
    paramdict['rp_err1'] = nparr(a_rp_err1)
    paramdict['rp_err2'] = nparr(a_rp_err2)
    paramdict['period'] = nparr(a_period)
    paramdict['age'] = nparr(a_age)
    paramdict['age_err1'] = nparr(a_age_err1)
    paramdict['age_err2'] = nparr(a_age_err2)
    paramdict['age_pcterr1'] = nparr(a_age_pcterr1)
    paramdict['age_pcterr2'] = nparr(a_age_pcterr2)
    paramdict['pl_name'] = nparr(a_pl_name)
    paramdict['mes'] = nparr(mes)

    df['adopted_rp'] = a_rp
    df['adopted_period'] = a_period

    return df, paramdict, st_ages


def get_kicstar_data(sampleid):
    """
    Get Kepler field star Prot, Teff, and stellar information.  (including gyro
    ages!)

    Args:
        sampleid (str):  most common will be "Santos19_Santos21_all", which
        concatenates Santos19 and Santos21, and then crossmatches against
        Berger20 (tables1&2).  Adopted Teffs and adopted loggs are then
        assigned in a rank-ordered preference scheme, as are period
        uncertainties.  Other options include "Santos19_Santos21_clean0",
        "Santos19_Santos21_logg", "Santos19_Santos21_dquality".  These latter
        options impose a posteriori cuts on the returned dataframe (not the
        computed one).

        Santos19_Santos21_logg:
            sel &= df.logg > 4.2
        Santos19_Santos21_clean0:
            sel &= df.logg > 4.2
            sel &= (
                df.Sph >= 500
            )
            not_CP_CB = pd.isnull(df.s21_flag1) & pd.isnull(df.s19_flag1)
            sel &= not_CP_CB
        Santos19_Santos21_dquality:
            Returns a more sophisticated data quality flagging scheme than
            either the logg or "clean0" cuts above, implemented in
            construct_field_star_gyro_quality_flags.py.
            Has flags for:
                - subgiants
                - photometric binaries
                - ruwe outliers
                - crowding
                - non-single-stars
                - "CP/CB candidates"
            and it defines:
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

    Returns:
        dataframe matching the requested sampleid
    """

    assert sampleid in ['Santos19_Santos21_all', 'Santos19_Santos21_dquality']
    # 'Santos19_Santos21_clean0', 'Santos19_Santos21_logg' both
    # deprecated

    csvpath = join(DATADIR, 'interim', 'S19_S21_KOIbonus_merged_X_GDR3_X_B20.csv')

    if sampleid == 'Santos19_Santos21_dquality':
        # made by construct_field_star_gyro_quality_flags.py driver
        csvpath = join(
            TABLEDIR,
            'field_gyro_posteriors_20240405_gyro_ages_X_GDR3_S19_S21_B20_with_qualityflags.csv'
        )
        assert os.path.exists(csvpath)
        df = pd.read_csv(
            csvpath, dtype={
                'dr3_source_id':str, 'KIC':str, 'kepid':str
            }
        )
        cols = ['median', 'peak', 'mean', '+1sigma', '-1sigma', '+2sigma',
                '-2sigma', '+3sigma', '-3sigma', '+1sigmapct', '-1sigmapct']
        for c in cols:
            df = df.rename({c: f'gyro_{c}'}, axis='columns')
        return df

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
        _df['KIC'] = _df.KIC.astype(str)

        # Add the "KOI supplement list"
        _csvpath = join(
            DATADIR, "literature", "Santos_privcomm_KOIs_Porb_Prot.csv"
        )
        bonusdf = pd.read_csv(_csvpath, dtype={'KIC':str})
        bonusdf = bonusdf[~bonusdf.KIC.isin(_df.KIC)]
        _selcols = "KIC,Prot".split(",")
        bonusdf = bonusdf[_selcols]
        for c in selcols:
            if c not in bonusdf:
                bonusdf[c] = np.nan
        bonusdf['Provenance'] = 'SantosPrivComm'

        _df = pd.concat((
            s19_df[selcols], s21_df[selcols], bonusdf[selcols])
        )

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
        df = pd.read_csv(
            csvpath, dtype={
                'dr3_source_id':str, 'KIC':str, 'kepid':str
            }
        )

    #############
    # get Teffs #
    #############

    # default Teffs
    df['adopted_Teff'] = df['b20t2_Teff']
    df['adopted_Teff_provenance'] = 'Berger2020_table2'
    df['adopted_Teff_err'] = np.nanmean([
        np.array(df['b20t2_E_Teff']),
        np.array(np.abs(df['b20t2_e_Teff']))
    ], axis=0)

    ## else, take Gaia DR3 Teff and logg, and assume σ_Teff = 200 K
    #_sel = pd.isnull(df['adopted_Teff'])
    #df.loc[_sel, 'adopted_Teff'] = df.loc[_sel, 'dr3_teff_gspphot']
    #df.loc[_sel, 'adopted_Teff_err'] = 200
    #df.loc[_sel, 'adopted_Teff_provenance'] = 'Gaia DR3 GSP-Phot'

    # else, take Santos+19 or Santos+21 Teff and logg, which are
    # mostly Mathur+17 (DR25) in this case.
    _sel = pd.isnull(df['adopted_Teff'])
    df.loc[_sel, 'adopted_Teff'] = df.loc[_sel, 'Teff']
    df.loc[_sel, 'adopted_Teff_err'] = df.loc[_sel, 'e_Teff']
    df.loc[_sel, 'adopted_Teff_provenance'] = df.loc[_sel, 'Provenance']

    assert np.sum(pd.isnull(df['adopted_Teff'])) == 0
    assert np.sum(pd.isnull(df['adopted_Teff_err'])) == 0

    #############
    # get loggs #
    #############

    # default Teffs
    df['adopted_logg'] = df['b20t2_logg']
    df['adopted_logg_provenance'] = 'Berger2020_table2'
    df['adopted_logg_err'] = np.nanmean([
        np.array(df['b20t2_E_logg']),
        np.array(np.abs(df['b20t2_e_logg']))
    ], axis=0)

    ## else, take Gaia DR3 Teff and logg, assume σ_logg = 0.3 dex
    #_sel = pd.isnull(df['adopted_logg'])
    #df.loc[_sel, 'adopted_logg'] = df.loc[_sel, 'dr3_logg_gspphot']
    #df.loc[_sel, 'adopted_logg_err'] = 0.3
    #df.loc[_sel, 'adopted_logg_provenance'] = 'Gaia DR3 GSP-Phot'

    _sel = pd.isnull(df['adopted_logg'])
    df.loc[_sel, 'adopted_logg'] = df.loc[_sel, 'logg']
    df.loc[_sel, 'adopted_logg_err'] = df.loc[_sel, 'e_logg']
    df.loc[_sel, 'adopted_logg_provenance'] = df.loc[_sel, 'Provenance']

    assert np.sum(pd.isnull(df['adopted_logg'])) == 0
    assert np.sum(pd.isnull(df['adopted_logg_err'])) == 0

    ######################
    # get prots and errs #
    ######################

    Prots = df['Prot']
    from gyrojo.prot_uncertainties import get_empirical_prot_uncertainties
    Prot_errs = get_empirical_prot_uncertainties(np.array(Prots))

    df['Prot'] = Prots
    df['Prot_err'] = Prot_errs

    assert np.sum(pd.isnull(df['Prot'])) == 0
    assert np.sum(pd.isnull(df['Prot_err'])) == 0

    ##################################
    # Return the requested subsample #
    ##################################

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
        #       Note (G1): Flag as follows:
        #               1 = main-sequence or subgiant solar-like targets in
        #                       DR25 Mathur+ (2017, J/ApJS/229/30) and Berger+ (2020, J/AJ/159/280);
        #               2 = main-sequence or subgiant solar-like targets only in
        #                       DR25 Mathur et al. (2017, J/ApJS/229/30);
        #               3 = main-sequence or subgiant solar-like targets only in
        #                       DR25 Berger et al. (2020, J/AJ/159/280).
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
        #       Flag as follows:
        #           0 = confirmed;
        #           1 = candidate;
        #           2 = false positive. 
        #
        # s21_flag5: stellar property source flag
        #   
        #       Flag as follows:
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
        #
        #   Counter({0.0: 14056, 2.0: 799, 1.0: 191})
        #   ...rest nan

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

    cgk_df['count_n_gaia_nbhr'] = n_gaia_nbhr.astype(int)

    okcols = ['kepid', 'nconfp', 'nkoi', 'ntce', 'jmag', 'hmag', 'kmag',
              'planet?', 'count_n_gaia_nbhr']
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
               'nkoi', 'ntce', 'planet?', 'count_n_gaia_nbhr']
    # "cleaned" Gaia-Kepler dataframe.  duplicates accounted for; dud columns
    # dropped (most columns kept).
    cgk_df = cgk_df[selcols]
    cgk_df['dr3_source_id'] = cgk_df['dr3_source_id'].astype(str)
    assert np.all(~pd.isnull(cgk_df.dr3_source_id))

    return cgk_df


def get_cleaned_gaiadr3_X_kepler_supplemented_dataframe():
    """
    as in get_cleaned_gaiadr3_X_kepler_dataframe (the Kepler-Gaia DR3
    crossmatch), but supplemented with Berger+20 parameters, and a few other
    odds and ends.
    """

    cachecsv = join(DATADIR, "interim", "kic_X_dr3_supp.csv")
    if os.path.exists(cachecsv):
        return pd.read_csv(cachecsv)

    cgk_df = get_cleaned_gaiadr3_X_kepler_dataframe()

    from cdips.utils.gaiaqueries import apparent_to_absolute_mag
    cgk_df['M_G'] = apparent_to_absolute_mag(
        cgk_df.dr3_phot_g_mean_mag, cgk_df.dr3_parallax
    )

    from astroquery.vizier import Vizier
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

    # start merging!
    mdf0 = left_merge(cgk_df, b20t1_df, 'kepid', 'b20t1_KIC')
    mdf1 = left_merge(mdf0, b20t2_df, 'kepid', 'b20t2_KIC')
    df = deepcopy(mdf1)

    #############
    # get Teffs #
    #############

    # default Teffs
    df['adopted_Teff'] = df['b20t2_Teff']
    df['adopted_Teff_provenance'] = 'Berger2020_table2'
    df['adopted_Teff_err'] = np.nanmean([
        np.array(df['b20t2_E_Teff']),
        np.array(np.abs(df['b20t2_e_Teff']))
    ], axis=0)

    ## else, take Gaia DR3 Teff and logg, and assume σ_Teff = 200 K
    #_sel = pd.isnull(df['adopted_Teff'])
    #df.loc[_sel, 'adopted_Teff'] = df.loc[_sel, 'dr3_teff_gspphot']
    #df.loc[_sel, 'adopted_Teff_err'] = 200
    #df.loc[_sel, 'adopted_Teff_provenance'] = 'Gaia DR3 GSP-Phot'

    #  # (NOTE: can't really do this in KIC case?)
    #  # else, take Santos+19 or Santos+21 Teff and logg, which are
    #  # mostly Mathur+17 (DR25) in this case.
    #  _sel = pd.isnull(df['adopted_Teff'])
    #  df.loc[_sel, 'adopted_Teff'] = df.loc[_sel, 'Teff']
    #  df.loc[_sel, 'adopted_Teff_err'] = df.loc[_sel, 'e_Teff']
    #  df.loc[_sel, 'adopted_Teff_provenance'] = df.loc[_sel, 'Provenance']

    #  assert np.sum(pd.isnull(df['adopted_Teff'])) == 0
    #  assert np.sum(pd.isnull(df['adopted_Teff_err'])) == 0

    #############
    # get loggs #
    #############

    # default loggs
    df['adopted_logg'] = df['b20t2_logg']
    df['adopted_logg_provenance'] = 'Berger2020_table2'
    df['adopted_logg_err'] = np.nanmean([
        np.array(df['b20t2_E_logg']),
        np.array(np.abs(df['b20t2_e_logg']))
    ], axis=0)

    #    ## else, take Gaia DR3 Teff and logg, assume σ_logg = 0.3 dex
    #    #_sel = pd.isnull(df['adopted_logg'])
    #    #df.loc[_sel, 'adopted_logg'] = df.loc[_sel, 'dr3_logg_gspphot']
    #    #df.loc[_sel, 'adopted_logg_err'] = 0.3
    #    #df.loc[_sel, 'adopted_logg_provenance'] = 'Gaia DR3 GSP-Phot'

    #    _sel = pd.isnull(df['adopted_logg'])
    #    df.loc[_sel, 'adopted_logg'] = df.loc[_sel, 'logg']
    #    df.loc[_sel, 'adopted_logg_err'] = df.loc[_sel, 'e_logg']
    #    df.loc[_sel, 'adopted_logg_provenance'] = df.loc[_sel, 'Provenance']

    df.to_csv(cachecsv, index=False)


    return df



def get_koi_data(sampleid, drop_grazing=1):
    """
    Get the KOI tables from the NASA exoplanet archive -- either the
    cumulative KOI table, or the homogeneous DR25 table.

    Supplement them with specific flags of interest.

    https://exoplanetarchive.ipac.caltech.edu/docs/PurposeOfKOITable.html
    """

    assert sampleid in ['cumulative-KOI', 'DR25-KOI']

    if sampleid == 'cumulative-KOI':
        # note: the cumulative table evolves!
        koipath = os.path.join(
            DATADIR, 'raw',
            'cumulative_2023.06.06_10.54.24.csv'
        )

    if sampleid == 'DR25-KOI':
        koipath = os.path.join(
            DATADIR, 'raw',
            'q1_q17_dr25_koi_2023.06.06_10.54.36.csv'
        )

    koi_df = pd.read_csv(koipath, comment='#', sep=',')

    koi_df['flag_koi_is_fp'] = (
        (koi_df.koi_disposition == "FALSE POSITIVE")
    ).astype(bool)

    # NOTE: Petigura+22 and Petigura+20 describe how, since b is nearly
    # unconstrained, T/Tmax,circ is a better way to perform this cut.  However
    # computing Tmax,circ requires rho_star,iso which is not readily available.
    koi_df['flag_koi_is_grazing'] = (
        koi_df.koi_impact > 0.8
    ).astype(bool)

    koi_df['flag_koi_is_low_snr'] = (
        (koi_df.koi_max_mult_ev < 10)
        |
        (pd.isnull(koi_df.koi_max_mult_ev))
    ).astype(bool)

    flag_is_ok_planetcand = (
        (~koi_df['flag_koi_is_fp'])
        &
        (~koi_df['flag_koi_is_low_snr'])
    )
    if drop_grazing:
        flag_is_ok_planetcand &= (~koi_df['flag_koi_is_grazing'])

    koi_df['flag_is_ok_planetcand'] = flag_is_ok_planetcand

    return koi_df


def select_by_quality_bits(df, bit_positions, target_values):
    """
    Selects data points from a DataFrame based on the values of specific bits in a column.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the 'flag_gyro_quality' column.
        bit_positions (list): A list of bit positions to check (0-indexed).
        target_values (list): A list of target values (0 or 1) for each bit position.

    Returns:
        pandas.Series: A boolean mask indicating the selected data points.

    Example:
        # Select data points where bit 4 is False (0) and bit 5 is False (0)
        mask = select_data_by_bits(df, [4, 5], [0, 0])
        selected_data = df[mask]
    """
    if len(bit_positions) != len(target_values):
        raise ValueError("The lengths of bit_positions and target_values must be the same.")

    data = df['flag_gyro_quality'].to_numpy()
    mask = np.ones(len(data), dtype=bool)

    for bit, target in zip(bit_positions, target_values):
        bit_mask = 1 << bit
        bit_values = (data & bit_mask) >> bit
        mask &= (bit_values == target)

    return mask
