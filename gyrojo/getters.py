"""
Getters:
    | get_kicstar_data
    | get_gyro_data
    | get_li_data
    | get_age_results
    | get_cleaned_gaiadr3_X_kepler_dataframe
    | get_cleaned_gaiadr3_X_kepler_supplemented_dataframe
    | get_koi_data
    | get_prot_metacatalog
Selector:
    | select_by_quality_bits
"""
import os
from copy import deepcopy
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from os.path import join
from glob import glob
from numpy import array as nparr
from astropy import units as u

from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1

from astropy.io import fits
from astropy.table import Table

from gyrointerp.helpers import prepend_colstr, left_merge

from gyrojo.paths import LOCALDIR, DATADIR, RESULTSDIR, TABLEDIR
from gyrojo.papertools import update_latex_key_value_pair as ulkvp


def get_gyro_data(sampleid, koisampleid='cumulative-KOI',
                  grazing_is_ok=0, drop_highruwe=1):
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

        grazing_is_ok (bool): whether you want "flag_is_ok_planetcand"
        to drop grazing planets, or not.  The "flag_koi_is_grazing"
        will be true no matter what.
    """

    assert sampleid in [
        'Santos19_Santos21_dquality',
        'koi_X_S19S21dquality',
        'McQ14_dquality'
    ]

    if sampleid == 'Santos19_Santos21_dquality':
        # made by construct_field_star_gyro_quality_flags.py driver
        csvpath = join(
            TABLEDIR,
            'field_gyro_posteriors_20240530_gyro_ages_X_GDR3_S19_S21_B20_with_qualityflags.csv'
        )
        fdf = pd.read_csv(
            csvpath, dtype={
                'dr3_source_id':str, 'KIC':str, 'kepid':str
            }
        )
        # return dataframe of stars only
        return fdf

    elif sampleid == 'McQ14_dquality':
        # made by construct_field_star_gyro_quality_flags.py driver
        csvpath = join(
            TABLEDIR,
            'field_gyro_posteriors_McQ14_20240613_gyro_ages_X_GDR3_S19_S21_B20_with_qualityflags.csv'
        )
        fdf = pd.read_csv(
            csvpath, dtype={
                'dr3_source_id':str, 'KIC':str, 'kepid':str
            }
        )
        # return dataframe of stars only
        return fdf

    elif sampleid == 'koi_X_S19S21dquality':

        df, paramdict, st_ages = get_age_results(
            whichtype='gyro', COMPARE_AGE_UNCS=0,
            grazing_is_ok=grazing_is_ok,
            drop_highruwe=drop_highruwe
        )
        return df

    else:
        raise NotImplementedError



def get_li_data(sampleid, whichwindowlen=7.5):

    assert sampleid in ['koi_X_S19S21dquality', 'koi_X_JUMP']

    from cdips.utils.mamajek import get_interp_BmV_from_Teff

    # Get B-V colors for stars with HIRES data available.  Take statistical
    # uncertainties to be propagated from the uncertainties in Teff.
    # (Systematic uncertainties with the conversion currently unknown, but
    # certainly important for any mildly evolved stars.)
    csvpath = join(DATADIR, "interim", f"koi_jump_getter_{sampleid}.csv")
    mjdf = pd.read_csv(csvpath)

    #TODO FIXME: many nan teffs bc this is all kepler... not just kois...

    # rotators (...& w/ the S19S21 teffs)
    sdf1 = get_kicstar_data('Santos19_Santos21_litsupp_all')
    # all KIC (w/ only B21 teffs)
    sdf2 = get_cleaned_gaiadr3_X_kepler_supplemented_dataframe()

    mjdf['kepid'] = mjdf.kepid.astype(str)
    sdf1['kepid'] = sdf1.kepid.astype(str)
    sdf2['kepid'] = sdf2.kepid.astype(str)

    selcols = ('kepid,adopted_Teff,adopted_Teff_err,adopted_Teff_provenance,'
               'dr3_source_id,b20t1_recno'.split(","))
    _df0 = mjdf.merge(sdf1[selcols], on='kepid', how='left')
    _df1 = _df0.merge(sdf2[selcols], on='kepid', how='left')

    teffkeys = ['adopted_Teff','adopted_Teff_err','adopted_Teff_provenance']
    for teffkey in teffkeys:
        _df1[teffkey] = _df1[f'{teffkey}_x'].fillna(_df1[f'{teffkey}_y'])

    _sel = pd.isnull(_df1['adopted_Teff'])
    _df1.loc[_sel, 'adopted_Teff'] = _df1.loc[_sel, 'koi_steff']
    _df1['mean_koi_steff_err'] = np.nanmean([
        np.array(np.abs(_df1['koi_steff_err1'])),
        np.array(np.abs(_df1['koi_steff_err2']))
    ], axis=0)
    _df1.loc[_sel, 'adopted_Teff_err'] = _df1.loc[_sel, 'mean_koi_steff_err']
    _df1.loc[_sel, 'adopted_Teff_provenance'] = 'KOI Q1-Q17 Stellar Properties Catalog'

    # Above provenance hierarchy:
    # Preferred is Berger+21
    # Next is Santos19/21 if available
    # Final is KOI SPC if available
    # require Teff never-NaN, because here it is used for calculating Li ages
    assert np.sum(pd.isnull(_df1.adopted_Teff)) == 0

    mjdf = deepcopy(_df1)

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
        f"{name}_{spec_id}_Li_EW_deltawav{whichwindowlen}_xshift*_results.csv")
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


def get_age_results(whichtype='gyro', COMPARE_AGE_UNCS=0,
                    grazing_is_ok=0, drop_highruwe=1,
                    manual_includes=None):
    """
    Get age results for the planet hosts.

    "gyro" results are from Santos19_Santos21_dquality, stars for
        which gyro is applicable, and planets (by default) from the
        cumulative-KOI table, which are "OK" planet candidates.

    "gyro_li" all "flag_is_ok_planetcand" and "flag_is_gyro_applicable" KOIs,
        **with gyro results**, and optionally lithium results.

    "allageinfo": all "flag_is_ok_planetcand" KOIs, with any available
        rotation-based or lithium-based age information.
    """

    assert whichtype in ['allageinfo', 'gyro', 'gyro_li']

    #
    # Get KOI list... rotating star list... and lithium list...
    #
    koi_df = get_koi_data('cumulative-KOI', grazing_is_ok=grazing_is_ok)
    koi_df['kepid'] = koi_df['kepid'].astype(str)

    # rotators... with Berger+S19+S21 teffs...
    kicrot_df = get_gyro_data('Santos19_Santos21_dquality')
    kicrot_df['KIC'] = kicrot_df['KIC'].astype(str)

    # made by plot_process_koi_li_posteriors.py
    li_method = 'eagles'
    lidatestr = '20240405'
    outdir = join(RESULTSDIR, f"koi_lithium_posteriors_{li_method}_{lidatestr}")
    csvpath = join( outdir, f"{li_method}_koi_X_JUMP_lithium_ages.csv" )
    li_df = pd.read_csv(csvpath)
    li_df = li_df.sort_values(by='li_eagles_lMed') # eagles median...
    li_df['kepid'] = li_df.kepid.astype(str)
    li_df = li_df.drop_duplicates(subset='kepid', keep='first')

    # REQUIRE "flag_is_ok_planetcand"; change what this means based on
    # "grazing_is_ok" kwarg.
    skoi_df = koi_df[koi_df['flag_is_ok_planetcand']]

    # If "gyro" or "gyro_li", then REQUIRE "flag_is_gyro_applicable";
    # change what this means based on the kwargs.  Otherwise, (for
    # "allageinfo") just note whether or not gyro is supposedly applicable.
    if drop_highruwe:
        sel = select_by_quality_bits(
            kicrot_df,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # drop high ruwe...
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )
        kicrot_df['flag_is_gyro_applicable'] = sel
        sel = (kicrot_df['flag_is_gyro_applicable'])
    else:
        sel = select_by_quality_bits(
            kicrot_df,
            [0, 1, 2, 3, 4, 5, 6, 8, 9, 10],  # leaving high ruwe...
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )
        kicrot_df['flag_is_gyro_applicable'] = sel
        sel = (kicrot_df['flag_is_gyro_applicable'])

    if isinstance(manual_includes, list):
        for m in manual_includes:
            sel |= kicrot_df.KIC.astype(str).str.contains(m)

    st_ages = None

    if 'gyro' in whichtype:

        # For "gyro" and "gyro_li", return the inner product of the "gyro-ok"
        # KOIs with the KIC information.  (In other words, 'gyro_li' requires a
        # rotation period for the stars).

        skicrot_df = kicrot_df[sel]

        # parent sample age distribution
        st_ages = 1e6*nparr(skicrot_df['gyro_median'])

        df = skoi_df.merge(skicrot_df, how='inner', left_on='kepid',
                           right_on='KIC', suffixes=('','_KIC'))

    if whichtype == 'gyro_li':

        # here, if eagles, take single lithium ages for all finite cases.
        df['kepid'] = df.kepid.astype(str)
        selcols = [ c for c in li_df if 'li_' in c or c == 'kepid' ]
        _df = df.merge(li_df[selcols], how='left', on='kepid')
        assert len(_df) == len(df)
        df = deepcopy(_df)

    if whichtype == 'allageinfo':

        # For "allageinfo", you want the "flag_is_ok_planetcand" KOIs, with any
        # available rotation-based or lithium-based age information.
        # (i.e. no cuts on whether or not gyro is ok).

        # gyro-applicable stars
        rot_df = skoi_df.merge(kicrot_df, how='inner', left_on='kepid',
                               right_on='KIC', suffixes=('','_KIC'))
        rot_df = rot_df.drop_duplicates(subset='kepid', keep='first')

        rcols = [c for c in rot_df.columns if not c.startswith('koi_') and not
                 c.startswith('b20t') and not c.startswith('dr3_')]
        licols = [c for c in li_df.columns if not c.startswith('koi_') and not
                  c.startswith('b20t') and not c.startswith('dr3_') and not
                  c.startswith('flag_')]

        mdf0 = skoi_df.merge(rot_df[rcols], how='left', left_on='kepid',
                             right_on='kepid', suffixes=('','_rotdf'))
        mdf1 = mdf0.merge(li_df[licols], how='left', left_on='kepid',
                          right_on='kepid', suffixes=('','_lidf'))

        assert len(mdf0) == len(skoi_df)
        assert len(mdf1) == len(skoi_df)

        df = deepcopy(mdf1)

        # Selection function is: you have at least some kind of useful age
        # information from either lithium or rotation.
        sel = (
            ~pd.isnull(df.Prot)
            |
            ~pd.isnull(df.li_eagles_LiEW)
        )
        df = df[sel]

        # for Li-only detections, adopted_Teff is not propagated because of the
        # left-join logic above.  in such cases, draw from cgkic_df...
        sel = pd.isnull(df.adopted_Teff)
        missing_kepids = df[sel].kepid

        # all KIC, with only Berger+ teffs...
        cgkic_df = get_kicstar_data("allKIC_Berger20_dquality")
        foo = pd.DataFrame({'kepid': missing_kepids})
        selcols = ['adopted_Teff', 'adopted_Teff_err',
                   'adopted_Teff_provenance', 'adopted_logg',
                   'adopted_logg_err', 'adopted_logg_provenance',
                   'flag_gyro_quality', 'kepid']
        cgkic_df['kepid'] = cgkic_df['kepid'].astype(str)
        mfoo = foo.merge(cgkic_df[selcols], left_on='kepid', right_on='kepid', how='left')

        for c in selcols[:-1]:
            df.loc[sel, c] = np.array(mfoo[c])

        #~600->30 nan teffs.  here, correct the remainder.
        sel = pd.isnull(df.adopted_Teff)
        df.loc[sel,'adopted_Teff'] = df.loc[sel,'koi_steff']
        df.loc[sel,'adopted_Teff_provenance'] = 'Mathur_2017_DR25'

        assert pd.isnull(df.adopted_Teff).sum() == 0
        assert pd.isnull(df.adopted_Teff).sum() == 0
        assert pd.isnull(df.adopted_Teff_provenance).sum() == 0

    #
    # in all instances, use the gyro age as the adopted age...
    #
    for N in [1,2,3]:
        if N == 1:
            df[f'adopted_age_+{N}sigmapct'] = df[f'gyro_+{N}sigmapct']
            df[f'adopted_age_-{N}sigmapct'] = df[f'gyro_-{N}sigmapct']
        df['adopted_age_median'] = df['gyro_median']
        df[f'adopted_age_+{N}sigma'] = df[f'gyro_+{N}sigma']
        df[f'adopted_age_-{N}sigma'] = df[f'gyro_-{N}sigma']

    #
    # planetary radius collection.
    #
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
        a_rp_prov = np.repeat('Berger2020_AJ_160_108_t1', len(df))
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
        df['Rp_provenance'] = np.repeat('Petigura_2022_CKS_X', len(df))
        df['E_Rp'] = df['P22_E_Rp']
        df['e_Rp'] = np.abs(df['P22_e_rp_lc'])

        if GET_BESTGUESS_RADII:
            # else, take Berger+20
            _sel = pd.isnull(df['Rp'])
            df.loc[_sel, 'Rp'] = df.loc[_sel, 'B20_Radius']
            df.loc[_sel, 'Rp_provenance'] = 'Berger2020_AJ_160_108_t1'
            df.loc[_sel, 'E_Rp'] = df.loc[_sel, 'B20_E_Radius']
            df.loc[_sel, 'e_Rp'] = np.abs(df.loc[_sel, 'B20_e_radius_lc'])

            # else, take KOI radii(?)
            _sel = pd.isnull(df['Rp'])
            df.loc[_sel, 'Rp'] = df.loc[_sel, 'koi_prad']
            df.loc[_sel, 'Rp_provenance'] = 'KOI_table'
            df.loc[_sel, 'E_Rp'] = df.loc[_sel, 'koi_prad_err1']
            df.loc[_sel, 'e_Rp'] = np.abs(df.loc[_sel, 'koi_prad_err2'])

        # Kepler-447 grazing; adopt Lillo-Box+2015 size.
        # Bizarre that this thing showed no evidence for rotation
        # variability...
        _sel = df.kepoi_name == 'K01800.01'
        df.loc[_sel, 'Rp'] = (1.65*u.Rjup).to(u.Rearth).value
        df.loc[_sel, 'E_Rp'] = (0.59*u.Rjup).to(u.Rearth).value
        df.loc[_sel, 'e_Rp'] = (0.56*u.Rjup).to(u.Rearth).value

        a_rp = df['Rp']
        a_rp_prov = df['Rp_provenance']
        a_rp_err1 = df['E_Rp']
        a_rp_err2 = df['e_Rp']

    else:
        # just the KOI radii.  crappy uncertainties.
        a_rp = df['koi_prad']
        a_rp_prov = np.repeat('KOI_table', len(df))
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
    df['adopted_rp_provenance'] = a_rp_prov
    df['adopted_period'] = a_period

    df.loc[df.kepoi_name=='K01951.01','adopted_rp'] = 2.352
    df.loc[df.kepoi_name=='K01951.01','adopted_rp_provenance'] = 'Berger2018'
    df.loc[df.kepoi_name=='K07368.01','adopted_rp'] = 2.22 # source: me, 2022b
    df.loc[df.kepoi_name=='K07368.01','adopted_rp_provenance'] = 'Bouma2022b' # source: me, 2022b
    df.loc[df.kepoi_name=='K05245.01','adopted_rp'] = 3.79 # source: me, 2022a
    df.loc[df.kepoi_name=='K05245.01','adopted_rp_provenance'] = 'Bouma2022a' # source: me, 2022a

    assert df.adopted_rp.isna().sum() == 0

    return df, paramdict, st_ages


def get_kicstar_data(sampleid):
    """
    Get Kepler field star Prot, Teff, and stellar information.  (including gyro
    ages!)

    Args:
        sampleid (str): Options are: [
            'allKIC_Berger20_dquality',
            'Santos19_Santos21_litsupp_all',
            'Santos19_Santos21_all',
            'Santos19_Santos21_dquality',
            'McQuillan2014only'
        ]

        "Santos19_Santos21_all" concatenates Santos19 and Santos21 ,
        and then crossmatches against Berger20 (tables1&2).  Adopted
        Teffs and adopted loggs are then assigned in a rank-ordered
        preference scheme, as are period uncertainties.

        "Santos19_Santos21_litsupp_all" concatenates Santos19 and
        Santos21 (with the Santos+ bonus KOIs, and the David+ bonus
        KOIs), and then crossmatches against Berger20 (tables1&2).
        Adopted Teffs and adopted loggs are then assigned in a
        rank-ordered preference scheme, as are period uncertainties.

        "Santos19_Santos21_dquality" imposes a posteriori cuts on the returned
        dataframe (not the computed one).  This specifically just returns
        field_gyro_posteriors_20240530_gyro_ages_X_GDR3_S19_S21_B20_with_qualityflags.csv

        "allKIC_Berger20_dquality" which is the KIC/Berger20 stars, without any
        parsing of whether rotation is reported, with quality flags calculated.

        "McQuillan2014only" is the McQuillan2014 sample, crossmatched against
        Berger20.

    Returns:
        dataframe matching the requested sampleid
    """

    assert sampleid in [
        'Santos19_Santos21_litsupp_all',
        'Santos19_Santos21_all',
        'Santos19_Santos21_dquality',
        'allKIC_Berger20_dquality',
        'McQuillan2014only',
        'McQuillan2014only_dquality'
    ]
    # 'Santos19_Santos21_clean0', 'Santos19_Santos21_logg' both
    # deprecated

    if sampleid == 'Santos19_Santos21_litsupp_all':
        csvpath = join(DATADIR, 'interim', 'S19_S21_KOIbonus_litsupp_merged_X_GDR3_X_B20.csv')

    if sampleid == 'Santos19_Santos21_all':
        csvpath = join(DATADIR, 'interim', 'S19_S21_KOIbonus_merged_X_GDR3_X_B20.csv')

    if sampleid == 'McQuillan2014only':
        csvpath = join(DATADIR, 'interim', 'McQ14_merged_X_GDR3_X_B20.csv')

    if sampleid == 'allKIC_Berger20_dquality':
        csvpath = join(
            TABLEDIR,
            'allKIC_20240530_X_GDR3_B20_with_qualityflags.csv'
        )
        assert os.path.exists(csvpath)
        df = pd.read_csv(
            csvpath, dtype={
                'dr3_source_id':str, 'KIC':str, 'kepid':str
            }
        )
        return df

    if sampleid == 'Santos19_Santos21_dquality':
        # made by construct_field_star_gyro_quality_flags.py driver
        csvpath = join(
            TABLEDIR,
            'field_gyro_posteriors_20240530_gyro_ages_X_GDR3_S19_S21_B20_with_qualityflags.csv'
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

    if sampleid == 'McQuillan2014only_dquality':
        # made by construct_field_star_gyro_quality_flags.py driver
        csvpath = join(
            TABLEDIR,
            'field_gyro_posteriors_McQ14_20240613_gyro_ages_X_GDR3_S19_S21_B20_with_qualityflags.csv'
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
        s19_df['Prot_provenance'] = 'Santos2019'
        for ix in range(1,6):
            s19_df[f"s21_flag{ix}"] = np.nan

        # Santos+2021: G+F stars
        # https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJS/255/17
        # 'KIC', 'Kpmag', 'Q', 'Teff', 'E_Teff', 'e_Teff', 'logg', 'E_logg',
        # 'e_logg', 'Mass', 'E_Mass', 'e_Mass', 'Prot', 'E_Prot', 'Sph', 'E_Sph',
        # 'flag1', 'flag2', 'flag3', 'flag4', 'flag5', 'KCat', 'Simbad', '_RA', '_DE'
        catalogs = Vizier.get_catalogs("J/ApJS/255/17")
        s21_df = catalogs[0].to_pandas()
        s21_df['Prot_provenance'] = 'Santos2021'
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
            'Prot_provenance',
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
        bonusdf['Prot_provenance'] = 'SantosPrivComm'

        _df = pd.concat((
            s19_df[selcols], s21_df[selcols], bonusdf[selcols])
        )

        if sampleid == 'Santos19_Santos21_litsupp_all':

            # considered pulling W13, M13, M14, M15, A18, D21.  However, D21
            # already did A18, M13, M15, and W13.  **and** vetted them... But
            # did so only for the CKS sample.  Also, M14 supposedly omitted
            # KOIs.  (...At least, those known at the time).

            perioddata = {
                #'Walkowicz2013': ['wb13', "J/MNRAS/436/1883", "Per", "KIC"],
                #'McQuillan2013': ['m13', "J/ApJ/775/L11", "Prot", "KIC"],
                #'McQuillan2014': ['m14', "J/ApJS/211/24", "Prot", "KIC"],
                #'Mazeh2015': ['m15', "J/ApJ/801/3", "Prot", "KIC"],
                #'Angus2018': ['a18', "J/MNRAS/474/2094", "Per", "KOI"],
                'David2021': ['d21', "J/AJ/161/265", "Prot", "KIC"]
            }

            shortkey = 'd21'
            periodkey = "Prot"
            kicidkey = "KIC"
            _v = Vizier(columns=["**"])
            _v.ROW_LIMIT = -1
            catalogs = _v.get_catalogs("J/AJ/161/265")
            litdf = catalogs[0].to_pandas()
            litdf = prepend_colstr(f"{shortkey}_", litdf)

            # David+21 CKS stars with either "reliable" or "highly reliable"
            # periods.  If they are not already in our rotation period catalog,
            # then merge in the reported periods.  This is a small perturbation
            # on the overall *stellar* sample, but adds 178 stars to the planet
            # sample.  (10-20% increase?).  Worth the headache for
            # completeness, and to help this study really go further.
            sel = (
                (
                    (litdf.d21_f_Prot == 2)
                    |
                    (litdf.d21_f_Prot == 3)
                )
                &
                (~litdf['d21_KIC'].astype(str).isin(_df.KIC.astype(str)))
            )

            ulkvp('nnewdavidtwentyone', sel.sum())

            sdf = litdf[sel]

            # NaN "d21_Prot" column, but the David+21 period in d21_D21Per is
            # good
            selkics = [8222813, 9634821, 8416523]
            for selkic in selkics:
                _sel = sdf.d21_KIC.astype(str) == str(selkic)
                sdf.loc[_sel, 'd21_Prot'] = sdf.loc[_sel, 'd21_D21Per']
                sdf.loc[_sel, 'd21_r_Prot'] = "D21"

            _selcols = 'd21_Prot,d21_KIC,d21_r_Prot'.split(",")
            sdf = sdf[_selcols]
            renamedict = {
                'd21_Prot':"Prot",
                'd21_KIC':"KIC",
                'd21_r_Prot':"Prot_provenance"
            }
            sdf = sdf.rename(renamedict, axis='columns')
            sdf.to_csv(
                join(TABLEDIR, 'david2021_extra_koi_info.csv'), index=False
            )

            bonus_david21_koi_df = deepcopy(sdf)

            _df = pd.concat((
                s19_df[selcols], s21_df[selcols],
                bonusdf[selcols], bonus_david21_koi_df
            ))

            assert np.sum(pd.isnull(_df['Prot'])) == 0

        if sampleid == 'McQuillan2014only':
            fitspath = join(
                DATADIR, "literature", "McQuillan_2014_table1.fits"
            )
            df = Table(fits.open(fitspath)[1].data).to_pandas()
            df['Prot'] = df['Prot']
            # overwrite earlier contactenation ; this is McQuillan2014 only.
            _df = deepcopy(df)
            assert np.sum(pd.isnull(_df['Prot'])) == 0

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
    df['adopted_Teff_err'] = np.nanmax([
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
    if sampleid != 'McQuillan2014only':
        _sel = pd.isnull(df['adopted_Teff'])
        df.loc[_sel, 'adopted_Teff'] = df.loc[_sel, 'Teff']
        df.loc[_sel, 'adopted_Teff_err'] = df.loc[_sel, 'e_Teff']
        df.loc[_sel, 'adopted_Teff_provenance'] = df.loc[_sel, 'Prot_provenance']

    # else, Mathur+2017 (like four cases)
    _v = Vizier(columns=["**"])
    _v.ROW_LIMIT = -1
    catalogs = _v.get_catalogs("J/ApJS/229/30")
    m17_df = catalogs[0].to_pandas()
    m17_df.KIC = m17_df.KIC.astype(str)
    _sel = pd.isnull(df['adopted_Teff'])
    _mdf = pd.DataFrame(df.loc[_sel, 'KIC'].astype(str)).merge(
        m17_df, how='left', on='KIC'
    )
    df.loc[_sel, 'adopted_Teff'] = np.array(_mdf.Teff)
    df.loc[_sel, 'adopted_Teff_err'] = np.nanmean([
        np.array(np.abs(_mdf.E_Teff)),
        np.array(np.abs(_mdf.e_Teff))
    ])
    df.loc[_sel, 'adopted_Teff_provenance'] = "Mathur2017"

    assert np.sum(pd.isnull(df['adopted_Teff'])) == 0
    assert np.sum(pd.isnull(df['adopted_Teff_err'])) == 0

    #############
    # get loggs #
    #############

    # default loggs
    df['adopted_logg'] = df['b20t2_logg']
    df['adopted_logg_provenance'] = 'Berger2020_table2'
    df['adopted_logg_err'] = np.nanmax([
        np.array(df['b20t2_E_logg']),
        np.array(np.abs(df['b20t2_e_logg']))
    ], axis=0)

    ## else, take Gaia DR3 Teff and logg, assume σ_logg = 0.3 dex
    #_sel = pd.isnull(df['adopted_logg'])
    #df.loc[_sel, 'adopted_logg'] = df.loc[_sel, 'dr3_logg_gspphot']
    #df.loc[_sel, 'adopted_logg_err'] = 0.3
    #df.loc[_sel, 'adopted_logg_provenance'] = 'Gaia DR3 GSP-Phot'

    if sampleid != 'McQuillan2014only':
        _sel = pd.isnull(df['adopted_logg'])
        df.loc[_sel, 'adopted_logg'] = df.loc[_sel, 'logg']
        df.loc[_sel, 'adopted_logg_err'] = df.loc[_sel, 'e_logg']
        df.loc[_sel, 'adopted_logg_provenance'] = df.loc[_sel, 'Prot_provenance']

    # else, Mathur+2017 (like four cases)
    _sel = pd.isnull(df['adopted_logg'])
    _mdf = pd.DataFrame(df.loc[_sel, 'KIC'].astype(str)).merge(
        m17_df, how='left', on='KIC'
    )
    df.loc[_sel, 'adopted_logg'] = np.array(_mdf['log_g_'])
    df.loc[_sel, 'adopted_logg_err'] = np.nanmean([
        np.array(np.abs(_mdf.E_log_g_)),
        np.array(np.abs(_mdf.e_log_g_))
    ])
    df.loc[_sel, 'adopted_logg_provenance'] = "Mathur2017"

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

    # Bedell's list...
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
    df['adopted_Teff_err'] = np.nanmax([
        np.array(np.abs(df['b20t2_E_Teff'])),
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



def get_koi_data(sampleid, grazing_is_ok=0):
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
        (~(
            (koi_df.koi_disposition == "CONFIRMED")
            |
            (koi_df.koi_disposition == "CANDIDATE")
        ))
        |
        (
            koi_df.koi_pdisposition != "CANDIDATE"
        )
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

    # see description of flags in docstring
    flag_bits = {
        'flag_koi_is_fp': 0,
        'flag_koi_is_low_snr': 1,
        'flag_koi_is_grazing': 2,
    }

    # Iterate over the flag columns and update the flag_gyro_quality column
    koi_df['flag_planet_quality'] = 0
    for flag, bit_pos in flag_bits.items():
        # Convert the flag column to NumPy array and perform left-shift operation
        shifted_values = np.left_shift(koi_df[flag].astype(int).values, bit_pos)
        # Update the 'flag_gyro_quality' column using the shifted values
        koi_df['flag_planet_quality'] |= shifted_values

    # Define the mask to check bits 0 through 9 inclusive
    if grazing_is_ok:
        mask = 0b000000011
    else:
        mask = 0b000000111

    # Create the 'flag_is_ok_planetcand' column - uses bits 0,1,2 as requested
    koi_df['flag_is_ok_planetcand'] = ((koi_df['flag_planet_quality'] & mask) == 0)

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

    Example 1:
        # Select stars for which gyro is "applicable"
        mask = select_by_quality_bits(
            df, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )
        selected_df = df[mask]

    Example 2:
        # Select data points where bit 4 is False (0) and bit 5 is False (0)
        mask = select_data_by_bits(df, [4, 5], [0, 0])
        selected_df = df[mask]
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


def get_prot_metacatalog():
    """
    Get S19_S21_M14_R23_X_KIC_B20_DR3_rotation_period_metacatalog :

        Santos2019, Santos2021, McQuillan2014, Reinhold2023, left-joined
        against the KIC, Berger+2020, and Gaia DR3.

    Contains a range of rotation-related keys:
        ['r23_ProtACF', 'r23_ProtGPS', 'r23_ProtFin', 'r23_ProtMcQ14',
        'r23_ProtS21', 'm14_Prot', 'm14_e_Prot', 'm14_n_Prot', 'Prot', 'Prot_err',
        'Prot_provenance']
        where "Prot" is the S19/S21/SantosPrivComm periods, as noted in
        Prot_provenance.

    This was constructed largely b/c I didn't trust the Reinhold23 vs Santos
    crossmatch, which was the correct thing to do - some stars that do overlap
    (e.g. between McQ2014 and Santos19/21) are not listed by Reinhold.

    Useful for considering alternative periods.
    """

    outpath = join(DATADIR, "interim",
                   "S19_S21_M14_R23_X_KIC_B20_DR3_rotation_period_metacatalog.csv")
    if os.path.exists(outpath):
        return pd.read_csv(outpath)

    # Prots
    fitspath = join(DATADIR, "literature", "Reinhold_2023_tablec1_secret.fits")
    hl = fits.open(fitspath)
    r23_df = Table(hl[1].data).to_pandas()
    r23_df = prepend_colstr("r23_", r23_df)
    hl.close()

    s1921_df = get_kicstar_data("Santos19_Santos21_litsupp_all")
    sel = (
        (s1921_df.Prot_provenance == 'Santos2019')
        |
        (s1921_df.Prot_provenance == 'Santos2021')
        |
        (s1921_df.Prot_provenance == 'SantosPrivComm')
    )
    s1921_df = s1921_df[sel]

    fitspath = join(DATADIR, "literature", "McQuillan_2014_table1.fits")
    hl = fits.open(fitspath)
    m14_df = Table(hl[1].data).to_pandas()
    m14_df = prepend_colstr("m14_", m14_df)
    hl.close()

    # get Teffs (gaia-kepler + berger2020)
    gkb_df = get_cleaned_gaiadr3_X_kepler_supplemented_dataframe()

    mdf0 = left_merge(gkb_df, r23_df, 'kepid', 'r23_KIC')
    mdf1 = left_merge(mdf0, m14_df, 'kepid', 'm14_KIC')
    mdf = left_merge(
        mdf1, s1921_df[['KIC','Prot','Prot_err','Prot_provenance']],
        'kepid', 'KIC'
    )

    mdf.to_csv(outpath, index=False)
    print(f"Wrote {outpath}")

    return mdf
