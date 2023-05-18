"""
    | get_gyro_data
    | get_li_data
    | get_joint_results
"""
import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from os.path import join
from glob import glob
from numpy import array as nparr
from agetools.paths import DATADIR, RESULTSDIR

from astropy.io import fits
from astropy.table import Table

def get_gyro_data(sampleid):

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



def prepend_colstr(colstr, df):
    # prepend a string, `colstr`, to all columns in a dataframe
    return df.rename(
        {c:colstr+c for c in df.columns}, axis='columns'
    )



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
