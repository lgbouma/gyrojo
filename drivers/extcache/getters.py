"""
    | get_nea_data
    | get_auxiliary_data
"""
import pandas as pd, numpy as np
from astropy import units as u
from os.path import join
from copy import deepcopy
from collections import Counter

from cdips.utils.catalogs import get_nasa_exoplanet_archive_pscomppars

# This "VER" string caches the NASA exoplanet archive `ps` table at a
# particular date, in the YYYYMMDD format.

#VER = '20230111' # could be today_YYYYMMDD()
VER = '20240415'

def get_nea_data(VER, colorbyage, yvalue='Rp', preciseradii=0,
                 requirerealmass=1, NEAselfn='anyyoung'):
    """
    Get and clean exoplanet parameter data from the NASA exoplanet archive.
    """

    #
    # Columns are described at
    # https://exoplanetarchive.ipac.caltech.edu/docs/API_exoplanet_columns.html
    #
    ea_df = get_nasa_exoplanet_archive_pscomppars(VER)

    #
    # In most cases, we will select systems with finite ages (has a value, and
    # +/- error bar). We may also want to select on "is transiting", "has
    # mass", etc.
    #
    has_rp_value = ~pd.isnull(ea_df['pl_rade'])
    has_rp_errs  = (~pd.isnull(ea_df['pl_radeerr1'])) & (~pd.isnull(['pl_radeerr2']))

    has_mp_value = ~pd.isnull(ea_df['pl_bmasse'])
    has_mp_errs = (~pd.isnull(ea_df['pl_bmasseerr1'])) & (~pd.isnull(ea_df['pl_bmasseerr2']))

    rp_gt_0 = (ea_df['pl_rade'] > 0)
    mp_gt_0 = (ea_df['pl_bmassj'] > 0)

    transits = (ea_df['tran_flag']==1)
    rvflag = (ea_df['rv_flag']==1)

    has_age_value = ~pd.isnull(ea_df['st_age'])
    has_age_errs  = (~pd.isnull(ea_df['st_ageerr1'])) & (~pd.isnull(ea_df['st_ageerr2']))

    # We will be coloring the points by their ages.
    if yvalue == 'Rp':
        sel = (
            has_age_value & has_rp_value & has_rp_errs & transits & rp_gt_0
        )

        if preciseradii:

            has_factorN_radius = (
                (ea_df['pl_rade'] / np.abs(ea_df['pl_radeerr1']) >= 10)
                &
                (ea_df['pl_rade'] / np.abs(ea_df['pl_radeerr2']) >= 10)
            )
            sel &= has_factorN_radius

        # Show these planets, because they have good ages, but the NASA
        # exoplanet archive has bad entries for them up.
        EXCEPTIONS = (
            ea_df.pl_name.str.contains('TOI-451')
            |
            ea_df.pl_name.str.contains('HD 114082')
            |
            ea_df.pl_name.str.contains('Kepler-1627')
        )

        sel |= EXCEPTIONS

        # Do not show these planets.
        HARDEXCLUDE = (
            # this is an isochrone age? G9V? wat?
            (ea_df.pl_name == 'CoRoT-18 b')
            |
            # age uncertainties understated.
            (ea_df.pl_name == 'Qatar-3 b')
            |
            # age uncertainties understated.
            (ea_df.pl_name == 'Qatar-5 b')
            |
            # age uncs understated. Rotn OK. Li is not. isochrones idk.
            (ea_df.pl_name == 'Qatar-4 b')
            |
            # massive upper limit on age. why is median quoted at 20myr.
            (ea_df.pl_name == 'WASP-25 b')
            |
            # wasp-14b: 0.75+/-0.25 gyr reported (...)
            (ea_df.pl_name == 'WASP-14 b')
            |
            # kepler-411: Sun+19 decided to use Barnes07...
            (ea_df.pl_name.str.contains('Kepler-411'))
            #|
            ## wasp-189b: 0.73+/-0.13 gyr reported, teff=8000K.  Lendl+20, OK.
            #(ea_df.pl_name == 'WASP-189 b')
        )

        sel &= (~HARDEXCLUDE)

        # Hard overwrite HD 109833 (TOI 1097) age to the average of 27Myr (LCC
        # subgroup from Wood+23
        # https://ui.adsabs.harvard.edu/abs/2022arXiv221203266W/abstract) and
        # 100 Myr, the more conservative upper limit based on the properties of
        # the actual star.
        # i.e., write its age as (100+27)/2 = 63.5Myr, and assume +/-20Myr
        # uncertainties for inclusion.
        # NOTE that even this is an oversimplification.  There are two separate
        # solutions for the age posterior -- it's in a sense bimodal.  Both are
        # <100 Myr.
        _sel = ea_df.pl_name.str.contains("HD 109833")
        ea_df.loc[_sel, 'st_age'] = 0.0635
        ea_df.loc[_sel, 'st_ageerr1'] = 0.02
        ea_df.loc[_sel, 'st_ageerr2'] = -0.02

        # Shrink HD 114082 age uncertainty (LCC) to make the factor of three cut..
        _sel = ea_df.pl_name.str.contains("HD 114082")
        ea_df.loc[_sel, 'st_ageerr1'] = 0.004
        ea_df.loc[_sel, 'st_ageerr2'] = -0.004

        # Kepler-1627: correct size and age based on Bouma+2022
        _sel = ea_df.pl_name.str.contains("Kepler-1627")
        ea_df.loc[_sel, 'st_age'] = 0.036
        ea_df.loc[_sel, 'st_ageerr1'] = 0.010
        ea_df.loc[_sel, 'st_ageerr2'] = -0.008
        ea_df.loc[_sel, 'pl_rade'] = 3.85
        ea_df.loc[_sel, 'pl_radeerr1'] = 0.11
        ea_df.loc[_sel, 'pl_radeerr2'] = 0.11

        # these are just two very long period planets that are in the way of
        # the legend - simpler than fixing background to inset axes because
        # matplotlib is hard to work with
        CONVENIENCEEXCLUDE = (ea_df.pl_name == "Kepler-452 b") | (ea_df.pl_name == "Kepler-62 f")

        sel &= (~CONVENIENCEEXCLUDE)

    elif yvalue == 'Mp':
        if requirerealmass:
            sel = (
                has_mp_value & mp_gt_0 & (
                    rvflag
                )
                # has_rp_value & has_rp_errs & transits & rp_gt_0
            )
        else:
            sel = (
                has_mp_value & mp_gt_0
            )

    # Impose the initial selection function defined above: select stars
    # with finite ages, sizes, size unceratinties, that transit and have
    # radii greater than zero.  (While including EXCEPTIONS and excluding
    # HARDEXCLUDE and CONVENIENCEEXCLUDE).
    sdf = ea_df[sel]

    # Read parameters.
    mp = sdf['pl_bmasse']
    rp = sdf['pl_rade']
    rp_err1 = sdf['pl_radeerr1']
    rp_err2 = sdf['pl_radeerr2']
    age = sdf['st_age']*1e9
    age_err1 = sdf['st_ageerr1']*1e9
    age_err2 = sdf['st_ageerr2']*1e9
    period = sdf['pl_orbper']
    discoverymethod = sdf['discoverymethod']
    disc_pubdate = sdf['disc_pubdate']
    pl_name = sdf['pl_name']

    # Does the age have S/N > 3?
    has_factorN_age = (
        (sdf['st_age'] / np.abs(sdf['st_ageerr1']) >= 3)
        &
        (sdf['st_age'] / np.abs(sdf['st_ageerr2']) >= 3)
    )

    # set0 := Old ages
    s0 = age > 1e9 #((age > 1e9) | (~has_factorN_age))

    if NEAselfn == 'anyyoung':
        # set1 := younger than 1 Gyr & SNR>3... OR <1Gyr at 2sigma.
        s1 = (
            #(age <= 1e9) & (has_factorN_age)
            #|
            (age + 2*age_err1 < 1e9)
        )
    elif NEAselfn == 'strictyoung':
        # set1 := younger than 1 Gyr & SNR>3
        s1 = (
            (age <= 1e9) & (has_factorN_age)
        )

    # (nb that set0 and set1 are not exact complements)

    verbose = 0
    if verbose:

        # Print stuff about the young transiting planet sample.

        sdf_young = (
            sdf[has_factorN_age &
                (sdf['st_age']*u.Gyr < 0.5*u.Gyr) &
                (sdf['st_age']*u.Gyr > 0*u.Gyr)]
        )
        scols = ['pl_name', 'pl_rade', 'pl_orbper', 'st_age', 'st_ageerr1',
                 'st_ageerr2']
        print(42*'-')
        print('Age less than 0.5 Gyr, S/N>3')
        print(sdf_young[scols].sort_values(by='st_age'))
        print(42*'-')
        sdf_young_hj = (
            sdf[(sdf['st_age']*u.Gyr < 0.5*u.Gyr) &
                (sdf['st_age']*u.Gyr > 0*u.Gyr) &
                (sdf['pl_rade'] > 8) &
                (sdf['pl_orbper'] < 10) &
                has_factorN_age
               ]
        )
        scols = ['pl_name', 'pl_rade', 'pl_orbper', 'st_age', 'st_ageerr1',
                 'st_ageerr2']
        print('Age less than 0.5 Gyr, S/N>3, Rp>7Re, P<10d')
        print(sdf_young_hj[scols].sort_values(by='st_age'))
        print(42*'-')
        sdf_young2 = sdf[s1]
        pd.options.display.max_rows = 5000
        _sel = ~sdf_young2.pl_name.isin(sdf_young.pl_name)
        print('Age (<1 Gyr & S/N>3) OR (age+2*σt <1 Gyr) -- ADDITIONS')
        print(sdf_young2[_sel][scols].sort_values(by='st_age'))


    paramdict = {}
    paramdict['mp'] = mp
    paramdict['rp'] = rp
    paramdict['rp_err1'] = rp_err1
    paramdict['rp_err2'] = rp_err2
    paramdict['age'] = age
    paramdict['age_err1'] = age_err1
    paramdict['age_err2'] = age_err2
    paramdict['period'] = period
    paramdict['discoverymethod'] = discoverymethod
    paramdict['disc_pubdate'] = disc_pubdate
    paramdict['pl_name'] = pl_name

    return paramdict, ea_df, sdf, sel, s0, s1


def get_auxiliary_data(showauxiliarysample, nograzingnoruwe = False):
    # NOTE: if you have some other sample to add, it can go here.
    # Just needs to yield a_pl_name, a_period, a_rp, a_age

    if 'gyro' in showauxiliarysample:
        from gyrojo.paths import TABLEDIR, PAPERDIR
        from gyrojo.getters import select_by_quality_bits
        # made by make_gyroonly_table.py
        # generally, require grazing to not be okay, and to drop high
        # ruwe cases.  this selection function is for gyro ages only.
        csvpath = (join(
            PAPERDIR,
            "table_allageinfo_agesformatted.csv"
        ))
        aux_df = pd.read_csv(csvpath)

        if nograzingnoruwe:
            sel_star = select_by_quality_bits(
                aux_df,
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # drop high ruwe...
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            )
            sel_pl = (aux_df.flag_planet_quality.astype(str) == '0')
        else:
            sel_star = select_by_quality_bits(
                aux_df,
                [0, 1, 2, 3, 4, 5, 6, 8, 9, 10],  # drop high ruwe...
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            )
            sel_pl = (
                (aux_df.flag_planet_quality.astype(str) == '0')
                |
                (aux_df.flag_planet_quality.astype(str) == '4')
            )

        if showauxiliarysample == 'gyro_sel2s':

            sel = sel_star & sel_pl & (aux_df['gyro_median'] + aux_df['gyro_+2sigma'] < 1000)

        if showauxiliarysample == 'gyro_selsnr3':
            sel = sel_star & sel_pl & (
                (aux_df['gyro_median'] /
                 np.abs(aux_df['gyro_+1sigma']) > 3)
                &
                (aux_df['gyro_median'] /
                 np.abs(aux_df['gyro_-1sigma']) > 3)
                &
                (aux_df['gyro_median'] < 1000)
            )
        if showauxiliarysample == 'gyro_selsnrupper3':
            sel = (
                (aux_df['gyro_median'] / aux_df['gyro_+1sigma'] > 3)
                &
                (aux_df['gyro_median'] < 1000)
            )
        if showauxiliarysample == 'gyro_anyyoung':
            sel = sel_star & sel_pl & (
                aux_df['gyro_median'] + aux_df['gyro_+2sigma'] < 1000
            )

        a_period = aux_df.loc[sel, 'adopted_period']
        a_rp = aux_df.loc[sel, 'adopted_rp']
        a_rp_err1 = np.zeros(len(aux_df)) # aux_df.loc[sel, 'koi_prad_err1']
        a_rp_err2 = np.zeros(len(aux_df)) # aux_df.loc[sel, 'koi_prad_err2']
        a_age = 1e6*(aux_df.loc[sel, 'gyro_median'])
        a_age_err1 = 1e6*(aux_df.loc[sel, 'gyro_+1sigma'])
        a_age_err2 = 1e6*(aux_df.loc[sel, 'gyro_-1sigma'])
        a_starquality = aux_df.loc[sel, 'flag_gyro_quality']
        a_planetquality = aux_df.loc[sel, 'flag_planet_quality']

        aux_df['pl_name'] = aux_df['kepler_name']
        _sel = pd.isnull(aux_df['pl_name'])
        aux_df.loc[_sel, 'pl_name'] = aux_df.loc[_sel, 'kepoi_name']
        a_pl_name = aux_df.loc[sel, 'pl_name']

        auxparamdict = {}
        auxparamdict['rp'] = a_rp
        auxparamdict['rp_err1'] = a_rp_err1
        auxparamdict['rp_err2'] = a_rp_err2
        auxparamdict['period'] = a_period
        auxparamdict['age'] = a_age
        auxparamdict['age_err1'] = a_age_err1
        auxparamdict['age_err2'] = a_age_err2
        auxparamdict['pl_name'] = a_pl_name
        auxparamdict['starquality'] = a_starquality
        auxparamdict['planetquality'] = a_planetquality

        GET_PLANET_TYPE_COUNTS = 1
        if GET_PLANET_TYPE_COUNTS and showauxiliarysample == 'gyro_anyyoung':

            seldf = aux_df[sel]

            from gyrojo.plotting import get_planet_class_labels
            from gyrojo.papertools import update_latex_key_value_pair as ulkvp
            from gyrojo.papertools import int_to_string

            OFFSET = 0
            seldf = get_planet_class_labels(
                seldf, OFFSET=OFFSET, rpkey='adopted_rp',
                periodkey='adopted_period'
            )

            plclasses = [
                'Mini-Neptunes',
                'Sub-Saturns',
                'Super-Earths',
                'Earths',
                'Jupiters'
            ]
            r = Counter(seldf['pl_class'])
            for plclass in plclasses:
                n = int(r[plclass])
                ckey = plclass.lower().replace("-","")
                ulkvp(f'n{ckey}highq', int_to_string(n))

            nlongperiod = len(seldf[seldf['adopted_period'] > 50])
            ulkvp(f'nlongperiodhighq', int_to_string(nlongperiod))


    if showauxiliarysample in ['joint_sel2s', 'joint_sel2gyr']:
        raise NotImplementedError
        # output from calc_koi_joint_posteriors.py
        if showauxiliarysample == 'joint_sel2s':
            csvpath = (
                '/Users/luke/Dropbox/proj/young-KOIs/results/'
                'koi_gyro_X_lithium_posteriors_20230116/'
                'sel_2s_merged_joint_age_posteriors.csv'
            )
            aux_df = pd.read_csv(csvpath)
            sel = np.ones(len(aux_df)).astype(bool)

        elif showauxiliarysample == 'joint_sel2gyr':
            csvpath = (
                '/Users/luke/Dropbox/proj/young-KOIs/results/'
                'koi_gyro_X_lithium_posteriors_20230205/'
                'all_merged_joint_age_posteriors.csv'
            )
            aux_df = pd.read_csv(csvpath)

            # otherwise, miss one interesting object for which joint-median is
            # nan: Kepler-1939 b, 800 myr, 1Rearth
            _sel = pd.isnull(aux_df['joint_median'])
            aux_df.loc[_sel, 'joint_median'] = aux_df.loc[_sel, 'gyro_median']
            aux_df.loc[_sel, 'joint_+1sigma'] = aux_df.loc[_sel, 'gyro_+1sigma']
            aux_df.loc[_sel, 'joint_-1sigma'] = aux_df.loc[_sel, 'gyro_-1sigma']

            sel = (
                (
                    (
                    (aux_df['joint_median'] / aux_df['joint_+1sigma'] > 2)
                    &
                    (aux_df['joint_median'] / aux_df['joint_-1sigma'] > 2)
                    )
                    |
                    (
                        aux_df['joint_median'] + aux_df['joint_+2sigma'] < 1000
                    )
                )
                &
                (aux_df['joint_median'] < 2000)
            )

        GET_BERGER20_RADII = True

        if GET_BERGER20_RADII:
            # as a quick-look -- pull the Berger+20 radii.
            d0 = '/Users/luke/Dropbox/proj/young-KOIs/data/literature'
            p0 = 'Berger_2020_AJ_160_108_table1_planet_radii.fits'
            fitspath = join(d0, p0)
            from astropy.io import fits
            from astropy.table import Table
            hl = fits.open(fitspath)
            bdf = Table(hl[1].data).to_pandas()

            bdf['kepoi_name'] = (
                bdf.KOI.astype(str).apply(
                    lambda x: "K"+str(x[:-2]).zfill(6)+str(x[-2:])
                )
            )

            aux_df = aux_df.merge(bdf, how='left', on='kepoi_name')

            #sel &= aux_df.kepoi_name.isin(bdf.kepoi_name)

            # if berger radius is null, take KOI
            _sel = pd.isnull(aux_df['Radius'])
            N_null_0 = len(aux_df[_sel])
            aux_df.loc[_sel, 'Radius'] = aux_df.loc[_sel, 'koi_prad']
            aux_df.loc[_sel, 'E_Radius'] = aux_df.loc[_sel, 'koi_prad_err1']
            aux_df.loc[_sel, 'e_radius_lc'] = aux_df.loc[_sel, 'koi_prad_err2']
            N_null_1 = len(aux_df[pd.isnull(aux_df['Radius'])])
            print(N_null_0)
            print(N_null_1)

            a_rp = aux_df.loc[sel, 'Radius']
            a_rp_err1 = aux_df.loc[sel, 'E_Radius']
            a_rp_err2 = aux_df.loc[sel, 'e_radius_lc']

        else:
            # just the KOI radii.  crappy uncertainties.
            a_rp = aux_df.loc[sel, 'koi_prad']
            a_rp_err1 = aux_df.loc[sel, 'koi_prad_err1'] # upper
            a_rp_err2 = aux_df.loc[sel, 'koi_prad_err2'] # lower

        a_period = aux_df.loc[sel, 'koi_period']

        a_age = 1e6*(aux_df.loc[sel, 'joint_median'])
        a_age_err1 = 1e6*(aux_df.loc[sel, 'joint_+1sigma'])
        a_age_err2 = 1e6*(aux_df.loc[sel, 'joint_-1sigma'])

        aux_df['pl_name'] = aux_df['kepler_name']
        _sel = pd.isnull(aux_df['pl_name'])
        aux_df.loc[_sel, 'pl_name'] = aux_df.loc[_sel, 'kepoi_name']
        a_pl_name = aux_df.loc[sel, 'pl_name']

        auxparamdict = {}
        auxparamdict['rp'] = a_rp
        auxparamdict['rp_err1'] = a_rp_err1
        auxparamdict['rp_err2'] = a_rp_err2
        auxparamdict['period'] = a_period
        auxparamdict['age'] = a_age
        auxparamdict['age_err1'] = a_age_err1
        auxparamdict['age_err2'] = a_age_err2
        auxparamdict['pl_name'] = a_pl_name

    return auxparamdict
