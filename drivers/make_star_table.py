"""
Trim down the "Santos19_Santos21_dquality" gyro table, which has a long list of
columns, to something CSV and TeX manageable.

Available cols:

    ['gyro_median', 'gyro_peak', 'gyro_mean', 'gyro_+1sigma', 'gyro_-1sigma',
    'gyro_+2sigma', 'gyro_-2sigma', 'gyro_+3sigma', 'gyro_-3sigma',
    'gyro_+1sigmapct', 'gyro_-1sigmapct', 'KIC', 'Kpmag', 'Q', 'Teff', 'E_Teff',
    'e_Teff', 'logg', 'E_logg', 'e_logg', 'Mass', 'E_Mass', 'e_Mass', 'Prot',
    'E_Prot', 'Sph', 'E_Sph', 'Provenance', 's19_flag1', 's19_flag2', 's19_flag3',
    's19_flag4', 's19_flag5', 's21_flag1', 's21_flag2', 's21_flag3', 's21_flag4',
    's21_flag5', 'b20t1_recno', 'b20t1_KIC', 'b20t1_gmag', 'b20t1_e_gmag',
    'b20t1_Ksmag', 'b20t1_e_Ksmag', 'b20t1_plx', 'b20t1_e_plx', 'b20t1___Fe_H_',
    'b20t1_e__Fe_H_', 'b20t1_RUWE', 'b20t1_Ncomp', 'b20t1_KsCorr', 'b20t1_State',
    'b20t1_output', 'b20t1_KSPC', 'b20t1__RA', 'b20t1__DE', 'b20t2_recno',
    'b20t2_KIC', 'b20t2_Mass', 'b20t2_E_Mass', 'b20t2_e_Mass', 'b20t2_Teff',
    'b20t2_E_Teff', 'b20t2_e_Teff', 'b20t2_logg', 'b20t2_E_logg', 'b20t2_e_logg',
    'b20t2___Fe_H_', 'b20t2_E__Fe_H_', 'b20t2_e__Fe_H_', 'b20t2_Rad',
    'b20t2_E_Rad', 'b20t2_e_Rad', 'b20t2_rho', 'b20t2_E_rho', 'b20t2_e_rho',
    'b20t2_Lum', 'b20t2_E_Lum', 'b20t2_e_Lum', 'b20t2_Age', 'b20t2_f_Age',
    'b20t2_E_Age', 'b20t2_e_Age', 'b20t2_Dist', 'b20t2_E_Dist', 'b20t2_e_Dist',
    'b20t2_Avmag', 'b20t2_GOF', 'b20t2_TAMS', 'kepid', 'dr3_source_id', 'dr3_ra',
    'dr3_dec', 'dr3_parallax', 'dr3_parallax_error', 'dr3_parallax_over_error',
    'dr3_pmra', 'dr3_pmra_error', 'dr3_pmdec', 'dr3_pmdec_error', 'dr3_ruwe',
    'dr3_phot_g_mean_flux_over_error', 'dr3_phot_g_mean_mag',
    'dr3_phot_bp_mean_flux_over_error', 'dr3_phot_bp_mean_mag',
    'dr3_phot_rp_mean_flux_over_error', 'dr3_phot_rp_mean_mag',
    'dr3_phot_bp_rp_excess_factor', 'dr3_bp_rp', 'dr3_radial_velocity',
    'dr3_radial_velocity_error', 'dr3_rv_nb_transits', 'dr3_rv_renormalised_gof',
    'dr3_rv_chisq_pvalue', 'dr3_phot_variable_flag', 'dr3_l', 'dr3_b',
    'dr3_non_single_star', 'dr3_teff_gspphot', 'dr3_logg_gspphot',
    'dr3_mh_gspphot', 'dr3_distance_gspphot', 'dr3_ag_gspphot',
    'dr3_ebpminrp_gspphot', 'dr3_kepler_gaia_ang_dist', 'dr3_pm_corrected',
    'nconfp', 'nkoi', 'ntce', 'planet?', 'count_n_gaia_nbhr', 'adopted_Teff',
    'adopted_Teff_provenance', 'adopted_Teff_err', 'adopted_logg',
    'adopted_logg_provenance', 'adopted_logg_err', 'Prot_err', 'M_G',
    'flag_dr3_ruwe_outlier', 'flag_kepler_gaia_ang_dist', 'flag_Teffrange',
    'flag_logg', 'flag_dr3_M_G', 'flag_is_CP_CB', 'flag_kepler_gaia_xmambiguous',
    'flag_in_KEBC', 'flag_dr3_non_single_star', 'nbhr_count', 'flag_dr3_crowding',
    'flag_farfrommainsequence', 'b20t2_rel_E_Age', 'b20t2_rel_e_Age',
    'flag_gyro_quality', 'flag_is_gyro_applicable']
"""
import os
from os.path import join
from copy import deepcopy
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from gyrojo.getters import (
    get_gyro_data, get_age_results, get_koi_data
)
from gyrojo.papertools import update_latex_key_value_pair as ulkvp
from gyrojo.papertools import (
    format_lowerlimit, cast_to_int_string, replace_nan_string, format_prot_err
)

from gyrojo.paths import PAPERDIR, DATADIR, TABLEDIR

def make_star_table(
):

    df = get_gyro_data("Santos19_Santos21_dquality")

    selcols = (
        "KIC,dr3_source_id,"
        "gyro_median,gyro_+1sigma,gyro_-1sigma,"
        "adopted_Teff,adopted_Teff_err,adopted_Teff_provenance,"
        "Prot,Prot_err,Prot_provenance,"
        "flag_gyro_quality"
    ).split(",")
    #TODO FIXME probably split latex columns from the csv ones...

    sdf = df[selcols].sort_values(
        by=['gyro_median','KIC'],
        ascending=[True, True]
    ).reset_index(drop=1)

    impose_floor = 0
    if impose_floor:
        # impose 10% gyro uncertainty floor
        rel_punc = sdf['gyro_+1sigma'].astype(float) / sdf['gyro_median'].astype(float)
        rel_munc = sdf['gyro_-1sigma'].astype(float) / sdf['gyro_median'].astype(float)
        FLOOR = 0.1
        sel = rel_punc < FLOOR
        sdf.loc[sel, 'gyro_+1sigma'] = sdf.loc[sel, 'gyro_median'] * FLOOR
        sel = rel_munc < FLOOR
        sdf.loc[sel, 'gyro_-1sigma'] = sdf.loc[sel, 'gyro_median'] * FLOOR

    for c in sdf.columns:
        if 'gyro_' in c or c == 'adopted_Teff':
            sdf[c] = sdf[c].apply(cast_to_int_string)
        if c == 'adopted_period' or c == 'Prot':
            sdf[c] = np.round(sdf[c], 2)
        if 'flag' in c or 'has_hires' in c:
            sdf[c] = sdf[c].apply(cast_to_int_string)
        if c.startswith('li_') and c not in [
            'li_eagles_limlo', 'li_eagles_limlo_formatted'
        ]:
            sdf[c] = sdf[c].apply(cast_to_int_string)

    # quote gyro ages
    sdf['t_gyro'] = sdf.apply(
        lambda row:
        "$"+
        f"{row['gyro_median']}"+
        "^{+"+
        f"{row['gyro_+1sigma']}"+
        "}_{-"+
        f"{row['gyro_-1sigma']}"
        +"}$",
        axis=1
    )
    sdf['t_gyro'] = sdf['t_gyro'].apply(replace_nan_string)

    # These columns will be written to CSV:
    ###    "KIC,dr3_source_id,"
    ###    "gyro_median,gyro_+1sigma,gyro_-1sigma,"
    ###    "adopted_Teff,adopted_Teff_err,adopted_Teff_provenance,"
    ###    "Prot,Prot_err,"
    ###    "flag_gyro_quality"

    mapdict = {
        'KIC': 'KIC',
        'dr3_source_id': 'dr3_source_id',
        "adopted_Teff": "adopted_Teff",
        "Prot": "Prot",
        "t_gyro": "t_gyro",
        "flag_gyro_quality": "flag_gyro_quality",
    }
    rounddict = {
        'Prot': 2,
    }
    formatters = {}
    for k,v in rounddict.items():
        formatters[mapdict[k]] = lambda x: np.round(x, v)

    sdf = sdf.rename(mapdict, axis='columns')

    # trim out all columns except those above...
    _sdf = sdf[mapdict.values()]

    latexpath = join(PAPERDIR, 'table_star_gyro.tex')
    np.random.seed(123)

    # format ages
    sel = sdf.gyro_median.astype(float) > 4000
    _sdf.loc[sel, 't_gyro'] = '$> 4000$'
    _sdf.sample(n=10).to_latex(
        latexpath, index=False, na_rep='--', escape=False, formatters=formatters
    )

    # trim them..
    for texpath in [latexpath]:
        with open(texpath, 'r') as f:
            texlines = f.readlines()
        texlines = texlines[4:-2] # trim header and footer
        with open(texpath, 'w') as f:
            f.writelines(texlines)
        print(f"Wrote {texpath}")

    sdf['adopted_Teff_err'] = sdf['adopted_Teff_err'].apply(cast_to_int_string)

    sdf['Prot_err'] = sdf.apply(format_prot_err, axis=1)
    sdf['Prot'] = sdf['Prot'].apply(
        lambda x: f"{x:.3f}" if x <= 5 else f"{x:.2f}"
    )
    csvpath = join(TABLEDIR, 'table_star_gyro.csv')

    from make_younghighlight_table import BIBCODEDICT
    for column in sdf.columns:
        if sdf[column].dtype == 'object':  # Check if the column is of string type
            for key, value in BIBCODEDICT.items():
                sdf[column] = sdf[column].str.replace(key, value)

    sdf.to_csv(csvpath, index=False)
    print(f'Wrote {csvpath}')

    csvpath = join(PAPERDIR, 'table_star_gyro_agesformatted.csv')
    # format ages
    sel = sdf.gyro_median.astype(float) > 4000
    sdf.loc[sel, 't_gyro'] = '$> 4000$'
    sdf.loc[sel, 'gyro_median'] = np.nan
    sdf.loc[sel, 'gyro_+1sigma'] = np.nan
    sdf.loc[sel, 'gyro_-1sigma'] = np.nan
    sdf.to_csv(csvpath, index=False, na_rep='')
    print(f'Wrote {csvpath}')

    csvpath = join(TABLEDIR, 'table_star_gyro_allcols.csv')
    odf = df.sort_values(
        by=['gyro_median','KIC'],
        ascending=[True, True]
    ).reset_index(drop=1)
    odf.to_csv(csvpath, index=False)
    print(f'Wrote {csvpath}')

if __name__ == "__main__":
    make_star_table()
