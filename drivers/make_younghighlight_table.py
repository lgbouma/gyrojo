import os
from os.path import join
from copy import deepcopy
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from gyrojo.getters import (
    get_gyro_data, get_age_results, get_koi_data
)
from gyrojo.papertools import update_latex_key_value_pair as ulkvp
from gyrojo.paths import PAPERDIR, DATADIR

def format_lowerlimit(value):
    if pd.isna(value):
        return np.nan
    else:
        return f"$> {int(value)}$"

def cast_to_int_string(value):
    if pd.isna(value):
        return np.nan
    else:
        return str(int(value))

def replace_nan_string(value):
    if pd.isna(value) or 'nan' in str(value).lower():
        return '--'
    else:
        return value

def make_table(
    # ...with age results
    drop_highruwe = 0,
    grazing_is_ok = 1,
    #manual_includes = [
    #    '10736489', # KIC10736489 = KOI-7368: adopted_logg = 4.448...teff=5068.2... cutoff 4.48
    #    '8873450', # KOI-7913 binary
    #    '9471268', # Kepler-326 700myr multi
    #],
    SELECT_YOUNG = 0,
):

    _df, _, _ = get_age_results(
        whichtype='allageinfo', grazing_is_ok=grazing_is_ok,
        drop_highruwe=drop_highruwe#, manual_includes=manual_includes
    )
    _ldf = pd.read_csv(join(
        DATADIR, 'interim', 'koi_jump_getter_koi_X_JUMP.csv'
    ))
    _df['kepid'] = _df.kepid.astype(str)
    _ldf['kepid'] = _ldf.kepid.astype(str)
    _df['has_hires'] = _df.kepid.isin(_ldf.kepid)
    _df['li_eagles_limlo_formatted'] = 10**(_df['li_eagles_limlo'])/(1e6) # 2σ lower limit in myr
    _df['li_eagles_limlo_formatted'] = _df['li_eagles_limlo_formatted'].apply(format_lowerlimit)

    _df['li_eagles_limlo_forsort'] = 10**(_df['li_eagles_limlo'])/(1e6) # 2σ lower limit in myr
    _df.loc[_df['li_eagles_limlo_forsort']<0, 'li_eagles_limlo_forsort'] = 9999

    gyro = np.array(_df.gyro_median)
    gyro[pd.isnull(gyro)] = np.nan
    li = np.array(_df.li_median)
    li[_df['li_eagles_limlo'] > 0] = np.nan # force nan li ages for actual upper limits
    li[pd.isnull(li)] = np.nan
    _df['min_age'] = np.nanmin([gyro, li], axis=0)

    df = deepcopy(_df)
    if SELECT_YOUNG:
        sel = (_df['min_age'] <= 1000)
        N = len(_df[sel])
        df = _df[sel]

    selcols = (
        "kepoi_name,kepler_name,koi_disposition,"
        "min_age,"
        "adopted_Teff,Prot,li_eagles_LiEW,li_eagles_eLiEW,"
        "gyro_median,gyro_+1sigma,gyro_-1sigma,"
        "li_median,li_+1sigma,li_-1sigma,li_eagles_limlo,li_eagles_limlo_formatted,"
        "li_eagles_limlo_forsort,"
        "adopted_rp,adopted_period,"
        "flag_dr3_ruwe_outlier,flag_koi_is_grazing,flag_gyro_quality,has_hires"
    ).split(",")
    pdf = df[selcols].sort_values(
        by=['koi_disposition','min_age','li_eagles_limlo_forsort','kepler_name','kepoi_name'],
        ascending=[False, True, True, True, True]
    ).reset_index(drop=1)

    for c in pdf.columns:
        if 'gyro_' in c or 'adopted_Teff' in c:
            pdf[c] = pdf[c].apply(cast_to_int_string)
        if c == 'adopted_period' or c == 'Prot':
            pdf[c] = np.round(pdf[c], 2)
        if 'flag' in c or 'has_hires' in c:
            pdf[c] = pdf[c].apply(cast_to_int_string)
        if c.startswith('li_') and c not in [
            'li_eagles_limlo', 'li_eagles_limlo_formatted'
        ]:
            pdf[c] = pdf[c].apply(cast_to_int_string)

    pcols = (
        "kepoi_name,kepler_name,gyro_median,"
        "gyro_+1sigma,gyro_-1sigma,li_median,li_+1sigma,li_-1sigma,"
        "adopted_rp,adopted_period,"
        "flag_dr3_ruwe_outlier,flag_koi_is_grazing,"
        "flag_gyro_quality,has_hires"
    ).split(",")

    print(42*'-')
    print(pdf[pdf.koi_disposition == 'CONFIRMED'][pcols])
    print(42*'-')
    print(pdf[pdf.koi_disposition == 'CANDIDATE'][pcols])
    print(42*'~')
    print('\n')
    print(pdf[(pdf.koi_disposition == 'CONFIRMED') & (pdf.has_hires == 0)][pcols])
    print(10*'-')
    print(pdf[(pdf.koi_disposition == 'CANDIDATE') & (pdf.has_hires == 0)][pcols])
    print('\n')
    print(42*'~')

    # quote gyro ages
    pdf['t_gyro'] = pdf.apply(
        lambda row:
        "$"+
        f"{row['gyro_median']}"+
        "^{+"+
        f"{row['gyro_+1sigma']}"+
        "}_{-"+
        f"{row['gyro_-1sigma']}"
        +"} $",
        axis=1
    )
    pdf['t_gyro'] = pdf['t_gyro'].apply(replace_nan_string)

    # quote lithium ages
    pdf['t_li'] = pdf.apply(
        lambda row:
        "$"+
        f"{row['li_median']}"+
        "^{+"+
        f"{row['li_+1sigma']}"+
        "}_{-"+
        f"{row['li_-1sigma']}"
        +"}$",
        axis=1
    )
    pdf['t_li'] = pdf['t_li'].apply(replace_nan_string)

    _sel = (pdf.li_eagles_limlo > 0)
    pdf.loc[_sel, 't_li'] = pdf.loc[_sel, 'li_eagles_limlo_formatted']


    # report Li EW
    pdf['Li_EW'] = pdf.apply(
        lambda row:
        "$"+
        f"{row['li_eagles_LiEW']}"+
        "\pm"+
        f"{row['li_eagles_eLiEW']}"+
        "$",
        axis=1
    )
    pdf['Li_EW'] = pdf['Li_EW'].apply(replace_nan_string)


    # Drop the original age columns
    pdf = pdf.drop(
        columns=['gyro_median', 'gyro_+1sigma', 'gyro_-1sigma', 'li_median',
                 'li_+1sigma', 'li_-1sigma', "li_eagles_LiEW", "li_eagles_eLiEW"]
    )

    # These columns will be written.
    mapdict = {
        'kepoi_name': "KOI",
        "kepler_name": "Kepler",
        "adopted_Teff": r"$T_{\rm eff}$",
        "Prot": "Prot",
        "Li_EW": "Li_EW",
        "t_gyro": r"$t_{\rm gyro}$",
        "t_li": r"$t_{\rm Li}$",
        "adopted_rp": r"$R_{\rm p}$",
        "adopted_period": r"$P$",
        "flag_dr3_ruwe_outlier": r"$f_{\rm RUWE}$",
        "flag_koi_is_grazing": r"$f_{\rm grazing}$",
        "flag_gyro_quality": r"$Q_{\rm gyro}$",
        "has_hires": r"Spec?",
    }
    rounddict = {
        'adopted_period': 2,
        'adopted_rp': 2,
        'Prot': 2,
    }
    formatters = {}
    for k,v in rounddict.items():
        formatters[mapdict[k]] = lambda x: np.round(x, v)

    pdf = pdf.rename(mapdict, axis='columns')

    if SELECT_YOUNG:
        sel = (pdf.koi_disposition == 'CONFIRMED')
        latexpath1 = join(PAPERDIR, 'table_subgyr_confirmed.tex')
        pdf[sel][mapdict.values()].to_latex(
            latexpath1, index=False, na_rep='--', escape=False, formatters=formatters
        )

        sel = (pdf.koi_disposition == 'CANDIDATE')
        latexpath2 = join(PAPERDIR, 'table_subgyr_candidate.tex')
        t2 = pdf[sel][mapdict.values()].to_latex(
            latexpath2, index=False, na_rep='--', escape=False, formatters=formatters
        )

        # trim them..
        for texpath in [latexpath1, latexpath2]:
            with open(texpath, 'r') as f:
                texlines = f.readlines()
            texlines = texlines[4:-2] # trim header and footer
            with open(texpath, 'w') as f:
                f.writelines(texlines)
            print(f"Wrote {texpath}")

    pdf = pdf.drop(columns=['li_eagles_limlo', 'li_eagles_limlo_formatted',
                            'li_eagles_limlo_forsort', 'min_age'])

    if not SELECT_YOUNG:

        ulkvp('nnonfopkoissomeageinfo', len(pdf))

        csvpath = join(PAPERDIR, 'table_allageinfo.csv')
        pdf.to_csv(csvpath, index=False)
        print(f'Wrote {csvpath}')


if __name__ == "__main__":
    make_table(SELECT_YOUNG = 0)
    make_table(SELECT_YOUNG = 1)
