import os
from numpy import array as nparr
from os.path import join
from copy import deepcopy
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from gyrojo.getters import (
    get_gyro_data, get_age_results, get_koi_data,
    select_by_quality_bits
)
from gyrojo.papertools import update_latex_key_value_pair as ulkvp
from gyrojo.papertools import (
    format_lowerlimit, cast_to_int_string, replace_nan_string, format_prot_err,
    int_to_string
)
from collections import Counter

from gyrojo.paths import PAPERDIR, DATADIR, TABLEDIR

COMMENTDICT = {
    'K05245.01': "Cep-Her",
    'K07368.01': "Cep-Her",
    'K06186.01': "Cep-Her",
    'K07913.01': "Cep-Her",
    'K03876.01': "MELANGE-3",
    'K01838.01': "MELANGE-3",
    'K01833.01': 'Theia-520',
    'K01833.02': 'Theia-520',
    'K01833.03': 'Theia-520',
    'K00775.02': 'Theia-520',
    'K00775.01': 'Theia-520',
    'K00775.03': 'Theia-520',
    'K06228.01': 'Unres. Binary',
    'K03933.01': 'Unres. Binary',
    'K01199.01': 'Mystery',
}

BIBCODEDICT = {
    'Santos2019':'2019ApJS..244...21S',
    'Santos2021':'2021ApJS..255...17S',
    'A18':'2018MNRAS.474.2094A',
    'M15':'2015ApJ...801....3M',
    'D21':'2021AJ....161..265D',
    'M13':'2013ApJ...775L..11M',
    "Berger2018": "2018ApJ...866...99B",
    'Berger2020_table2':'2020AJ....159..280B',
    'Mathur_2017_DR25':'2017ApJS..229...30M',
    'Mathur2017':'2017ApJS..229...30M',
    'Bouma2022a':'2022AJ....163..121B',
    'Bouma2022b':'2022AJ....164..215B',
    'Petigura_2022_CKS_X':'2022AJ....163..179P',
    'Berger2020_AJ_160_108_t1':'2020AJ....160..108B',
    'KOI_table':'10.26133/NEA4',
}

import numpy as np
import pandas as pd

def are_values_consistent(t_gyro, t_gyro_err_pos, t_gyro_err_neg, t_li,
                          t_li_err_pos, t_li_err_neg, spec, t_li_orig, teff):
    if int(spec) == 0:
        return "--"
    elif ((t_li != '--') and (not ">" in str(t_li_orig))) and t_gyro == "--":
        # finite two-sided t_Li and no gyro shouldn't happen
        if int(teff) >= 3800 and int(teff) <= 6200:
            return "No"
        else:
            return "--"
    elif str(t_li).startswith(">") and t_gyro != '--':
        t_li_limit = float(str(t_li)[2:])
        t_gyro_upper = t_gyro + t_gyro_err_pos
        if t_li_limit <= t_gyro_upper:
            return "Yes"
        elif t_li_limit <= t_gyro_upper + 2 * t_gyro_err_pos:
            return "Maybe"
        else:
            return "No"
    elif not pd.isna(t_gyro) and t_gyro != '--' and not pd.isna(t_li) and t_li != "--":
        t_gyro_lower = float(t_gyro) - float(t_gyro_err_neg)
        t_gyro_upper = float(t_gyro) + float(t_gyro_err_pos)
        t_li_lower = float(str(t_li).split("^")[0]) - t_li_err_neg
        t_li_upper = float(str(t_li).split("^")[0]) + t_li_err_pos
        combined_err = np.sqrt(t_gyro_err_pos**2 + t_li_err_pos**2)
        if (t_li_lower <= t_gyro_upper) and (t_gyro_lower <= t_li_upper):
            return "Yes"
        elif (abs(float(str(t_li).split("^")[0]) - t_gyro) <= 3 * combined_err):
            return "Maybe"
        else:
            return "No"
    else:
        return "--"

def extract_value_and_error(value):
    value = str(value).replace("$", "").replace("{", "").replace("}", "")
    if pd.isna(value) or value == "--" or str(value).startswith(">"):
        return value, np.nan, np.nan
    else:
        parts = value.split("^")
        main_value = float(parts[0].split("_")[0])
        error_pos = float(parts[1].split("_")[0].replace("+", ""))
        error_neg = float(parts[1].split("_")[1].replace("-", ""))
        return main_value, error_pos, error_neg


def make_table(
    # ...with age results
    drop_highruwe = 0,
    grazing_is_ok = 1,
    SELECT_YOUNG = 0,
):

    _df, _, _ = get_age_results(
        whichtype='allageinfo', grazing_is_ok=grazing_is_ok,
        drop_highruwe=drop_highruwe
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
        "kepid,kepoi_name,kepler_name,koi_disposition,"
        "min_age,"
        #"adopted_logg,"
        "adopted_Teff,adopted_Teff_err,adopted_Teff_provenance,"
        "Prot,Prot_err,Prot_provenance,li_eagles_LiEW,li_eagles_eLiEW,"
        "gyro_median,gyro_+1sigma,gyro_-1sigma,gyro_+2sigma,gyro_-2sigma,"
        "li_median,li_+1sigma,li_-1sigma,li_eagles_limlo,li_eagles_limlo_formatted,"
        "li_eagles_limlo_forsort,"
        "adopted_rp,adopted_rp_provenance,adopted_period,"
        "flag_dr3_ruwe_outlier,flag_koi_is_grazing,flag_gyro_quality,"
        "flag_planet_quality,has_hires"
    ).split(",")
    pdf = df[selcols].sort_values(
        by=['koi_disposition','min_age','li_eagles_limlo_forsort','kepler_name','kepoi_name'],
        ascending=[False, True, True, True, True]
    ).reset_index(drop=1)
    longdf = df.sort_values(
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
        "kepid,kepoi_name,kepler_name,gyro_median,"
        "gyro_+1sigma,gyro_-1sigma,li_median,li_+1sigma,li_-1sigma,"
        "adopted_rp,adopted_period,"
        "flag_dr3_ruwe_outlier,flag_koi_is_grazing,"
        "flag_gyro_quality,flag_planet_quality,has_hires"
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
        +"}$",
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
    _pdf = deepcopy(pdf)
    pdf = pdf.drop(
        columns=['gyro_median', 'gyro_+1sigma', 'gyro_-1sigma', 'li_median',
                 'li_+1sigma', 'li_-1sigma', "li_eagles_LiEW", "li_eagles_eLiEW"]
    )

    # Look for consistent cases with positive evidence from both rotation and
    # lithium.
    pdf['flag_gyro_quality'] = pdf['flag_gyro_quality'].astype(int)
    _pdf['flag_gyro_quality'] = _pdf['flag_gyro_quality'].astype(int)
    sel_gyro_quality = select_by_quality_bits(
        pdf, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    sel_gyro_weakquality = select_by_quality_bits(
        pdf, [0, 1, 2, 3, 4, 5, 6, 8, 9],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    _sel0 = (
        (pdf.min_age < 1000) &
        sel_gyro_quality &
        (pdf.flag_planet_quality=='0') &
        (pdf.koi_disposition == 'CONFIRMED') &
        (pdf.li_eagles_limlo == -1.)
    )
    _wsel0 = (
        (pdf.min_age < 1000) &
        sel_gyro_weakquality &
        (pdf.flag_planet_quality=='0') &
        ( ( pdf.koi_disposition == 'CONFIRMED') |
          (pdf.koi_disposition == 'CANDIDATE') ) &
        (pdf.li_eagles_limlo == -1.)
    )
    for k in pdf[_sel0].kepoi_name:
        if k not in COMMENTDICT:
            COMMENTDICT[k] = '\checkmark \checkmark'

    _sel1 = (
        (pdf.min_age < 1000) &
        sel_gyro_quality &
        (pdf.flag_planet_quality=='0') &
        (pdf.koi_disposition == 'CONFIRMED') &
        (pdf.li_eagles_limlo > -1.)
    )
    _wsel1 = (
        (pdf.min_age < 1000) &
        sel_gyro_weakquality &
        (pdf.flag_planet_quality=='0') &
        ((pdf.koi_disposition == 'CONFIRMED') |
         (pdf.koi_disposition == 'CANDIDATE')
        ) &
        (pdf.li_eagles_limlo > -1.)
    )
    for k in pdf[_sel1].kepoi_name:
        if k not in COMMENTDICT:
            COMMENTDICT[k] = '\checkmark'

    pdf['comment'] = np.repeat('', len(pdf))
    for k,v in COMMENTDICT.items():
        pdf.loc[pdf.kepoi_name == k, 'comment'] = v

    # These columns will be written.
    mapdict = {
        'kepid': 'kepid',
        'kepoi_name': "KOI",
        "kepler_name": "Kepler",
        "adopted_Teff": r"$T_{\rm eff}$",
        "Prot": "Prot",
        "Li_EW": "Li_EW",
        "t_gyro": r"$t_{\rm gyro}$",
        "t_li": r"$t_{\rm Li}$",
        'are_gyro_and_li_consistent': 'are_gyro_and_li_consistent',
        "adopted_rp": r"$R_{\rm p}$",
        "adopted_period": r"$P$",
        #"flag_dr3_ruwe_outlier": r"$f_{\rm RUWE}$",
        "flag_planet_quality": r"$Q_{\rm planet}$",
        "flag_gyro_quality": r"$Q_{\rm gyro}$",
        "has_hires": r"Spec?",
        "comment": "Comment"
    }
    rounddict = {
        'adopted_period': 2,
        'adopted_rp': 2,
        'Prot': 2,
    }
    formatters = {}
    for k,v in rounddict.items():
        formatters[mapdict[k]] = lambda x: np.round(x, v)

    invdict = {v:k for k,v in mapdict.items()}

    pdf = pdf.rename(mapdict, axis='columns')

    pdf[['_t_gyro', '_t_gyro_err_pos', '_t_gyro_err_neg']] = (
        pdf[r'$t_{\rm gyro}$'].apply(lambda x: pd.Series(extract_value_and_error(x)))
    )
    pdf[['_t_li', '_t_li_err_pos', '_t_li_err_neg']] = (
        pdf[r'$t_{\rm Li}$'].apply(lambda x: pd.Series(extract_value_and_error(x)))
    )
    pdf['are_gyro_and_li_consistent'] = pdf.apply(lambda row: are_values_consistent(
        row['_t_gyro'], row['_t_gyro_err_pos'], row['_t_gyro_err_neg'],
        row['_t_li'], row['_t_li_err_pos'], row['_t_li_err_neg'],
        row['Spec?'], row[r"$t_{\rm Li}$"], row[r"$T_{\rm eff}$"]), axis=1
    )
    _pdf['are_gyro_and_li_consistent'] = pdf['are_gyro_and_li_consistent']

    pdf = pdf.drop(columns=['_t_gyro', '_t_gyro_err_pos', '_t_gyro_err_neg'])
    pdf = pdf.drop(columns=['_t_li', '_t_li_err_pos', '_t_li_err_neg'])

    print("Are gyro and lithium consistent?")

    sample = ['allages',
              'minageltonegyr',
              'allagesqflag',
              'minageltonegyrqflag']
    dfs = [pdf,
           pdf[pdf.min_age < 1e3],
           pdf[(pdf[r"$Q_{\rm gyro}$"]==0) &
               (pdf.koi_disposition=='CONFIRMED')],
           pdf[(pdf.min_age < 1e3) &
               (pdf[r"$Q_{\rm gyro}$"]==0) &
               (pdf.koi_disposition=='CONFIRMED')]
          ]

    for s, _df in zip(sample, dfs):

        nyes = len(np.unique(_df[_df.are_gyro_and_li_consistent == 'Yes'].kepid))
        nmaybe = len(np.unique(_df[_df.are_gyro_and_li_consistent == 'Maybe'].kepid))
        nno = len(np.unique(_df[_df.are_gyro_and_li_consistent == 'No'].kepid))

        if s == 'minageltonegyrqflag':
            assert nno == 2

        frac_consistent = int(np.round(100*(
            (nyes) / (nyes + nmaybe + nno)
        ),0))
        frac_potentiallyconsistent = int(np.round(100*(
            (nyes+nmaybe) / (nyes + nmaybe + nno)
        ),0))

        if not SELECT_YOUNG:
            ulkvp(f'{s}yesconsistent', nyes)
            ulkvp(f'{s}maybeconsistent', nmaybe)
            ulkvp(f'{s}noconsistent', nno)
            ulkvp(f'fracconsistent{s}', frac_consistent)
            ulkvp(f'fracpotentiallyconsistent{s}', frac_potentiallyconsistent)

            consistent_df = (
                pd.DataFrame(Counter(_df.are_gyro_and_li_consistent),
                             index=["Consistent?"])
            )
            print(consistent_df.T)
            print(f"{s}: {frac_consistent}% are consistent")
            print(10*'-')

    kepids = nparr(pdf['kepid'])
    pdf = pdf.drop(columns=['kepid'])
    mapdict.pop('kepid')

    if SELECT_YOUNG:
        sel = (pdf.koi_disposition == 'CONFIRMED')
        latexpath1 = join(PAPERDIR, 'table_subgyr_confirmed.tex')
        pdf[sel][mapdict.values()].head(n=95).to_latex(
            latexpath1, index=False, na_rep='--', escape=False, formatters=formatters
        )

        sel = (pdf.koi_disposition == 'CANDIDATE')
        latexpath2 = join(PAPERDIR, 'table_subgyr_candidate.tex')
        t2 = pdf[sel][mapdict.values()].head(n=20).to_latex(
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

        outtxt = join(TABLEDIR, 'inconsistent_tli_tgyro.txt')
        with open(outtxt, 'w') as f:
            f.writelines(
                pdf[(pdf.are_gyro_and_li_consistent == 'No')].to_string(max_colwidth=None)
            )
        print(f'Made {outtxt}')

        _s = (pdf.are_gyro_and_li_consistent != 'No')

        ulkvp(f'ltonegyrhighqconfirmedtwosided', len(pdf[_sel0 & _s]))
        ulkvp(f'ltonegyrhighqconfirmedonesided', len(pdf[_sel1 & _s]))
        ulkvp(f'ltonegyrmediumqconfirmedtwosided', len(pdf[_wsel0 & _s]))
        ulkvp(f'ltonegyrmediumqconfirmedonesided', len(pdf[_wsel1 & _s]))


    if not SELECT_YOUNG:

        ulkvp('nnonfopkoissomeageinfo', len(pdf))
        assert len(pdf) == len(_pdf)

        sel_cols = ['kepid', 'kepoi_name', 'kepler_name', 'koi_disposition',
                    'adopted_Teff', 'adopted_Teff_err', 'adopted_Teff_provenance',
                    'Prot', 'Prot_err', 'Prot_provenance',
                    'li_eagles_LiEW', 'li_eagles_eLiEW',
                    'gyro_median', 'gyro_+1sigma', 'gyro_-1sigma',
                    'gyro_+2sigma', 'gyro_-2sigma',
                    'li_median', 'li_+1sigma', 'li_-1sigma',
                    'adopted_rp', 'adopted_rp_provenance', 'adopted_period',
                    'flag_gyro_quality', 'flag_planet_quality', 'has_hires',
                    't_gyro', 't_li', 'are_gyro_and_li_consistent'
                   ]

        # format prot
        _pdf['Prot_err'] = _pdf.apply(format_prot_err, axis=1)
        _pdf['Prot'] = _pdf['Prot'].apply(
            lambda x: f"{x:.3f}" if x <= 5 else f"{x:.2f}"
        )
        outdf = _pdf[sel_cols]

        for column in outdf.columns:
            if outdf[column].dtype == 'object':  # Check if the column is of string type
                for key, value in BIBCODEDICT.items():
                    outdf[column] = outdf[column].str.replace(key, value)

        csvpath = join(TABLEDIR, 'table_allageinfo.csv')
        outdf.to_csv(csvpath, index=False)
        print(f'Wrote {csvpath}')

        # format ages
        sel = outdf['t_li'].str.contains(">")
        outdf.loc[sel, 'li_median'] = np.nan
        outdf.loc[sel, 'li_+1sigma'] = np.nan
        outdf.loc[sel, 'li_-1sigma'] = np.nan

        sel = outdf.gyro_median.astype(float) > 4000
        outdf.loc[sel, 't_gyro'] = '$> 4000$'
        outdf.loc[sel, 'gyro_median'] = np.nan
        outdf.loc[sel, 'gyro_+1sigma'] = np.nan
        outdf.loc[sel, 'gyro_-1sigma'] = np.nan

        csvpath = join(PAPERDIR, 'table_allageinfo_agesformatted.csv')
        outdf.to_csv(csvpath, index=False)
        print(f'Wrote {csvpath}')

        csvpath = join(TABLEDIR, 'table_allageinfo_allcols.csv')
        longdf.to_csv(csvpath, index=False)
        print(f'Wrote {csvpath}')



if __name__ == "__main__":
    make_table(SELECT_YOUNG = 0)
    make_table(SELECT_YOUNG = 1)
