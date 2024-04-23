"""
e.g. sent to Fei 2024/04/09
"""

import os
from os.path import join
import pandas as pd, numpy as np
from gyrojo.getters import (
    get_age_results, get_kicstar_data
)
from gyrojo.paths import TABLEDIR

def make_table(
    ABBREVCOLS = 0, # abbreviate columns?
    grazing_is_ok = 0,
    drophighruwe = 1,
):

    # planets
    koidf, _, _ = get_age_results(
        whichtype='gyro_li', COMPARE_AGE_UNCS=0,
        grazing_is_ok=grazing_is_ok, drop_highruwe=drophighruwe
    )

    dropcols = [c for c in koidf.columns if 'adopted_age' in c]
    koidf = koidf.drop(labels=dropcols, axis='columns')

    koidf = koidf.rename({'Provenance':'Prot_provenance'}, axis='columns')
    selcols = ['kepid', 'kepoi_name', 'kepler_name', 'koi_disposition',
               'adopted_Teff', 'adopted_Teff_provenance', 'adopted_Teff_err',
               'Prot', 'Prot_provenance']
    cols = [c for c in koidf.columns if 'gyro_' in c or
            ('flag_' in c and 'koi_' not in c)]
    for c in cols:
        selcols.append(c)

    if ABBREVCOLS:
        kdf = koidf[selcols]
    else:
        kdf = koidf

    outcsv = join(
        TABLEDIR,
        f'koi_gyro_ages_20240405_grazingisok{grazing_is_ok}_drophighruwe{drophighruwe}.csv'
    )
    kdf = kdf.sort_values(by=['koi_disposition', 'gyro_median'],
                          ascending=[False,True])
    kdf.to_csv(outcsv, index=False)

    assert kdf.flag_is_gyro_applicable.sum() == len(kdf)
    assert kdf.flag_is_ok_planetcand.sum() == len(kdf)

if __name__ == "__main__":
    for g in [0,1]:
        for r in [0,1]:
            make_table(grazing_is_ok=g, drophighruwe=r)
