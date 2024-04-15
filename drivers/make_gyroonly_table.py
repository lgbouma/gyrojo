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
    dropgrazing = 1,
    drophighruwe = 1,
):

    # planets
    koidf, _, _ = get_age_results(
        whichtype='gyro_li', COMPARE_AGE_UNCS=0,
        drop_grazing=dropgrazing, drop_highruwe=drophighruwe
    )

    dropcols = [c for c in koidf.columns if 'adopted_age' in c]
    koidf = koidf.drop(labels=dropcols, axis='columns')

    #    kic_df['flag_is_gyro_applicable'] = (
    #        (~kic_df['flag_logg'])
    #        #&
    #        #(~df['flag_ruwe_outlier'])
    #        &
    #        (~kic_df['flag_dr3_non_single_star'])
    #        &
    #        (~kic_df['flag_camd_outlier'])
    #        #&
    #        #(df['flag_not_CP_CB'])
    #        &
    #        (~kic_df['flag_in_KEBC'])
    #        &
    #        (kic_df['adopted_Teff'] > 3800)
    #        &
    #        (kic_df['adopted_Teff'] < 6200)
    #    )

    #
    #    flag_is_ok_planetcand = (
    #       (~koi_df['flag_koi_is_fp'])
    #       &
    #       (~koi_df['flag_koi_is_low_snr'])
    #   )
    #   if drop_grazing:
    #       flag_is_ok_planetcand &= (~koi_df['flag_koi_is_grazing'])


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
        f'koi_gyro_ages_20240405_dropgrazing{dropgrazing}_drophighruwe{drophighruwe}.csv'
    )
    kdf = kdf.sort_values(by=['koi_disposition', 'gyro_median'],
                          ascending=[False,True])
    kdf.to_csv(outcsv, index=False)

    assert kdf.flag_is_gyro_applicable.sum() == len(kdf)
    assert kdf.flag_is_ok_planetcand.sum() == len(kdf)

if __name__ == "__main__":
    make_table(dropgrazing=1, drophighruwe=1)
    make_table(dropgrazing=0, drophighruwe=0)
