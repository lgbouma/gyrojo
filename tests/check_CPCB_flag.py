"""
Check whether the S19/S21 "CP/CB" flag biases us against rapid rotators.

The answer is that it does!

Indeed, many of the fastest Prot stars are both CP/CB, and known KOI FPs.

However, there are 19 KOIs that are not known false positives, with rapidly
rotating target stars.  And they are ALL labelled CP/CB.

Therefore if we use the Santos CP/CB flag, we would be throwing out the 20
youngest planets based on what is basically an erroneous flag.  Nope.

(NB there is a separate missing selection function in the Santos sample,
described by Angela in correspondence over summer of 2022.  For any KOIs for
which Porb~=Prot, to within 20% tolerance, they OMITTED the KOIs from the
detection rotation period sample.  For instance Kepler-1643 meets this bill.)
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from gyrojo.getters import (
    get_gyro_data, get_li_data, get_age_results,
    get_kicstar_data, get_koi_data
)
from aesthetic.plot import set_style, savefig

def plot_CPCB_flag(includeFPs=1):
    # all S19+S21 prots, ignoring gyro applicability
    kic_df = get_kicstar_data('Santos19_Santos21_dquality')
    kic_df['KIC'] = kic_df.KIC.astype(str)

    # cumulative KOI table, ignoring whether planet is FP
    koi_df = get_koi_data('cumulative-KOI')
    koi_df['kepid'] = koi_df.kepid.astype(str)
    if not includeFPs:
        # no FPs, no grazing, MES>10
        koi_df = koi_df[koi_df.flag_is_ok_planetcand]

    # inner xmatch
    df = koi_df.merge(kic_df, how='inner', left_on='kepid', right_on='KIC')

    fig, ax = plt.subplots()

    s = '(incl FP KOIs)' if includeFPs else '(PC KOIs only)'
    ax.scatter(
        df.adopted_Teff,
        df.Prot,
        zorder=1,
        color='lightgray',
        s=1,
        label=f'S19+S21 x cumulative-KOI {s}'
    )

    sdf = df[~df.flag_not_CP_CB]
    ax.scatter(
        sdf.adopted_Teff,
        sdf.Prot,
        zorder=2,
        color='k',
        s=1,
        label=f'S19/S21 CPCB label {s}'
    )

    ax.set_xlim(ax.get_xlim()[::-1])
    ax.legend(fontsize='x-small')

    s = '_inclFPs' if includeFPs else ''
    savefig(fig, f'../results/test_results/Prot_vs_Teff_CPCB_label{s}.png')

if __name__ == '__main__':
    plot_CPCB_flag(includeFPs=1)
    plot_CPCB_flag(includeFPs=0)
