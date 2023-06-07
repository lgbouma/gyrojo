"""
Check whether particular stars are on Chris Burke's "Noisy Target List"
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from gyrojo.getters import (
    get_gyro_data, get_li_data, get_age_results,
    get_kicstar_data, get_koi_data
)
from aesthetic.plot import set_style, savefig

from gyrojo.paths import DATADIR
from os.path import join

def plot_BurkeNoisy_flag(KOIs=1, field=0):

    assert KOIs + field == 1

    # all S19+S21 prots, include gyro applicability
    kic_df = get_kicstar_data('Santos19_Santos21_dquality')
    kic_df['KIC'] = kic_df.KIC.astype(str)
    kic_df = kic_df[kic_df.flag_is_gyro_applicable]

    # cumulative KOI table, no FPs, no grazing, MES>10
    koi_df = get_koi_data('cumulative-KOI')
    koi_df['kepid'] = koi_df.kepid.astype(str)
    koi_df = koi_df[koi_df.flag_is_ok_planetcand]

    csvpath = join(DATADIR, 'literature',
                   'Burke_DR25_DEModel_NoisyTargetList.csv')
    b_df = pd.read_csv(csvpath, comment='#')
    b_df['KIC'] = b_df['kic_id'].astype(str)

    if KOIs:
        # inner xmatch btwn KIC and KOI information.
        df = koi_df.merge(kic_df, how='inner', left_on='kepid', right_on='KIC')
    elif field:
        from copy import deepcopy
        df = deepcopy(kic_df)

    # build the flag column
    df['flag_in_BurkeNoisy'] = df.KIC.isin(b_df.KIC)

    fig, ax = plt.subplots()

    s = ''
    if KOIs:
        s = '(PC KOIs only)'
    x = ''
    if KOIs:
        x = ' x cumulative-KOI '
    ax.scatter(
        df.adopted_Teff,
        df.Prot,
        zorder=1,
        color='lightgray',
        s=1,
        label=f'S19+S21 Prot, gyroOK;{x}{s}'
    )

    sdf = df[df.flag_in_BurkeNoisy]
    ax.scatter(
        sdf.adopted_Teff,
        sdf.Prot,
        zorder=2,
        color='k',
        s=1,
        label=f'S19/S21 BurkeNoisy label {s}'
    )

    ax.set_xlim(ax.get_xlim()[::-1])
    ax.legend(fontsize='x-small')

    if KOIs:
        s = '_KOIs'
    if field:
        s = '_field'
    savefig(fig, f'../results/test_results/Prot_vs_Teff_BurkeNoisy_label{s}.png')

if __name__ == '__main__':
    plot_BurkeNoisy_flag(KOIs=0, field=1)
    plot_BurkeNoisy_flag(KOIs=1, field=0)
