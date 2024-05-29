"""
There are a few interesting clusters that overlapped with the Kepler field...

CepHer... Melange3... Theia520... NGC6811...

They provide probably the best ground-truth ages, at least if spatio/kinematic
stellar ages can be acquired.

Let's start with NGC 6811 and see where it goes.
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt
from astropy.io import fits, ascii
from astropy.table import Table
import os
from os.path import join
from gyrojo.paths import LOCALDIR, PAPERDIR, DATADIR
from gyrojo.getters import select_by_quality_bits
from gyrointerp.helpers import prepend_colstr, left_merge
from astroquery.vizier import Vizier
from aesthetic.plot import set_style, savefig
from collections import Counter
from operator import itemgetter
from numpy import array as nparr

Vizier.ROW_LIMIT = -1

def get_cluster_dfs():

    cdipscsvpath = join(
        LOCALDIR, "cdips", "catalogs",
        "cdips_targets_v0.6_nomagcut_gaiasources.csv"
    )
    df = pd.read_csv(cdipscsvpath)
    sel = ~pd.isnull(df.cluster)
    sdf = df[sel]

    kepler_dr2_path = join(LOCALDIR, "kepler_dr2_1arcsec.fits")
    hdul = fits.open(kepler_dr2_path)
    gk_df = Table(hdul[1].data).to_pandas()
    hdul.close()

    kepler_dr3_path = join(LOCALDIR, "kepler_dr3_1arcsec.fits")
    hdul = fits.open(kepler_dr3_path)
    gdr3k_df = Table(hdul[1].data).to_pandas()
    hdul.close()


    #
    # Get NGC6811
    #
    # 355 from Cantat-Gaudin 2018, 2020ab.
    # Plus another 200 from Kounkel+2020.
    nsdf = sdf[sdf.cluster.str.contains("6811") &
               ~sdf.cluster.str.contains("kcs20group_6811")]
    print(Counter(nsdf.reference_bibcode).most_common(n=10))

    # Now xmatch against the kic <-> dr2 xmatch...
    ngc6811_df = nsdf.merge(gk_df, how='inner', on='source_id')

    #
    # Get NGC6819 (2.5 Gyr!)
    #
    nsdf = sdf[sdf.cluster.str.contains("6819") &
               ~sdf.cluster.str.contains("kcs20group_6819")]
    print(Counter(nsdf.reference_bibcode).most_common(n=10))
    ngc6819_df = nsdf.merge(gk_df, how='inner', on='source_id')

    # 
    # Get Theia520
    # The regex is a negative lookahead assertion that ensures the character
    # immediately following the end digit (e.g. UBC_"1") is not a digit.
    #
    t520_sdf = sdf[sdf.cluster.str.contains("kcs20group_520(?!\d)") |
                   sdf.cluster.str.contains(r"UBC_1(?!\d)")
                   ]
    print(Counter(t520_sdf.reference_bibcode).most_common(n=10))
    theia520_df = t520_sdf.merge(gk_df, how='inner', on='source_id')

    #
    # Get Melange-3.  Barber=22 used EDR3.
    #
    tabpath = join(DATADIR, "literature", "Barber_2022_ajac7b28t5_mrt.txt")
    _m3_df = ascii.read(tabpath, format='mrt').to_pandas()
    sel = _m3_df['Voff'] < 2 # km/s

    melange3_df = _m3_df[sel].merge(gdr3k_df, how='inner', left_on='Gaia',
                                    right_on='source_id')

    #
    # Get Cep-Her.
    #
    CH_BOUMA2022 = 0
    CH_KERR2024 = 1
    if CH_BOUMA2022:
        tabpath = join(DATADIR, "literature", "Bouma_2022_ajac93fft2_mrt.txt")
        _ch_df = ascii.read(tabpath, format='mrt').to_pandas()
        sel = _ch_df['weight'] > 0.1
        cepher_df = _ch_df[sel].merge(gdr3k_df, how='inner', left_on='DR3',
                                      right_on='source_id')
    if CH_KERR2024:
        # secret for github until this paper is published
        csvpath = join(DATADIR, "literature",
                       "secret_CepHer_FinalMembersCatalog.csv")
        ch_df = pd.read_csv(csvpath) # Kerr+2024 early release, dr3 source ids
        sel = ch_df.Pfin > 0.8
        cepher_df = ch_df[sel].merge(gdr3k_df, how='inner', left_on='ID',
                                     right_on='source_id')

    ##########################################

    cluster_dfs = [ngc6811_df, cepher_df, ngc6819_df, melange3_df, theia520_df]
    cluster_names = ['NGC6811 (1 Gyr)',
                     'Cep-Her (40 Myr)', #$w$>0.1 
                     'NGC6819 (2.5 Gyr)',
                     'Melange3 (150 Myr)', # Δv$_\mathrm{T}$<2km/s 
                     'Theia520 (300 Myr)'
                    ]
    true_ages = [1, 0.04, 2.5, 0.15, 0.3]

    return cluster_dfs, cluster_names, true_ages


cluster_dfs, cluster_names, true_ages = get_cluster_dfs()

# Bouma 2024 gyro ages...
csvpath = join(PAPERDIR, "table_star_gyro_allcols.csv")
bdf = pd.read_csv(csvpath)

# Berger+2020: Gaia-Kepler stellar properties catalog.
# Table 2: output parameters
# "E_" means upper err, "e_" means lower.  Note that "e_" is signed, so
# that all entries in these columns are negative.
# ['recno', 'KIC', 'Mass', 'E_Mass', 'e_Mass', 'Teff', 'E_Teff', 'e_Teff',
#  'logg', 'E_logg', 'e_logg', '__Fe_H_', 'E__Fe_H_', 'e__Fe_H_', 'Rad',
#  'E_Rad', 'e_Rad', 'rho', 'E_rho', 'e_rho', 'Lum', 'E_Lum', 'e_Lum',
#  'Age', 'f_Age', 'E_Age', 'e_Age', 'Dist', 'E_Dist', 'e_Dist', 'Avmag',
#  'GOF', 'TAMS']
_v = Vizier(columns=["**"])
_v.ROW_LIMIT = -1
catalogs = _v.get_catalogs("J/AJ/159/280")
b20t2_df = catalogs[1].to_pandas() # output parameters
b20t2_df = prepend_colstr('b20t2_', b20t2_df)

# Mathur+2023
tabpath = join(DATADIR, "literature", "Mathur2023_apjacd118t1_mrt.txt")
m23_df = ascii.read(tabpath, format='mrt').to_pandas()
tabpath = join(DATADIR, "literature", "Mathur2023_apjacd118t3_mrt.txt")
m23_df3 = ascii.read(tabpath, format='mrt').to_pandas()

# Lu+2021
tabpath = join(DATADIR, "literature", "Lu_2021_ajabe4d6t1_mrt.txt")
l21_df = ascii.read(tabpath, format='mrt').to_pandas()

# Lu+2024
tabpath = join(DATADIR, "literature", "Lu_2024_ajad28b9t2_mrt.txt")
l24_df = ascii.read(tabpath, format='mrt').to_pandas()

# Reinhold+2015
catalogs = _v.get_catalogs("J/A+A/583/A65")
r15_df = catalogs[1].to_pandas() # output parameters
r15_df = r15_df[~r15_df['tMH08'].isna()]

def _get_cluster_xm_lit_dfs(cluster_df):
    """
    Given "cluster dataframe" with some set of kepler id's, get the
    crossmatches between literature data and that cluster.
    """

    # Bouma24
    s0 = select_by_quality_bits(
        bdf,
        [0, 1, 2, 3, 4, 5, 6],  # leave high ruwe... and crowded... and far from ms
        [0, 0, 0, 0, 0, 0, 0]
    )
    loose_gyro_cut = True

    if not loose_gyro_cut:
        sel = (
            (bdf.flag_is_gyro_applicable)
            &
            (bdf.kepid.isin(cluster_df.kepid))
        )
    else:
        sel = (
            s0
            &
            (bdf.kepid.isin(cluster_df.kepid))
        )
    sb24df = bdf[sel]

    # Berger20
    # f_Age: Ages with uninformative posteriors (TAMS>20 Gyr) or
    # unreliable ages (GOF<0.99) are flagged with an asterisk (26581 occurrences).
    sel = (
        (b20t2_df.b20t2_KIC.isin(cluster_df.kepid))
        &
        (b20t2_df.b20t2_f_Age != '*')
    )
    sb20_df = b20t2_df[sel].sort_values(by='b20t2_Age')

    # Mathur23 table1
    sel = (
        (m23_df.KIC.isin(cluster_df.kepid))
        &
        (m23_df['flag-bin'] == 0)
        &
        (m23_df['flag'] == 0)
    )
    sm23_df = m23_df[sel].sort_values(by='Age')

    # Mathur23 table3
    sel = (
        (m23_df3.KIC.isin(sm23_df.KIC))
    )
    sm23_df3 = m23_df3[sel].sort_values(by='Age')

    #
    # Lu+2021 gyrokinematic
    # 
    sel = (
        (l21_df.kepid.isin(cluster_df.kepid))
    )
    sl21_df = l21_df[sel].sort_values(by='kinage')

    #
    # Lu+2024 gyro-only (...calibrated on kinematic)
    #
    sel = (
        (l24_df.KIC.isin(cluster_df.kepid))
    )
    sl24_df = l24_df[sel].sort_values(by='Age')

    #
    # Reinhold gyro-only
    #
    sel = (
        (r15_df.KIC.isin(cluster_df.kepid))
    )
    sr15_df = r15_df[sel].sort_values(by='tMH08')

    return sb24df, sm23_df, sm23_df3, sb20_df, sl21_df, sl24_df, sr15_df


for cluster_df, cluster_full_name, true_age in zip(
    cluster_dfs, cluster_names, true_ages
):

    cluster_short_name = cluster_full_name.split(" ")[0]

    sb24df, sm23_df, sm23_df3, sb20_df, sl21_df, sl24_df, sr15_df = (
        _get_cluster_xm_lit_dfs(cluster_df)
    )


    #
    # Make individual plots!
    #
    plt.close("all")
    set_style("clean")

    fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(0.8*7,0.8*1.8))
    axs = axs.flatten()

    # STAREVOL from SM23 qualtiatively similar, but mildly worse.
    dfs = [
        sb24df, sm23_df, #sm23_df3,
        sb20_df, sl21_df, sl24_df, sr15_df
    ]
    xvalnames = [
        'gyro_median', 'Age', #'Age',
        'b20t2_Age', 'kinage', "Age"
    ]

    # "E_" means upper err, "e_" means lower.  Note that "e_" is signed, so
    # that all entries in these columns are negative.
    xmerrnames = [
        'gyro_-1sigma', 'e_Age', #'e_Age',
        'b20t2_e_Age', 'e_kinage', "e_Age", 'tMH08'
    ]
    xperrnames = [
        'gyro_+1sigma', 'E_Age', #'E_Age',
        'b20t2_E_Age', 'e_kinage', "E_Age", 'e_tMH08'
    ]
    div1ks = [
        1, 0, #0,
        0, 0, 0, 1
    ]

    texts = [
        'gyro-interp (Bouma+24)', 'kiauhoku (Mathur+23)',
        #'STAREVOL (Mathur+23)',
        'Isochrones (Berger+20)',
        'gyro-kinematic (Lu+21)', 'gyro (Lu+24)', 'gyro (Reinhold+15)'
    ]

    for ix, (ax, df, xkey, xmerrkey, xperrkey, div1k, txt) in enumerate(zip(
        axs, dfs, xvalnames, xmerrnames, xperrnames, div1ks, texts
    )):

        if div1k:
            f = 1e3
        else:
            f = 1

        yval = np.linspace(0, 1, len(df))

        xerr = np.array(
            [np.abs(df[xmerrkey])/f, df[xperrkey]/f]
        ).reshape((2, len(df)))  / (true_age)
        xval = ( df[xkey]/f ) / true_age
        ax.errorbar(
            xval, yval, xerr=xerr,
            marker='o', elinewidth=0.3, capsize=0, lw=0, mew=0.5, color=f'C{ix}',
            markersize=0.5, zorder=5
        )

        for lval in [0.01, 0.1, 1, 10, 100]:
            lw = 0.5 if lval != 1 else 0.5
            ls = ':' if lval != 1 else '-'
            color = 'lightgray' if lval != 1 else 'k'
            ax.vlines(lval, 0, 1, zorder=-99, color=color, ls=ls, lw=lw)

        _txt = f'N={len(df)}\n{txt}'
        ax.text(
            0.5, 1, _txt, ha='center', va='bottom', fontsize='xx-small',
            zorder=10, color='k', transform=ax.transAxes
        )

        ax.set_xlim([0.003, 300])
        ax.set_xscale('log')

        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticks([0.01,1,100])
        ax.set_xticklabels(['0.01', '1', '100'])

        ax.spines['left'].set_visible(False)

    #fig.text(0.5,1.01, cluster_full_name, ha='center')
    fig.text(-0.01,0.5, f'{cluster_full_name}', va='center', rotation=90)
    fig.text(0.5,-0.01, 'Reported Stellar Age / Cluster Age', ha='center')

    outdir = '../results/lit_ages_vs_cluster_ages'
    if not os.path.exists(outdir): os.mkdir(outdir)
    outpath = join(outdir, f"{cluster_short_name}_lit_ages_vs_cluster_ages.png")
    fig.tight_layout()
    savefig(fig, outpath, dpi=400)


##########################################
##########################################
##########################################

## okay, now make the mega 5x5 plot
plt.close("all")
set_style("clean")

fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(0.8*7,4.5*0.8*1.8))
#axs = axs.flatten()

for ix, (cluster_df, cluster_full_name, true_age) in enumerate(zip(
    cluster_dfs, cluster_names, true_ages
)):

    cluster_short_name = cluster_full_name.split(" ")[0]

    sb24df, sm23_df, sm23_df3, sb20_df, sl21_df, sl24_df, sr15_df = (
        _get_cluster_xm_lit_dfs(cluster_df)
    )

    #
    # Make individual plots!
    #

    # STAREVOL from SM23 qualtiatively similar, but mildly worse.
    dfs = [
        sb24df, sm23_df, #sm23_df3,
        sb20_df, sl21_df, sl24_df
    ]
    xvalnames = [
        'gyro_median', 'Age', #'Age',
        'b20t2_Age', 'kinage', "Age"
    ]

    # "E_" means upper err, "e_" means lower.  Note that "e_" is signed, so
    # that all entries in these columns are negative.
    xmerrnames = [
        'gyro_-1sigma', 'e_Age', #'e_Age',
        'b20t2_e_Age', 'e_kinage', "e_Age"
    ]
    xperrnames = [
        'gyro_+1sigma', 'E_Age', #'E_Age',
        'b20t2_E_Age', 'e_kinage', "E_Age"
    ]
    div1ks = [
        1, 0, #0,
        0, 0, 0
    ]

    texts = [
        'gyro-interp\n(This work)', 'kiauhoku\n(Mathur+23)',
        #'STAREVOL (Mathur+23)',
        'Isochrones\n(Berger+20)',
        'gyro-kinematic\n(Lu+21)', 'gyro\n(Lu+24)'
    ]

    for iy, (df, xkey, xmerrkey, xperrkey, div1k, txt) in enumerate(zip(
        dfs, xvalnames, xmerrnames, xperrnames, div1ks, texts
    )):

        ax = axs[ix, iy]

        if div1k:
            f = 1e3
        else:
            f = 1

        yval = np.linspace(0, 1, len(df))

        xerr = np.array(
            [np.abs(df[xmerrkey])/f, df[xperrkey]/f]
        ).reshape((2, len(df)))  / (true_age)
        xval = ( df[xkey]/f ) / true_age
        ax.errorbar(
            xval, yval, xerr=xerr,
            marker='o', elinewidth=0.3, capsize=0, lw=0, mew=0.5, color=f'C{iy}',
            markersize=0.5, zorder=5
        )

        for lval in [0.01, 0.1, 1, 10, 100]:
            lw = 0.5 if lval != 1 else 0.5
            ls = ':' if lval != 1 else '-'
            color = 'lightgray' if lval != 1 else 'k'
            ax.vlines(lval, 0, 1, zorder=-99, color=color, ls=ls, lw=lw)

        _txt = f'N={len(df)}'
        ax.text(
            0.1, 0.9, _txt, ha='left', va='top', fontsize='xx-small',
            zorder=10, color='k', transform=ax.transAxes
        )

        ax.set_xlim([0.003, 300])
        ax.set_xscale('log')

        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticks([0.01,1,100])
        if ix == 4:
            ax.set_xticklabels(['0.01', '1', '100'])
        else:
            ax.set_xticklabels([])

        ax.spines['left'].set_visible(False)

        if iy == 0:
            ax.set_ylabel(f'{cluster_full_name}', fontsize='medium')
        if ix == 0:
            ax.set_title(txt, fontsize='medium')

        if ix == 4 and iy == 2:
            ax.set_xlabel('Reported Stellar Age / Cluster Age',
                          fontsize='medium')

#fig.text(0.5,-0.01, 'Reported Stellar Age / Cluster Age', ha='center', fontsize='medium')

outdir = '../results/lit_ages_vs_cluster_ages'
if not os.path.exists(outdir): os.mkdir(outdir)
outpath = join(outdir, f"merged_lit_ages_vs_cluster_ages.png")
fig.tight_layout()
savefig(fig, outpath, dpi=400)

##########################################
##########################################
##########################################

# todo: resort!

## okay, now make the 5x1 plot
plt.close("all")
set_style("clean")

fig = plt.figure(figsize=(0.8*7,0.8*5))
#axd = fig.subplot_mosaic(
#    """
#    AABBCC
#    .DDEE.
#    """)
axd = fig.subplot_mosaic(
    """
    AABBCC
    DDEEFF
    """
)


axs = [axd['A'], axd['B'], axd['C'], axd['D'], axd['E'], axd['F']]

inds = np.argsort(true_ages)

np.random.seed(42)

for ix, (cluster_df, cluster_full_name, true_age) in enumerate(zip(
    [cluster_dfs[i] for i in inds],
    [cluster_names[i] for i in inds],
    [true_ages[i] for i in inds]
)):

    cluster_short_name = cluster_full_name.split(" ")[0]

    sb24df, sm23_df, sm23_df3, sb20_df, sl21_df, sl24_df, sr15_df = (
        _get_cluster_xm_lit_dfs(cluster_df)
    )

    #
    # Make individual plots!
    #

    # STAREVOL from SM23 qualtiatively similar, but mildly worse.
    dfs = [
        sb24df, sm23_df, #sm23_df3,
        sb20_df, sl21_df, sl24_df, sr15_df
    ]
    xvalnames = [
        'gyro_median', 'Age', #'Age',
        'b20t2_Age', 'kinage', "Age", 'tMH08'
    ]

    # "E_" means upper err, "e_" means lower.  Note that "e_" is signed, so
    # that all entries in these columns are negative.
    xmerrnames = [
        'gyro_-1sigma', 'e_Age', #'e_Age',
        'b20t2_e_Age', 'e_kinage', "e_Age", 'e_tMH08'
    ]
    xperrnames = [
        'gyro_+1sigma', 'E_Age', #'E_Age',
        'b20t2_E_Age', 'e_kinage', "E_Age", 'e_tMH08'
    ]
    div1ks = [
        1, 0, #0,
        0, 0, 0, 1
    ]

    texts = [
        'gyro-interp\n(This work)', 'kiauhoku\n(Mathur+23)',
        #'STAREVOL (Mathur+23)',
        'Isochrones\n(Berger+20)',
        'gyro-kinematic\n(Lu+21)', 'gyro\n(Lu+24)', 'gyro\n(Reinhold+15)'
    ]

    for iy, (df, xkey, xmerrkey, xperrkey, div1k, txt) in enumerate(zip(
        dfs, xvalnames, xmerrnames, xperrnames, div1ks, texts
    )):

        ax = axs[iy]

        if div1k:
            f = 1e3
        else:
            f = 1

        yerr = nparr(
            [np.abs(df[xmerrkey])/f, df[xperrkey]/f]
        ).reshape((2, len(df)))  / (true_age)
        yval = ( nparr(df[xkey]/f) ) / true_age

        if 'gyro-kinematic' in txt:
            upperlim = yval - yerr[0,:] < 0
            if np.any(upperlim):
                eps = np.random.normal(loc=0, scale=0.12, size=len(yval[upperlim]))
                ax.errorbar(
                    eps+ix*np.ones_like(yval[upperlim]), yval[upperlim],
                    yerr=np.zeros_like(yval[upperlim]),
                    marker='v', elinewidth=0., capsize=0, lw=0, mew=0.5, color=f'C{iy}',
                    markersize=0.5, zorder=5, alpha=0.3
                )

                eps = np.random.normal(loc=0, scale=0.12, size=len(yval[~upperlim]))
                ax.errorbar(
                    eps+ix*np.ones_like(yval[~upperlim]), yval[~upperlim],
                    yerr=yerr[:, ~upperlim],
                    marker='o', elinewidth=0.2, capsize=0, lw=0, mew=0.5, color=f'C{iy}',
                    markersize=0.5, zorder=5
                )

        else:
            eps = np.random.normal(loc=0, scale=0.12, size=len(yval))
            ax.errorbar(
                eps+ix*np.ones_like(yval), yval, yerr=yerr,
                marker='o', elinewidth=0.2, capsize=0, lw=0, mew=0.5, color=f'C{iy}',
                markersize=0.5, zorder=5
            )

        for lval in [0.01, 0.1, 1, 10, 100]:
            lw = 0.5 if lval != 1 else 0.5
            ls = ':' if lval != 1 else '-'
            color = 'lightgray' if lval != 1 else 'k'
            ax.hlines(lval, -2, 6, zorder=-99, color=color, ls=ls, lw=lw)

        _txt = f'N={len(df)}'
        #ax.text(
        #    0.1, 0.9, _txt, ha='left', va='top', fontsize='xx-small',
        #    zorder=10, color='k', transform=ax.transAxes
        #)

        ax.set_xlim([-0.5, 4.5])
        ax.set_ylim([0.03, 300])
        ax.set_yscale('log')

        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xticklabels([
            'Cep-Her\n(40 Myr)',
            'Melange-3\n(150 Myr)',
            'Theia 520\n(300 Myr)',
            'NGC6811\n(1 Gyr)',
            'NGC6819\n(2.5 Gyr)'
        ], rotation=75, fontsize='x-small')
        ax.set_yticks([0.1,1,10,100])
        if iy in [0, 3]:
            ax.set_yticklabels(['0.1', '1', '10', '100'])
            ax.set_ylabel('Star Age / Cluster Age', fontsize='medium')
        else:
            ax.set_yticklabels(['0.1', '1', '10', '100'])

        #ax.spines['left'].set_visible(False)

        #if iy == 0:
        #    ax.set_ylabel(f'{cluster_full_name}', fontsize='medium')
        #if ix == 0:
        ax.set_title(txt, fontsize='medium')

#fig.text(-0.01,0.5, 'Reported Star Age / Cluster Age', va='center',
#         fontsize='medium', rotation=90)

outdir = '../results/lit_ages_vs_cluster_ages'
if not os.path.exists(outdir): os.mkdir(outdir)
outpath = join(outdir, f"olympic_merged_lit_ages_vs_cluster_ages.png")
fig.tight_layout(w_pad=0.1)
savefig(fig, outpath, dpi=400)


