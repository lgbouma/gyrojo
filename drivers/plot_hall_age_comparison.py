import numpy as np, matplotlib.pyplot as plt, pandas as pd
import os
from os.path import join
from collections import Counter
from aesthetic.plot import savefig, set_style
from gyrojo.paths import DATADIR
from astropy.io import fits
from astropy.table import Table
from gyrojo.getters import select_by_quality_bits

def get_merge():

    litdir = join(DATADIR, "literature")
    fitspath = join(litdir, "Hall_2021_asteroseismic.fits")
    hl = fits.open(fitspath)
    hdf = Table(hl[1].data).to_pandas()
    # ok.
    sel = (hdf.Teff < 6200)
    hdf = hdf[sel]

    # NOTE: the hall sample seems to be biased to planet-hosts.  although
    # frankly the reason why is... not at all apparent to me!  shouldn't it be
    # agnostic to that?  regardless, for gyro ages, we hsould compare to the
    # full stellar sample...
    tabdir = '/Users/luke/Dropbox/proj/gyrojo/papers/paper/'
    df = pd.read_csv(join(tabdir, "table_star_gyro_agesformatted.csv"))
    df = df[df.adopted_Teff < 6200]

    mdf = hdf.merge(df, left_on='KIC', right_on='KIC', how='inner')
    sel = ~pd.isnull(mdf.gyro_median)
    mdf = mdf[sel]

    return mdf


def plot_hall_age_comparison(smalllim=0):

    mdf = get_merge()

    set_style('science')
    plt.close("all")

    fig, ax = plt.subplots(figsize=(4,4))

    sdf = mdf

    mask = select_by_quality_bits(
        sdf, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    sdf['flag_is_gyro_applicable'] = mask

    cols = 'KIC,adopted_Teff,gyro_median,gyro_-1sigma,gyro_+1sigma,Prot,Age,e_age_lc,e_Age,P,__Fe_H_'.split(",")
    print(sdf[sdf['flag_is_gyro_applicable']][cols].sort_values(by='gyro_median'))

    yerr = 1e3*np.array(
        [sdf['e_age_lc'], sdf['e_Age']]
    ).reshape((2, len(sdf)))
    xerr = np.array(
        [sdf['gyro_-1sigma'], sdf['gyro_+1sigma']]
    ).reshape((2, len(sdf)))

    ax.errorbar(
        sdf.gyro_median, 1e3*sdf.Age, xerr=xerr, yerr=yerr,
        marker='o', elinewidth=0.25, capsize=0, lw=0, mew=0.5, color='darkgray',
        markersize=1, zorder=5, label='Q$_\mathrm{star}$ > 0'
    )

    ssdf = sdf[sdf.flag_is_gyro_applicable]
    yerr = 1e3*np.array(
        [ssdf['e_age_lc'], ssdf['e_Age']]
    ).reshape((2, len(ssdf)))
    xerr = np.array(
        [ssdf['gyro_-1sigma'], ssdf['gyro_+1sigma']]
    ).reshape((2, len(ssdf)))

    ax.errorbar(
        ssdf.gyro_median, 1e3*ssdf.Age, xerr=xerr, yerr=yerr,
        marker='o', elinewidth=0.25, capsize=0, lw=0, mew=0.5, color='k',
        markersize=1, zorder=10, label='Q$_\mathrm{star}$ = 0'
    )

    ax.legend(fontsize='x-small', loc='upper right')

    ax.plot([1,1e4], [1,1e4], c='k', alpha=0.1, ls='--', zorder=-5)

    ax.update({
        'xlabel': 'gyro age [myr]',
        'ylabel': 'H21 astseis age [myr]',
    })

    s = ''
    if smalllim:
        ax.set_xlim([0,1000])
        ax.set_ylim([0,1000])
        s += "_smalllim"
    else:
        ax.set_xlim([0,10000])
        ax.set_ylim([0,10000])

    outdir = '/Users/luke/Dropbox/proj/gyrojo/results/hall_age_comparison'
    if not os.path.exists(outdir): os.mkdir(outdir)
    outpath = join(outdir, f'hall_age_comparison{s}.png')

    savefig(fig, outpath)

if __name__ == "__main__":
    plot_hall_age_comparison(smalllim=0)
    #plot_hall_age_comparison(smalllim=1)
