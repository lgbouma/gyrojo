import numpy as np, matplotlib.pyplot as plt, pandas as pd
from os.path import join
from collections import Counter
from aesthetic.plot import savefig, set_style

def plot_gyro_v_li_age_comparison(smalllim=0):

    tabdir = '/Users/luke/Dropbox/proj/gyrojo/results/tables'
    df = pd.read_csv(join(tabdir, "table_allageinfo_allcols.csv"))

    set_style('science')
    plt.close("all")

    lilimit = (df.li_eagles_limlo > 0)

    fig, ax = plt.subplots(figsize=(4,4))

    sdf = df[~lilimit]

    yerr = np.array(
        [sdf['li_-1sigma'], sdf['li_+1sigma']]
    ).reshape((2, len(sdf)))
    xerr = np.array(
        [sdf['gyro_-1sigma'], sdf['gyro_+1sigma']]
    ).reshape((2, len(sdf)))

    ax.errorbar(
        sdf.gyro_median, sdf.li_median, xerr=xerr, yerr=yerr,
        marker='o', elinewidth=0.25, capsize=0, lw=0, mew=0.5, color='darkgray',
        markersize=1, zorder=5, label='Q$_\mathrm{star}$ > 0'
    )

    ssdf = sdf[sdf.flag_is_gyro_applicable]
    yerr = np.array(
        [ssdf['li_-1sigma'], ssdf['li_+1sigma']]
    ).reshape((2, len(ssdf)))
    xerr = np.array(
        [ssdf['gyro_-1sigma'], ssdf['gyro_+1sigma']]
    ).reshape((2, len(ssdf)))

    ax.errorbar(
        ssdf.gyro_median, ssdf.li_median, xerr=xerr, yerr=yerr,
        marker='o', elinewidth=0.25, capsize=0, lw=0, mew=0.5, color='k',
        markersize=1, zorder=10, label='Q$_\mathrm{star}$ = 0'
    )

    ax.legend(fontsize='x-small', loc='upper right')

    ax.plot([1,6e3], [1,6e3], c='k', alpha=0.1, ls='--', zorder=-5)

    ax.update({
        'xlabel': 'gyro age [myr]',
        'ylabel': 'li age [myr]',
    })

    s = ''
    if smalllim:
        ax.set_xlim([0,1000])
        ax.set_ylim([0,1000])
        s += "_smalllim"
    else:
        ax.set_xlim([0,6000])
        ax.set_ylim([0,6000])

    outdir = '/Users/luke/Dropbox/proj/gyrojo/results/gyro_v_li_age_comparison'
    outpath = join(outdir, f'gyro_v_li_age_comparison{s}.png')

    savefig(fig, outpath)

if __name__ == "__main__":
    plot_gyro_v_li_age_comparison(smalllim=0)
    plot_gyro_v_li_age_comparison(smalllim=1)
