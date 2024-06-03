import os
import pandas as pd, matplotlib.pyplot as plt, numpy as np
from os.path import join
from gyrojo.paths import PAPERDIR, RESULTSDIR, TABLEDIR
from scipy.stats import truncnorm
from aesthetic.plot import savefig, set_style

def truncated_normal(mean, sd, lower, upper, N):
    return truncnorm((lower - mean) / sd, (upper - mean) / sd, loc=mean, scale=sd).rvs(N)



def plot_prot_vs_porb(df, yscale='log', fakedata=0):

    plt.close("all")
    set_style("clean")
    fig = plt.figure(figsize=(4,3))
    axd = fig.subplot_mosaic(
        """
        AAAA.
        BBBBC
        BBBBC
        BBBBC
        BBBBC
        """#,
        #gridspec_kw={
        #    "width_ratios": [1, 1, 1, 1]
        #}
    )

    ##### main scatter
    ax = axd['B']
    disps = ['CONFIRMED', 'CANDIDATE']
    labels = ['Confirmed', 'Candidate']
    sizes = [3, 1.5]
    alphas = [1, 0.3]

    for disp, s, a, l in zip(disps, sizes, alphas, labels):
        sel = df.koi_disposition == disp
        sdf = df[sel]
        ax.scatter(sdf.adopted_period, sdf.Prot, c='k', s=s, lw=0, alpha=a, label=l)
    ax.update({'xscale':'log', 'yscale': yscale,
               'xlabel':'$P_\mathrm{orb}$ [days]',
               'ylabel': '$P_\mathrm{rot}$ [days]'})
    if not fakedata:
        ax.legend(
            loc='best', fontsize='small',
            markerscale=2,
            framealpha=0,
            handletextpad=-0.2, borderaxespad=0.5, borderpad=0.5
        )

    ##### top hist
    axA = axd['A']
    bins_porb = 10 ** np.linspace(np.log10(0.1), np.log10(1000), int(10 * np.log10(1000/0.1) + 1))
    axA.hist(df.adopted_period, bins=bins_porb, color='k', alpha=0.3)
    sel = df.koi_disposition == 'CONFIRMED'
    sdf = df[sel]
    axA.hist(sdf.adopted_period, bins=bins_porb, color='k', alpha=1)
    axA.set_xscale('log')
    axA.set_xlim(ax.get_xlim())
    axA.set_ylabel('Count')
    if not fakedata:
        txt = 'Kepler Data'
    elif fakedata:
        txt = (
            'Random draws'
            '\n'
            '$P_{\mathrm{orb}}$$\sim$$\log\mathcal{N}$'
            '\n'
            '$P_{\mathrm{rot}}$$\sim$$\mathcal{T}_\mathcal{N}$'
        )
    axA.text(0.97, 0.97, txt, transform=axA.transAxes,
            ha='right',va='top', color='k', fontsize='small')
    axA.set_xticklabels([])

    ##### side hist
    axC = axd['C']
    if yscale == 'log':
        bins_prot = 10 ** np.linspace(np.log10(0.1), np.log10(100), int(10 * np.log10(100/0.1) + 1))
    elif yscale == 'linear':
        bins_prot = np.arange(0, 100, 2)
    axC.hist(df.Prot, bins=bins_prot, color='k', orientation='horizontal', alpha=0.3)
    sel = df.koi_disposition == 'CONFIRMED'
    sdf = df[sel]
    axC.hist(sdf.Prot, bins=bins_prot, color='k', alpha=1, orientation='horizontal')
    axC.set_yscale(yscale)
    axC.set_ylim(ax.get_ylim())
    axC.set_xlabel('Count')
    axC.set_yticklabels([])

    outdir = join(RESULTSDIR, "prot_vs_porb")
    if not os.path.exists(outdir): os.mkdir(outdir)
    s = ''
    if fakedata:
        s += '_fakedata'
    outpath = join(outdir, f"prob_vs_porb_yscale{yscale}{s}.png")
    savefig(fig, outpath)

if __name__ == "__main__":

    csvpath = join(TABLEDIR, "table_allageinfo.csv")
    df = pd.read_csv(csvpath)
    N = len(df)

    plot_prot_vs_porb(df, yscale='log')
    plot_prot_vs_porb(df, yscale='linear')

    np.random.seed(123)
    p_orbit = np.random.lognormal(mean=np.log(20), sigma=1, size=N)
    mean, sd, lower, upper = 18, 8, 1, np.inf
    p_rot = truncated_normal(mean, sd, lower, upper, N)

    df = pd.DataFrame({
        'adopted_period': p_orbit,
        'Prot': p_rot,
        'koi_disposition': np.repeat("CONFIRMED", len(p_rot))
    })

    plot_prot_vs_porb(df, yscale='log', fakedata=1)
    plot_prot_vs_porb(df, yscale='linear', fakedata=1)
