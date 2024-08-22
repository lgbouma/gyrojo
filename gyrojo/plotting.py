"""
Catch-all file for plotting scripts.  Contents:

    plot_koi_mean_prot_teff
    plot_star_Prot_Teff
    plot_reinhold_2015

    plot_li_vs_teff
    plot_liagefloor_vs_teff

    plot_gyroage_vs_teff
    plot_st_params

    plot_koi_gyro_posteriors
    plot_process_koi_li_posteriors
    plot_field_gyro_posteriors

    plot_hist_field_gyro_ages

    plot_rp_vs_age
    plot_rp_vs_porb_binage
    plot_rp_ks_test

    plot_age_comparison

    plot_multis_vs_age

    plot_gyromodeldispersion

    plot_perioddiff_vs_period

Helpers:
    _given_ax_append_spectral_types

    Sub-plot makers, to prevent code duplication:
        _plot_slow_sequence_residual
        _plot_prot_vs_teff_residual
"""
import os, sys
from os.path import join
from glob import glob
from datetime import datetime

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.transforms as transforms
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from numpy import array as nparr

from astropy.table import Table
from astropy.io import fits

from gyrojo.paths import DATADIR, RESULTSDIR, LOCALDIR, CACHEDIR
from gyrojo.getters import (
    get_gyro_data, get_li_data, get_age_results,
    get_kicstar_data, get_koi_data, get_prot_metacatalog,
    select_by_quality_bits
)
from gyrojo.papertools import update_latex_key_value_pair as ulkvp

from gyrointerp.models import (
    reference_cluster_slow_sequence
)
from gyrointerp.models import slow_sequence, slow_sequence_residual

from aesthetic.plot import set_style, savefig

from cdips.utils.gaiaqueries import propermotion_to_kms

from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1


###########
# helpers #
###########
def _given_ax_append_spectral_types(ax):
    # Append SpTypes (ignoring reddening)
    from cdips.utils.mamajek import get_SpType_Teff_correspondence

    tax = ax.twiny()
    xlim = ax.get_xlim()
    getter = get_SpType_Teff_correspondence
    sptypes, xtickvals = getter(
        ['F2V','F5V','G2V','K0V','K5V','M0V','M3V']
    )
    print(sptypes)
    print(xtickvals)

    xvals = np.linspace(min(xlim), max(xlim), 100)
    tax.plot(xvals, np.ones_like(xvals), c='k', lw=0) # hidden, but fixes axis.
    tax.set_xlim(xlim)
    ax.set_xlim(xlim)

    tax.set_xticks(xtickvals)
    tax.set_xticklabels(sptypes, fontsize='medium')

    tax.xaxis.set_ticks_position('top')
    tax.tick_params(axis='x', which='minor', top=False)
    tax.get_yaxis().set_tick_params(which='both', direction='in')


def get_planet_class_labels(df, OFFSET=0, rpkey='rp', periodkey='period'):
    # given a dataframe with keys "rp" and "period", return a dataframe with a
    # "pl_class" key 

    df['pl_class'] = ''

    sel = df[rpkey] <= 1
    df.loc[sel, 'pl_class'] = 'Earths'

    sel = (df[rpkey] >= 4) & (df[rpkey] < 10)
    df.loc[sel, 'pl_class'] = 'Sub-Saturns'

    sel = df[rpkey] >= 10
    df.loc[sel, 'pl_class'] = 'Jupiters'

    # van eylen+2018
    m = -0.09
    a = 0.37
    fn_Rmod = lambda log10Pmod: 10**(m*log10Pmod + a)

    R_mod = fn_Rmod(np.log10(df[periodkey])) + OFFSET

    sel = (df[rpkey] < 4) & (df[rpkey] > R_mod)
    df.loc[sel, 'pl_class'] = 'Mini-Neptunes'

    sel = (df[rpkey] > 1) & (df[rpkey] <= R_mod)
    df.loc[sel, 'pl_class'] = 'Super-Earths'

    return df


############
# plotters #
############
def plot_koi_mean_prot_teff(outdir, sampleid='koi_X_S19S21dquality',
                            grazing_is_ok=0):
    # For KOIs

    df = get_gyro_data(sampleid, grazing_is_ok=grazing_is_ok, drop_highruwe=1)

    if 'deprecated' not in sampleid:
        assert len(df) == df['flag_is_gyro_applicable'].sum()
        assert len(df) == df['flag_is_ok_planetcand'].sum()

    n_pl = len(np.unique(df.kepoi_name))
    if 'deprecated' not in sampleid:
        n_st = len(np.unique(df.KIC))
        Prots = np.round(nparr(df.Prot), 4)
    else:
        n_st = len(np.unique(df.kepid))
        Prots = np.round(nparr(df.mean_period), 4)

    Teffs = nparr(df.b20t2_Teff)
    Teff_errs = nparr(df.adopted_Teff_err)
    Prot_errs = nparr(df.Prot_err)

    set_style("clean")
    fig, ax = plt.subplots(figsize=(3,3))

    model_ids = ['120-Myr', 'Praesepe', 'NGC-6811', '2.6-Gyr', 'M67']
    ages = ['120 Myr', '670 Myr', '1 Gyr', '2.6 Gyr', '4 Gyr']
    yvals = [9.8,14.8,16.7,20,28]

    _Teff = np.linspace(3800, 6200, int(1e3))
    linestyles = ['dotted', 'solid', 'dashed', 'dashdot', 'solid']
    for model_id, ls, age, yval in zip(model_ids, linestyles, ages, yvals):
        color = 'k'
        poly_order = 7
        _Prot = reference_cluster_slow_sequence(
            _Teff, model_id, poly_order=poly_order
        )
        if model_id == 'M67':
            print(
                reference_cluster_slow_sequence( nparr([5800]), model_id,
                                                poly_order=poly_order)
            )

        ax.plot(
            _Teff, _Prot, color=color, linewidth=0.8, zorder=10, alpha=0.6, ls=ls
        )

        bbox = dict(facecolor='white', alpha=1, pad=0, edgecolor='white')
        ax.text(3680, yval, age, ha='right', va='center', fontsize='x-small',
                bbox=bbox, zorder=49)

    print(f"Mean Teff error is {np.nanmean(Teff_errs):.1f} K")

    if 'deprecated' in sampleid:
        N_reported_periods = df['N_reported_periods']
        sel = (N_reported_periods >= 2)

        ax.errorbar(
            Teffs[sel], Prots[sel], #xerr=Teff_errs,
            yerr=Prot_errs[sel],
            marker='o', elinewidth=0.5, capsize=0, lw=0, mew=0.5, color='k',
            markersize=1, zorder=5
        )

    # only one reported period
    sel = np.ones(len(df)).astype(bool)
    c = 'k'
    if 'deprecated' in sampleid:
        c = 'lightgray'
        sel = N_reported_periods == 1
    ax.errorbar(
        Teffs[sel], Prots[sel], #xerr=Teff_errs,
        yerr=Prot_errs[sel],
        marker='o', elinewidth=0.5, capsize=0, lw=0, mew=0.5, color=c,
        markersize=1, zorder=4
    )

    txt = (
        "$N_\mathrm{p}$ = " + f"{n_pl}\n"
        "$N_\mathrm{s}$ = " + f"{n_st}"
    )
    ax.text(0.03, 0.97, txt, transform=ax.transAxes,
            ha='left',va='top', color='k')

    ax.set_xlabel("Effective Temperature [K]")
    ax.set_ylabel("Rotation Period [days]")
    ax.set_xlim([ 6300, 3700 ])
    ax.set_ylim([ -1, 48 ])

    s = ''
    if not grazing_is_ok:
        s += "dropgrazing"
    else:
        s += "keepgrazing"
    outpath = os.path.join(outdir, f'koi_mean_prot_teff_{sampleid}_{s}.png')
    savefig(fig, outpath)


def plot_star_Prot_Teff(outdir, sampleid):
    # For KIC / all Santos stars

    assert sampleid in [
        'Santos19_Santos21_all', 'teff_age_prot_seed42_nstar20000',
        'Santos19_Santos21_dquality', 'Santos19_Santos21_litsupp_all',
        'McQuillan2014only', 'McQuillan2014only_dquality'
    ]

    if "Santos" in sampleid:
        df = get_kicstar_data(sampleid)
        if sampleid == 'Santos19_Santos21_dquality':
            df = df[df['flag_is_gyro_applicable']]
        n_st = len(np.unique(df.KIC))

    elif "seed" in sampleid:
        csvdir = (
            '/Users/luke/Dropbox/proj/gyro-interp/results/period_bimodality'
        )
        df = pd.read_csv(
            join(csvdir, 'teff_age_prot_seed42_nstar20000.csv')
        )
        df['adopted_Teff'] = df.teff
        df['adopted_Teff_err'] = 1
        df['Prot'] = df.prot_mod
        df['Prot_err'] = 1
        n_st = len(df)

    elif "McQuillan2014only_dquality" in sampleid:
        df = get_kicstar_data(sampleid)
        if sampleid == 'McQuillan2014only_dquality':
            df = df[df['flag_is_gyro_applicable']]
        n_st = len(np.unique(df.KIC))

    elif "McQuillan2014only" in sampleid:
        from gyrojo.prot_uncertainties import get_empirical_prot_uncertainties
        fitspath = join(
            DATADIR, "literature", "McQuillan_2014_table1.fits"
        )
        df = Table(fits.open(fitspath)[1].data).to_pandas()
        df['adopted_Teff'] = df.Teff
        df['adopted_Teff_err'] = 100
        df['Prot'] = df['Prot']
        df['Prot_err'] = get_empirical_prot_uncertainties(
            np.array(df['Prot'])
        )
        n_st = len(np.unique(df.KIC))

    Teffs = nparr(df.adopted_Teff)
    Teff_errs = nparr(df.adopted_Teff_err)
    Prots = np.round(nparr(df.Prot), 4)
    if sampleid == 'Santos19_Santos21_all':
        # as reported; probably underestimated?
        Prot_errs = nparr(df.E_Prot)
    else:
        Prot_errs = nparr(df.Prot_err)

    set_style("clean")
    fig, ax = plt.subplots(figsize=(3,3))

    model_ids = ['120-Myr', 'Praesepe', 'NGC-6811', '2.6-Gyr', 'M67']
    ages = ['120 Myr', '670 Myr', '1 Gyr', '2.6 Gyr', '4 Gyr']
    yvals = [9.8,14.8,16.7,20,28]

    _Teff = np.linspace(3800, 6200, int(1e3))
    linestyles = ['dotted', 'solid', 'dashed', 'dashdot', 'solid']
    for model_id, ls, age, yval in zip(model_ids, linestyles, ages, yvals):
        color = 'k'
        poly_order = 7
        _Prot = reference_cluster_slow_sequence(
            _Teff, model_id, poly_order=poly_order
        )
        if model_id == 'M67':
            print(
                reference_cluster_slow_sequence( nparr([5800]), model_id,
                                                poly_order=poly_order)
            )

        ax.plot(
            _Teff, _Prot, color=color, linewidth=0.8, zorder=10, alpha=0.7, ls=ls
        )

        bbox = dict(facecolor='white', alpha=1, pad=0, edgecolor='white')
        ax.text(3680, yval, age, ha='right', va='center', fontsize='x-small',
                bbox=bbox, zorder=49)

    print(f"Mean Teff error is {np.nanmean(Teff_errs):.1f} K")

    DO_POINTS = 0
    if DO_POINTS:
        ax.errorbar(
            Teffs, Prots,
            yerr=np.zeros(len(Prots)),
            marker='o', elinewidth=0., capsize=0, lw=0, mew=0., color='k',
            markersize=0.5, zorder=5
        )

    else:
        # Define the bin sizes
        dTeff = 100/5
        dProt = 1/4

        # Create the 2D histogram
        hist, xedges, yedges, im = plt.hist2d(
            Teffs, Prots,
            bins=[np.arange(3700, 6300, dTeff),
                  np.arange(-1, 46, dProt)]
        )

        # Create a custom colormap with white color for zero values
        cmap = plt.cm.YlGnBu
        cmaplist = [cmap(i) for i in list(range(cmap.N))[20:-60]]
        #cmaplist = [cmap(i) for i in range(cmap.N)]
        cmaplist[0] = (1.0, 1.0, 1.0, 1.0)  # Set the color for zero values to white
        cmap = mcolors.LinearSegmentedColormap.from_list('Custom YlGnBu', cmaplist, cmap.N)

        ## Apply log scaling to the colorbar
        #norm = mcolors.LogNorm(vmin=1, vmax=np.max(hist))
        norm = mcolors.Normalize(vmin=0.5, vmax=7.2)
        im.set_norm(norm)

        # Update the colormap of the plot
        im.set_cmap(cmap)

        show_colorbar = 1
        if show_colorbar:
            axins1 = inset_axes(ax, width="20%", height="2%", loc='upper right',
                                borderpad=1.0)

            cb = fig.colorbar(im, cax=axins1, orientation="horizontal",
                              extend="max", norm=norm)
            #cb.set_ticks([1,4,7,10])
            cb.set_ticks(np.arange(1,8))
            cb.set_ticklabels([1,None,3,None,5,None,7])
            cb.ax.tick_params(labelsize='small')
            cb.ax.tick_params(size=1, which='both') # remove the ticks
            #cb.ax.yaxis.set_ticks_position('left')
            cb.ax.xaxis.set_label_position('top')
            cb.set_label("$N_\mathrm{stars}$", fontsize='small', weight='normal')


    txt = (
        "$N_\mathrm{s}$ = " + f"{n_st}"
    )
    bbox = dict(facecolor='white', alpha=1, pad=0, edgecolor='white')
    ax.text(0.03, 0.97, txt, transform=ax.transAxes,
            ha='left',va='top', color='k', zorder=6, bbox=bbox)

    ax.set_xlabel("Effective Temperature [K]")
    ax.set_ylabel("Rotation Period [days]")
    ax.set_xlim([ 6300, 3700 ])
    ax.set_ylim([ -1, 46 ])

    outpath = os.path.join(outdir, f'prot_teff_{sampleid}.png')
    savefig(fig, outpath)


def plot_li_vs_teff(outdir, sampleid=None, yscale=None,
                    limodel='eagles', show_dispersion=0, nodata=0,
                    show_dispersionpoints=0, IRON_OFFSET=7.5 ):

    #df = get_li_data('all')
    df = get_li_data(sampleid)

    n_pl = len(np.unique(df.kepoi_name))
    n_st = len(np.unique(df.kepid))

    Teffs = nparr(df.adopted_Teff)
    assert pd.isnull(df.adopted_Teff).sum() == 0
    Teff_errs = nparr(df.adopted_Teff_err)

    # CKS-Young lithium dataset
    li_ew = df['Fitted_Li_EW_mA'] - IRON_OFFSET
    li_ew_perr = np.abs(df['Fitted_Li_EW_mA_perr'])
    li_ew_merr = np.abs(df['Fitted_Li_EW_mA_merr'])
    # anything below this is an upper limit
    upperlim = (li_ew - li_ew_merr < 10)
    det = ~upperlim
    li_ew_upper_lims = li_ew[upperlim] + li_ew_perr[upperlim]
    if yscale in [None, 'linear']:
        li_ew_upper_lims[li_ew_upper_lims < 0] = 1

    if limodel == 'baffles':

        # baffles
        from baffles.readData import read_lithium
        bv_m, upper_lim, fits = read_lithium()

        # Pleiades, m34, Hyades, M67
        inds = nparr([4, 6, 8, 9])
        ages = [120, 240, 670, 4000]

        from baffles.li_constants import BV
        li_models = [10**fits[ix][0](BV) for ix in inds]
        li_pleiades = li_models[0]
        li_m34 = li_models[1]
        li_hyades = li_models[2]
        li_m67 = li_models[3]
        li_hyades[li_hyades > li_m34] = li_m34[li_hyades > li_m34]
        li_m67[li_m67 > li_hyades] = li_hyades[li_m67 > li_hyades]

        from cdips.utils.mamajek import get_interp_Teff_from_BmV
        Teff_model = get_interp_Teff_from_BmV(nparr(BV))

        # smooth pleiades model
        from scipy.interpolate import make_interp_spline, BSpline
        _inds = np.argsort(Teff_model)
        spl = make_interp_spline(
            Teff_model[_inds][::50], li_pleiades[_inds][::50], k=3
        )
        li_pleiades = spl(Teff_model)

        li_models = [li_pleiades, li_m34, li_hyades, li_m67]

    elif limodel == 'eagles':
        #
        # ported from eagles/eagles_iso.py
        #
        sys.path.append('/Users/luke/Dropbox/proj/eagles')

        # import the EWLi prediction model from the main EAGLES code
        from eagles import AT2EWm
        from eagles import eAT2EWm

        # set up a an equally spaced set of log temperatures between 3000 and 6500 K
        tstep = 0.00002
        Teff_model = np.arange(3.4772, 3.8130, tstep)
        Teff_model = np.linspace(3.4772, 3.8130, int(1e4))
        Teff_model = np.linspace(np.log10(3500), np.log10(6400), int(1e4))

        agestrs = None
        if not show_dispersion and not show_dispersionpoints:
            ages = [100, 500, 2000, 4000]
            agestrs = ['100 Myr', '500 Myr', '2 Gyr', '4 Gyr']
            colors = ['C0', 'C1', 'purple', 'lime']
        elif show_dispersionpoints:
            ages = [100, 500, 2000]
            agestrs = ['100 Myr', '500 Myr', '2 Gyr']
            colors = ['C0', 'C1', 'purple', 'whatever']
        else:
            ages = [100, 500, 2000, 4000]
            agestrs = ['100 Myr', '500 Myr', '2 Gyr', '4 Gyr']
            colors = ['C0', 'C1', 'purple', 'lime']
        lAges = [np.log10(t)+6 for t in ages]  # log age in years

        li_models = []
        li_model_dispersion = []
        for lAge in lAges:
            ewm = AT2EWm(Teff_model, lAge)
            eewm = eAT2EWm(Teff_model, lAge)
            li_models.append(ewm)
            li_model_dispersion.append(eewm)
        Teff_model = 10**Teff_model

    set_style("clean")
    fig, ax = plt.subplots(figsize=(3,3))
    np.random.seed(42)

    if not isinstance(agestrs, list):
        agestrs = [f"{a}" for a in ages]

    linestyles = ['solid', 'dotted', 'dashed', 'dashdot', 'solid']
    yvals = [110, 68.5, 38, 999]
    for ix, (li_model, ls, age, _c, agestr, yval) in enumerate(
        zip(li_models, linestyles, ages, colors, agestrs, yvals)
    ):
        color = 'k'
        alpha = 0.9
        if not show_dispersion and not show_dispersionpoints:
            alpha = 0.8
        if not show_dispersionpoints:
            ax.plot(
                Teff_model, li_model, color=_c, linewidth=1, zorder=10,
                alpha=alpha, ls=ls, label=agestr
            )
        else:
            ax.plot(
                Teff_model, li_model, color='k', linewidth=1, zorder=10,
                alpha=alpha, ls=ls, label=agestr
            )

        if show_dispersion :
            ax.fill_between(
                Teff_model,
                li_model-li_model_dispersion[ix],
                li_model+li_model_dispersion[ix],
                alpha=0.3
            )

        elif show_dispersionpoints:

            sigma = li_model_dispersion[ix]/(2**0.5)
            loc = li_model

            Li_mod = np.random.normal(
                loc=li_model, scale=sigma, size=len(li_model)
            )

            ax.scatter(
                Teff_model, Li_mod, linewidths=0,
                marker='o', color=_c,
                s=0.4, rasterized=True
            )

            # annotate means
            bbox = dict(facecolor='white', alpha=1, pad=0, edgecolor='white')
            ax.text(6100, yval, agestr, ha='left', va='center', fontsize='x-small',
                    bbox=bbox, zorder=49, color=_c)

            txt = r"Model: ${\tt eagles}$"
            ax.text(0.97, 0.97, txt, transform=ax.transAxes,
                    ha='right',va='top', color='k', bbox=bbox)



    print(f"Mean Teff error is {np.nanmean(Teff_errs):.1f} K")

    yerr = np.array(
        [li_ew_merr[det], li_ew_perr[det]]
    ).reshape((2, len(li_ew[det])))

    if not nodata:
        N_det = len(Teffs[det])
        ax.errorbar(
            Teffs[det], li_ew[det], #xerr=Teff_errs,
            yerr=yerr,
            marker='o', elinewidth=0.25, capsize=0, lw=0, mew=0.5, color='k',
            markersize=1, zorder=5
        )

        N_upperlim = len(Teffs[upperlim])
        ax.scatter(
            Teffs[upperlim], li_ew_upper_lims,
            marker='$\downarrow$', s=2, color='k', zorder=4,
            linewidths=0
        )
        print(f"N_det: {N_det}")
        print(f"N_upperlim: {N_upperlim}")

    if not show_dispersionpoints:
        leg = ax.legend(loc='upper right', handletextpad=0.3, fontsize='small',
                        framealpha=0, borderaxespad=0, borderpad=0,
                        handlelength=1.6, bbox_to_anchor=(0.97, 0.97))

    txt = (
        "$N_\mathrm{p}$ = " + f"{n_pl}\n"
        "$N_\mathrm{s}$ = " + f"{n_st}"
    )
    if not nodata:
        ax.text(0.04, 0.97, txt, transform=ax.transAxes,
                ha='left',va='top', color='k')

    ax.set_ylabel('Li$_{6708}$ EW [m$\mathrm{\AA}$]')
    ax.set_xlabel('Effective Temperature [K]')

    if yscale == 'log':
        ax.set_yscale('log')
        ax.set_ylim([1, 300])

    ax.set_xlim([ 6300, 3700 ])
    ax.set_ylim([ -5, 270 ])
    if show_dispersionpoints:
        ax.set_ylim([ -5, 320 ])
    if not show_dispersion and not show_dispersionpoints:
        ax.set_ylim([-10, 270])

    s = f'_{sampleid}'
    s += f"_{limodel}"
    if yscale == 'log':
        s += "_logy"
    if show_dispersion:
        s += "_showdispersion"
    if show_dispersionpoints:
        s += "_showpoints"
    if nodata:
        s += "_nodata"

    outpath = os.path.join(outdir, f'li_vs_teff{s}.png')
    savefig(fig, outpath)


def plot_rp_vs_age(outdir, xscale='linear', elinewidth=0.1, shortylim=0,
                   lowmes=0):
    # and plot_rp_ks_test

    # get data
    _df, d, _ = get_age_results('gyro')
    df = pd.DataFrame(d)
    _outpath = join(outdir, "temp.csv")
    df.to_csv(_outpath, index=False)

    plt.close("all")
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    AGE_MAX = 4e9
    bins = np.linspace(0, AGE_MAX, 26)
    ax.hist(df.age, bins=bins, color='darkgray')
    ax.hlines(len(df)/len(bins), 0, AGE_MAX, zorder=99, color='k', ls=':')
    ax.update({'xlim':[0,4e9], 'xlabel': 'Age [yr]', 'ylabel': 'Count'})
    outpath = os.path.join(outdir, f'age_hist_nocleaning.png')
    savefig(fig, outpath)

    # >25% radii
    sel = (df['rp']/df['rp_err1'] > 4) & (df['rp']/df['rp_err2'] > 4)
    # adopted age < 2.5 gyr
    AGE_MAX = 10**9.5
    sel &= df['age'] < AGE_MAX
    # non-crap age in at least one direction
    sel &= ((df['age']/df['age_err1'] > 1) | (df['age']/df['age_err2'] > 1))

    df = df[sel]

    plt.close("all")
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    bins = np.linspace(0, AGE_MAX, 13)
    ax.hist(df.age, bins=bins, color='darkgray')
    ax.hlines(len(df)/len(bins), 0, AGE_MAX, zorder=99, color='k', ls=':')
    ax.update({'xlabel': 'Age [yr]', 'ylabel': 'Count'})
    outpath = os.path.join(outdir, f'age_hist_lt2gyr_25pctradii_singlesidedok.png')
    savefig(fig, outpath)

    plt.close("all")
    set_style("clean")
    if elinewidth == 0:
        fig, ax = plt.subplots(figsize=(3.3, 5))
    else:
        fig, ax = plt.subplots(figsize=(3.3, 2.5))

    if xscale == 'linear':
        agescale = 1e9
    else:
        agescale = 1

    ax.errorbar(
        df['age']/agescale, df['rp'],
        yerr=nparr([df['rp_err2'], df['rp_err1']]),
        xerr=nparr([df['age_err2'], df['age_err1']])/agescale,
        marker='o', elinewidth=elinewidth, capsize=0, lw=0, mew=0.5,
        color='k', markersize=2, zorder=5
    )
    if lowmes:
        assert np.min(df.mes) > 10
        MES_CUT = np.nanpercentile(df.mes, 20)
        sdf = df[df['mes'] < MES_CUT]
        ax.errorbar(
            sdf['age']/agescale, sdf['rp'],
            yerr=nparr([sdf['rp_err2'], sdf['rp_err1']]),
            xerr=nparr([sdf['age_err2'], sdf['age_err1']])/agescale,
            marker='o', elinewidth=elinewidth, capsize=0, lw=0, mew=0.5,
            color='crimson', markersize=2, zorder=6
        )

        txt = f'MES = 10-{MES_CUT:.1f}'
        ax.text(0.03, 0.03, txt, transform=ax.transAxes,
                ha='left', va='bottom', color='crimson', fontsize='small')


    if elinewidth != 0 :
        ax.hlines(
            1.8, 0.1e9/agescale, AGE_MAX/agescale, colors='lightgray', alpha=1,
            linestyles='-', zorder=-2, linewidths=2
        )

    ax.set_yscale('log')

    ax.set_xscale(xscale)
    ax.set_xlim([-0.1e9/agescale, (AGE_MAX+0.2e9)/agescale])

    if shortylim:
        ax.set_ylim([0.3, 10])

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.set_yticks([1,2,4,10])
    ax.set_yticklabels([1,2,4,10], fontsize='medium')

    #ax.set_xticks([1,10,100])
    #ax.set_xticklabels([1,10,100], fontsize='medium')

    if xscale == 'linear':
        ax.set_xlabel('Age [gigayears]', fontsize='medium')
    else:
        ax.set_xlabel('Age [years]', fontsize='medium')
    ax.set_ylabel('Planet radius [Earths]', fontsize='medium')

    s = f"_{xscale}"
    if elinewidth != 0.1:
        s += f"_elinewidth{elinewidth}"
    if shortylim:
        s += f"_shortylim"
    if lowmes:
        s += f"_lowmes"
    outpath = os.path.join(outdir, f'rp_vs_age{s}.png')

    if elinewidth != 0:
        savefig(fig, outpath)
    else:
        fig.tight_layout()
        fig.savefig(outpath, dpi=400, transparent=True)

    # KS test!
    from scipy import stats

    agemins = [2e8, 3e8, 4e8, 5e8, 6e8, 7e8, 8e8, 9e8, 1e9]

    for agemin in agemins:

        agemin_myr = int(agemin/1e6)
        outdir = join(RESULTSDIR, "rp_agebins_KS_test")
        if not os.path.exists(outdir): os.mkdir(outdir)
        outpath = os.path.join(outdir, f'rp_agebins_KS_test_agemin{agemin_myr}.png')
        if os.path.exists(outpath):
            continue

        sel_young = (df['age'] < agemin) & (df['rp'] <= 10)
        sel_old = (df['age'] > 1e9) & (df['age'] < 2e9) & (df['rp'] <= 10)

        plt.close("all")
        set_style("clean")
        fig, ax = plt.subplots(figsize=(3.3, 2.5))

        labels = [f'<{agemin_myr} Myr', '1-2 Gyr']
        for _sel, l in zip([sel_young, sel_old], labels):

            counts, bin_edges = np.histogram(
                df.loc[_sel, 'rp'], bins=len(df[_sel]), normed=True
            )
            cdf = np.cumsum(counts)
            ax.plot(bin_edges[1:], cdf/cdf[-1], lw=0.5, label=l)

        ax.update({'ylabel':'Fraction', 'xlabel':'Planet size [Earths]'})
        ax.legend(loc='lower right', fontsize='x-small')

        D, p_value = stats.ks_2samp(
            df.loc[sel_young, 'rp'], df.loc[sel_old, 'rp']
        )
        n_pl_y = len(df[sel_young])
        n_pl_o = len(df[sel_old])
        txt0 = "$N_\mathrm{p,young}$ = " + f"{n_pl_y}, "
        txt1 = "$N_\mathrm{p,old}$ = " + f"{n_pl_o}\n"
        txt2 = 'D={:.2f},p={:.4f}\nfor 2 sample KS'.format(D, p_value)
        txt = txt0 + txt1 + txt2
        ax.text(0.9, 0.5, txt,
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize='xx-small')
        savefig(fig, outpath)


def plot_rp_vs_porb_binage(outdir, teffcondition='allteff'):

    # get data
    whichtype = 'gyro' #'allageinfo' # or "gyro"
    _df, d, st_ages = get_age_results(
        whichtype=whichtype, grazing_is_ok=0, drop_highruwe=1
    )
    df = pd.DataFrame(d)

    # >33% radii
    #sel = (df['rp']/df['rp_err1'] > 3) & (df['rp']/df['rp_err2'] > 3)
    # >25% radii
    #sel = (df['rp']/df['rp_err1'] > 4) & (df['rp']/df['rp_err2'] > 4)
    # >20% radii
    sel = (df['rp']/df['rp_err1'] > 5) & (df['rp']/df['rp_err2'] > 5)
    #AGE_MAX = 10**9.5 #(3.2 gyr)
    AGE_MAX = 4e9

    sel &= df['age'] < AGE_MAX
    if teffcondition == 'allteff':
        pass
    elif teffcondition == 'sunliketeff':
        sel &= df['adopted_Teff'] >= 5500
        sel &= df['adopted_Teff'] <= 6000
    elif teffcondition == 'teffgt5000':
        sel &= df['adopted_Teff'] >= 5000
    elif teffcondition == 'tefflt5000':
        sel &= df['adopted_Teff'] < 5000

    df = df[sel]
    if whichtype == 'gyro':
        st_ages = st_ages[st_ages < AGE_MAX]

    df['age_pcterravg'] = np.nanmean(
        [df['age_pcterr1'], df['age_pcterr2']], axis=0
    )
    df['age_pcterrmax'] = np.max(
        [df['age_pcterr1'], df['age_pcterr2']], axis=0
    )
    print(df.describe())

    age_bins = [
        (0, AGE_MAX),
        # duo
        #(0, np.nanpercentile(st_ages, 100/2)),
        #(np.nanpercentile(st_ages, 100/2), AGE_MAX),
        # triple
        (0, 1e9),
        (1e9, 2e9),
        (2e9, 3e9),
        (3e9, 4e9),
        #(0, 667e6),
        #(667e6, 1333e6),
        #(1333e6, 2000e6)
        #
        ## logbinning...
        #(0, 10**8.75),
        #(10**8.75, 10**9),
        #(10**9, 10**9.25),
        #(10**9.25, 10**9.5),
        #(0, np.nanpercentile(st_ages, 100/3)),
        #(np.nanpercentile(st_ages, 100/3), np.nanpercentile(st_ages, 2*100/3)),
        #(np.nanpercentile(st_ages, 2*100/3), AGE_MAX),
        # quad
        #(0, np.nanpercentile(st_ages, 100/4)),
        #(np.nanpercentile(st_ages, 100/4), np.nanpercentile(st_ages, 2*100/4)),
        #(np.nanpercentile(st_ages, 2*100/4), np.nanpercentile(st_ages, 3*100/4)),
        #(np.nanpercentile(st_ages, 3*100/4), AGE_MAX),
    ]

    for ix, age_bin in enumerate(age_bins):

        plt.close("all")
        set_style("clean")
        fig, ax = plt.subplots(figsize=(3.3, 2.5))

        elw = 0.1
        ax.errorbar(
            df['period'], df['rp'],
            yerr=nparr([df['rp_err2'], df['rp_err1']]),
            marker='o', elinewidth=elw, capsize=0, lw=0, mew=0.,
            color='darkgray', markersize=1.5, zorder=5, alpha=0.5
        )

        lo_age, hi_age = age_bin[0], age_bin[1]
        sel = (df['age'] > lo_age) & (df['age'] <= hi_age)

        ax.errorbar(
            df.loc[sel, 'period'], df.loc[sel, 'rp'],
            yerr=nparr([df.loc[sel, 'rp_err2'], df.loc[sel, 'rp_err1']]),
            marker='o', elinewidth=elw, capsize=0, lw=0, mew=0.5,
            color='black', markersize=2, zorder=6
        )

        # van eylen+2018
        log10Pmod = np.log10(np.logspace(0,2,50))
        m = -0.09
        a = 0.37
        log10Rmod = m*log10Pmod + a
        OFFSET = -0.25
        Rmod = 10**log10Rmod + OFFSET

        SHOW_VE2018 = 0
        if SHOW_VE2018:
            ax.plot(
                10**log10Pmod, Rmod, c='lightgray', alpha=1, zorder=-2, lw=1,
                ls=':'
            )

        #ax.hlines(
        #    1.8, 1, 100, colors='lightgray', alpha=1,
        #    linestyles='-', zorder=-2, linewidths=2
        #)

        df = get_planet_class_labels(df, OFFSET=OFFSET)

        n_pl = len(df.loc[sel, 'period'])
        n_sn = int(np.sum( df[sel].pl_class == 'Mini-Neptunes' ))
        n_se = int(np.sum( df[sel].pl_class == 'Super-Earths' ))
        try:
            θ = n_se / n_sn
        except ZeroDivisionError:
            θ = np.nan

        DO_KEYNOTE_LABEL = 0
        if teffcondition == 'teffgt5000':
            tstr = ', Teff $\geq$ 5000 K'
        elif teffcondition == 'tefflt5000':
            tstr = ', Teff $<$ 5000 K'
        elif teffcondition == 'sunliketeff':
            tstr = ', Teff 5500-6000 K'
        else:
            tstr = ''

        if DO_KEYNOTE_LABEL:
            txt = f'{int(lo_age/1e9)} to {int(hi_age/1e9)} Gyr{tstr}\n'
        else:
            txt = f'$t$ = {lo_age:.1e} to {hi_age:.1e} yr\n'

        txt += (
            "$N_\mathrm{p}$ = " + f"{n_pl}/{len(df)}; "
            "$N_\mathrm{SE}/N_\mathrm{MN}$ = "
            f"{n_se}/{n_sn}={θ:.2f}"
        )

        ax.text(0.97, 0.03, txt, transform=ax.transAxes,
                ha='right', va='bottom', color='k', fontsize='small')

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_ylim([0.3, 10])
        ax.set_xlim([0.38, 230])

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        ax.set_yticks([1,2,4,10])
        ax.set_yticklabels([1,2,4,10], fontsize='medium')

        ax.set_xticks([1,10,100])
        ax.set_xticklabels([1,10,100], fontsize='medium')

        ax.set_xlabel('Orbital period [days]', fontsize='medium')
        ax.set_ylabel('Planet radius [Earths]', fontsize='medium')

        s = f'_{ix}_{lo_age:.2e}_{hi_age:.2e}_{teffcondition}'
        outpath = os.path.join(outdir, f'rp_vs_porb_binage{s}.png')
        savefig(fig, outpath)


def plot_koi_gyro_posteriors(outdir, cache_id):

    from gyrointerp.helpers import get_summary_statistics

    csvdir = os.path.join(LOCALDIR, "gyrointerp", cache_id)
    csvpaths = glob(os.path.join(csvdir, "*posterior.csv"))
    assert len(csvpaths) > 0

    # plot all stars
    plt.close("all")
    set_style('clean')
    fig, ax = plt.subplots()

    kepoi_names = []
    summaries = []

    for ix, csvpath in enumerate(csvpaths):

        kepoi_name = os.path.basename(csvpath).split("_")[0]
        kepoi_names.append(kepoi_name)

        df = pd.read_csv(csvpath)
        t_post = np.array(df.age_post)
        age_grid = np.array(df.age_grid)

        d = get_summary_statistics(age_grid, t_post)
        print(d)
        summaries.append(d)

        zorder = ix
        ax.plot(age_grid, 1e3*t_post/np.trapz(t_post, age_grid), alpha=0.1,
                lw=0.3, c='k', zorder=zorder)

    xmin = 0
    xmax = 4000
    ax.update({
        'xlabel': 'Age [Myr]',
        'ylabel': 'Probability ($10^{-3}\,$Myr$^{-1}$)',
        'xlim': [xmin, xmax],
        'ylim': [-0.5, 10.5]
    })
    outpath = os.path.join(outdir, f'step0_posteriors_verification_peaklt2600.png')
    savefig(fig, outpath, writepdf=1, dpi=400)

    df = pd.DataFrame(summaries, index=kepoi_names)
    df['kepoi_name'] = kepoi_names
    csvpath = os.path.join(outdir, "step0_koi_gyro_ages.csv")
    df.to_csv(csvpath, index=False)
    print(f"Wrote {csvpath}")

    _csvpath = os.path.join(DATADIR, "interim",
                           "koi_table_X_GDR3_B20_S19_S21_M14_M15.csv")
    kdf = pd.read_csv(_csvpath)

    mdf = df.merge(kdf, how='inner', on='kepoi_name')
    assert len(mdf) == len(df)

    # Contents: for the 864
    csvpath = os.path.join(outdir, "step0_koi_gyro_ages_X_GDR3_B20_S19_S21_M14_M15.csv")
    mdf = mdf.sort_values(by='median')
    mdf.to_csv(csvpath, index=False)
    print(f"Wrote {csvpath}")

    cols = ['kepoi_name', 'kepler_name', 'median', '+1sigma', '-1sigma',
            '+2sigma', '-2sigma', "+1sigmapct", "-1sigmapct", "koi_prad"]
    sel_2s = mdf['median'] + mdf['+2sigma'] < 1000

    print(mdf[sel_2s][cols])


def plot_process_koi_li_posteriors(outdir, cache_id, sampleid, li_method='eagles'):

    from gyrointerp.helpers import get_summary_statistics

    csvdir = os.path.join(RESULTSDIR, cache_id)
    if li_method == 'baffles':
        csvpaths = glob(os.path.join(csvdir, "*lithium.csv"))
        raise NotImplementedError(
            'deprecated, and would need a way to propagate Li EWs...'
        )
    elif li_method == 'eagles':
        csvpaths = glob(os.path.join(csvdir, "*_pos.csv"))

    assert len(csvpaths) > 0

    # plot all stars
    plt.close("all")
    set_style('clean')
    fig, ax = plt.subplots()

    kepoi_names = []
    summaries = []

    for ix, csvpath in enumerate(csvpaths):

        kepoi_name = os.path.basename(csvpath).split("_")[0]
        eaglesrespath = join(csvdir, f"{kepoi_name}.csv")
        assert os.path.exists(eaglesrespath)
        kepoi_names.append(kepoi_name)

        df = pd.read_csv(csvpath, names=['age_grid','age_post'], comment='#')
        eaglesdf = pd.read_csv(eaglesrespath)

        t_post = np.array(df.age_post)

        if li_method == 'baffles':
            age_grid = np.array(df.age_grid) # already in myr
        elif li_method == 'eagles':
            age_grid = 10**np.array(df.age_grid) / (1e6) # convert to myr

        d = get_summary_statistics(age_grid, t_post)
        # write the Li EWs that eagles / baffles actually used
        cols = ['ID', 'Teff', 'eTeff', 'LiEW', 'eLiEW', 'lApk', 'siglo',
                'sighi', 'limup', 'limlo', 'lMed']
        for c in cols:
            d[f"eagles_{c}"] = eaglesdf[c].iloc[0]
        print(d)
        summaries.append(d)

        zorder = ix
        ax.plot(age_grid, 1e3*t_post/np.trapz(t_post, age_grid), alpha=0.1,
                lw=0.3, c='k', zorder=zorder)

    xmin = 0
    xmax = 1500
    ax.update({
        'xlabel': 'Age [Myr]',
        'ylabel': 'Probability ($10^{-3}\,$Myr$^{-1}$)',
        'xlim': [xmin, xmax],
    })
    outpath = os.path.join(outdir, f'{li_method}_lithium_posteriors_verification.png')
    savefig(fig, outpath, writepdf=1, dpi=400)

    df = pd.DataFrame(summaries, index=kepoi_names)
    for c in df.columns:
        if c != 'kepoi_name':
            df = df.rename({c: f'li_{c}'}, axis='columns')
    df['kepoi_name'] = kepoi_names
    csvpath = os.path.join(outdir, f"{li_method}_koi_lithium_ages.csv")
    df.to_csv(csvpath, index=False)
    print(f"Wrote {csvpath}")

    _csvpath = join(DATADIR, "interim", f"koi_jump_getter_{sampleid}.csv")
    _kdf = pd.read_csv(_csvpath)
    _kdf = _kdf.sort_values(by=['kepoi_name','counts'],ascending=[True,False])
    kdf = _kdf[~_kdf.duplicated('kepoi_name', keep='first')]

    _mdf = df.merge(kdf, how='inner', on='kepoi_name')

    # Check if "li_median" agree for duplicate kepoi_name entries...
    result = _mdf.groupby('kepoi_name')['li_median'].apply(
        lambda x: x.nunique() == 1
    )
    all_match = result.all()
    assert all_match

    # If true, drop duplicates
    mdf = _mdf[~_mdf.duplicated('kepoi_name', keep='first')]

    # If this fails, did you remember to clear out
    # "/results/koi_lithium_posteriors_eagles_20240405", or its analog?
    assert len(mdf) == len(df)

    # Write lithium result contents
    csvpath = join(
        outdir, f"{li_method}_{sampleid}_lithium_ages.csv"
    )
    mdf = mdf.sort_values(by='li_median')
    mdf.to_csv(csvpath, index=False)
    print(f"Wrote {csvpath}")

    cols = ['kepoi_name', 'kepler_name', 'li_median', 'li_+1sigma', 'li_-1sigma',
            'li_+2sigma', 'li_-2sigma', "li_+1sigmapct", "li_-1sigmapct", "koi_prad"]
    sel_2s = mdf['li_median'] + mdf['li_+2sigma'] < 1000

    print(mdf[sel_2s][cols])



def plot_age_comparison(outdir, logscale=1, iso_v_gyro=0, ratio_v_gyro=0,
                        hist_age_unc_ratio=0):

    # get data
    df = get_age_results(whichtype='gyro', COMPARE_AGE_UNCS=1,
                         grazing_is_ok=0, drop_highruwe=1)

    # plot all stars
    plt.close("all")
    set_style('clean')
    fig, ax = plt.subplots()

    sel = (
        (~pd.isnull(df['adopted_age_median']))
        &
        (~pd.isnull(df['P22S_age-iso']))
    )

    if iso_v_gyro:

        ax.errorbar(
            1e6*df.loc[sel, 'adopted_age_median'], 1e9*df.loc[sel, 'P22S_age-iso'],
            xerr=1e6*nparr([df.loc[sel, 'adopted_age_-1sigma'],
                        df.loc[sel, 'adopted_age_+1sigma']]),
            yerr=1e9*nparr([np.abs(df.loc[sel, 'P22S_e_age-iso']),
                            df.loc[sel, 'P22S_E_age-iso']]),
            marker='o', elinewidth=0.1, capsize=0, lw=0, mew=0.5,
            color='black', markersize=1.5, zorder=6
        )

        ax.plot(
            np.logspace(1,10), np.logspace(1,10), c='lightgray', alpha=1,
            zorder=-2, lw=2
        )

    if ratio_v_gyro:

        ax.errorbar(
            1e6*df.loc[sel, 'adopted_age_median'],
            1e9*df.loc[sel, 'P22S_age-iso'] / (1e6*df.loc[sel, 'adopted_age_median']),
            xerr=1e6*nparr([df.loc[sel, 'adopted_age_-1sigma'],
                            df.loc[sel, 'adopted_age_+1sigma']]),
            #yerr=1e9*nparr([np.abs(df.loc[sel, 'P22S_e_age-iso']),
            #                df.loc[sel, 'P22S_E_age-iso']]),
            marker='o', elinewidth=0.1, capsize=0, lw=0, mew=0.5,
            color='black', markersize=1.5, zorder=6
        )

        ax.hlines(1, 1e7, 3e10, zorder=-2, color='lightgray', lw=2)

    if hist_age_unc_ratio:

        gyro_m1sig = np.abs(1e6*df.loc[sel, 'adopted_age_-1sigma'])
        gyro_p1sig = 1e6*df.loc[sel, 'adopted_age_+1sigma']
        gyro_med = 1e6*df.loc[sel, 'adopted_age_median']

        gyro_p1rel = gyro_p1sig / gyro_med
        gyro_m1rel = gyro_m1sig / gyro_med

        gyro_p1sig[gyro_p1rel<0.1] = 0.1*gyro_med
        gyro_m1sig[gyro_m1rel<0.1] = 0.1*gyro_med

        iso_m1sig = np.abs(1e9*df.loc[sel, 'P22S_e_age-iso'])
        iso_p1sig = 1e9*df.loc[sel, 'P22S_E_age-iso']

        iso_gyro_p1sig_ratio = iso_p1sig / gyro_p1sig
        iso_gyro_m1sig_ratio = iso_m1sig / gyro_m1sig
        print(42*'-')
        print("+1σ...")
        print(iso_gyro_p1sig_ratio.describe())
        print("-1σ...")
        print(iso_gyro_m1sig_ratio.describe())
        print(42*'-')

        #bins = np.logspace(-1.5,3,10)
        bins = np.logspace(-1.5,3,19)
        ax.hist(iso_gyro_p1sig_ratio, bins=bins, color='C0', alpha=0.5,
                density=False, histtype='step')
        ax.hist(iso_gyro_m1sig_ratio, bins=bins, color='C1', alpha=0.5,
                density=False, histtype='step')


    n_st = len(np.unique(df[sel].kepoi_number_str))
    n_pl = len(np.unique(df[sel].kepoi_name))
    txt = (
        "$N_\mathrm{p}$ = " + f"{n_pl}\n"
        "$N_\mathrm{s}$ = " + f"{n_st}"
    )
    if iso_v_gyro:
        xloc, yloc, va, ha = 0.97, 0.03, 'bottom', 'right'
    if ratio_v_gyro:
        xloc, yloc, va, ha = 0.97, 0.97, 'top', 'right'
    if hist_age_unc_ratio:
        xloc, yloc, va, ha = 0.05, 0.97, 'top', 'left'

    ax.text(xloc, yloc, txt, transform=ax.transAxes, ha=ha, va=va, color='k')

    if hist_age_unc_ratio:
        ax.text(0.97, 0.97, '+1σ', transform=ax.transAxes, ha='right',
                va='top', color='C0')
        ax.text(0.97, 0.92, '-1σ', transform=ax.transAxes, ha='right',
                va='top', color='C1')

    if iso_v_gyro:
        ax.set_xlabel("Rotation Age [years]")
        ax.set_ylabel("P+22 Isochrone Age [years]")
    if ratio_v_gyro:
        ax.set_xlabel("Rotation Age [years]")
        ax.set_ylabel("(P+22 Isochrone Age)/(Rotation Age)")
    if hist_age_unc_ratio:
        ax.set_xlabel(r"Age Precision Gain, σ$_{\rm Iso}^{\rm P\!+\!\!22}$ / σ$_{\rm Rotn}^{\rm This\ Work}$")
        ax.set_ylabel("Count")

    if logscale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    if iso_v_gyro and logscale:
        ax.set_xlim([3e7, 20e9])
        ax.set_ylim([3e7, 20e9])

    if iso_v_gyro and not logscale:
        ax.set_xlim([0, 10e9])
        ax.set_ylim([0, 10e9])


    s = ''
    if logscale:
        s += '_logscale'
    if iso_v_gyro:
        s += '_iso_v_gyro'
    if ratio_v_gyro:
        s += '_ratio_v_gyro'
    if hist_age_unc_ratio:
        s += '_hist_age_unc_ratio'

    outpath = os.path.join(outdir, f'age_comparison{s}.png')
    savefig(fig, outpath)


def plot_reinhold_2015(outdir):
    # histogram of t_MH08 from ReinholdGizon2015

    fitspath = join(DATADIR, "literature",
                    "Reinhold_2015_20934_stars.fits")
    hl = fits.open(fitspath)
    df = Table(hl[1].data).to_pandas()

    bins = np.arange(0,5e3+1e2,1e2)

    fig, ax = plt.subplots()
    set_style("clean")

    ax.hist(df.tMH08, bins=bins)
    ax.set_xlabel("t MH08 [myr]")
    ax.set_ylabel('count')

    outpath = os.path.join(outdir, "Reinhold_2015_t_MH08_hist.png")
    savefig(fig, outpath)


def get_sigma_range(result, median_age_range):
    start, end = median_age_range
    mid = start + 0.5*(end-start)
    mask = (result['median_age'] >= start) & (result['median_age'] <= end)
    filtered_result = result[mask]

    mean_plus_sigma = filtered_result['+1sigma'].mean() - mid
    mean_minus_sigma = mid - filtered_result['-1sigma'].mean()

    return mean_plus_sigma, mean_minus_sigma


def add_gradient_patch(ax, xmin, xmax, ymin, ymax, resolution=100):

    # Create a grayscale gradient colormap
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "black"])

    # Create a rectangle patch with the specified coordinates
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor="none", edgecolor="none")
    ax.add_patch(rect)

    # Create a linear space for the gradient
    gradient_array = np.linspace(0, 1, resolution)

    # Reshape the gradient array to a 2D matrix with 1 row
    gradient_array = gradient_array.reshape(1, -1)

    # Repeat the gradient array to create a 2D image
    gradient_image = np.repeat(gradient_array, 2, axis=0)

    # Create a gradient patch using the rectangle dimensions and the high-resolution gradient image
    gradient = ax.imshow(
        gradient_image,
        cmap=cmap,
        aspect="auto",
        extent=(xmin, xmax, ymin, ymax),
        alpha=0.5,
        zorder=0,
    )

    # Set the transform to match the data coordinates
    gradient.set_transform(ax.transData)


def fit_line_and_print_results(bin_centers, heights, poisson_uncertainties):
    # Select data where bin_centers < 2.5
    mask = bin_centers < 3
    x_data = bin_centers[mask]
    y_data = heights[mask]
    yerr_data = poisson_uncertainties[mask]

    # Fit a line using numpy polyfit with weighted least squares
    coefficients, cov = np.polyfit(x_data, y_data, 1, w=1/yerr_data**2, cov=True)
    slope = coefficients[0]
    y_intercept = coefficients[1]

    # Calculate x-intercept
    x_intercept = -y_intercept / slope

    # Calculate standard errors (1-sigma uncertainties)
    slope_se = np.sqrt(cov[0, 0])
    y_intercept_se = np.sqrt(cov[1, 1])
    x_intercept_se = np.sqrt((y_intercept_se/slope)**2 + (y_intercept*slope_se/(slope**2))**2)

    print(f"Slope: {slope:.3f}")
    print(f"Slope 1-sigma uncertainty: {slope_se:.3f}")
    print(f"Y-intercept: {y_intercept:.3f}")
    print(f"Y-intercept 1-sigma uncertainty: {y_intercept_se:.3f}")
    print(f"X-intercept: {x_intercept:.3f}")
    print(f"X-intercept 1-sigma uncertainty: {x_intercept_se:.3f}")


def plot_hist_field_gyro_ages(outdir, cache_id, MAXAGE=4000,
                              datestr='20240821', s19s21only=0,
                              preciseagesonly=0, cache_id1=None, datestr1=None,
                              dropfraclongrot=0):

    from gyrointerp.paths import CACHEDIR
    csvdir = join(CACHEDIR, f"samples_field_gyro_posteriors_{datestr}")
    flag_mcq14_comp = False
    if cache_id1 is not None and datestr1 is not None:
        flag_mcq14_comp = True

    mergedcsv = join(csvdir, f'merged_{cache_id}_samples_{datestr}.csv')

    if flag_mcq14_comp:
        csvdir1 = join(CACHEDIR, f"samples_field_gyro_posteriors_{datestr1}")
        mergedcsv1 = join(csvdir1, f'merged_{cache_id1}_samples_{datestr1}.csv')

    if not os.path.exists(mergedcsv):

        csvpaths = glob(join(csvdir, "*samples.csv"))
        assert len(csvpaths) > 0
        N_post_samples = 10*len(csvpaths)

        df_list = []
        for ix, f in enumerate(csvpaths):

            if ix % 100 == 0:
                print(f"{ix}/{len(csvpaths)}")

            bn = os.path.basename(f)
            kic_id = bn.split("_")[0]
            prot = float(bn.split("_")[1].lstrip("Prot"))
            teff = float(bn.split("_")[2].lstrip("Teff"))

            this_df = pd.read_csv(f)
            this_df['KIC'] = kic_id
            this_df['Prot'] = prot
            this_df['Teff'] = teff

            df_list.append(this_df)

        mdf = pd.concat(df_list)
        mdf.to_csv(mergedcsv, index=False)
        print(f"Wrote {mergedcsv}")

    else:
        mdf = pd.read_csv(mergedcsv)
        N_post_samples = len(mdf)

    if flag_mcq14_comp:
        mdf1 = pd.read_csv(mergedcsv1)

    if dropfraclongrot:
        N_before = len(mdf)
        # Randomly select 20% of these rows to drop
        filtered_mdf = mdf[mdf['Prot'] > 20]
        drop_indices = filtered_mdf.sample(frac=0.2, random_state=42).index
        mdf = mdf.drop(drop_indices).reset_index(drop=True)
        N_after = len(mdf)
        print(f'Droping 20% of Prot>20days.  Nbefore={N_before}, Nafter={N_after}')

    print(f"Got {N_post_samples} posterior samples...")

    if 'McQ14' not in datestr:
        kdf = get_gyro_data("Santos19_Santos21_dquality", grazing_is_ok=1)
        mcqstr = ''
    else:
        kdf = get_gyro_data("McQ14_dquality", grazing_is_ok=1)
        mcqstr = 'mcquillanonly'
    if flag_mcq14_comp:
        kdf1 = get_gyro_data("McQ14_dquality", grazing_is_ok=1)

    if s19s21only:
        assert 'McQ14' not in datestr
        skdf = kdf[(kdf.flag_is_gyro_applicable) & (~kdf.flag_Prot_provenance)]
        santosstr = '_s19s21only'
    else:
        skdf = kdf[(kdf.flag_is_gyro_applicable)]
        santosstr = ''
    if flag_mcq14_comp:
        skdf1 = kdf1[(kdf1.flag_is_gyro_applicable)]

    skdf['KIC'] = skdf.KIC.astype(str)
    mdf['KIC'] = mdf.KIC.astype(str)
    if flag_mcq14_comp:
        skdf1['KIC'] = skdf1.KIC.astype(str)

    if preciseagesonly:
        mdf = mdf[(mdf.Teff > 4400) & (mdf.Teff < 5400)]
        if flag_mcq14_comp:
            mdf1 = mdf1[(mdf1.Teff > 4400) & (mdf1.Teff < 5400)]

    # run analysis for average uncertainties
    result = mdf.groupby('KIC')['age'].agg([
        ('mean_age', 'mean'),
        ('median_age', 'median'),
        ('+1sigma', lambda x: x.mean() + x.std()),
        ('-1sigma', lambda x: x.mean() - x.std())
    ]).reset_index()

    median_age_ranges = [
        (800, 1200),
        (1800, 2200),
        (2800, 3200),
    ]
    mean_pms = []
    for median_age_range in median_age_ranges:
        mean_plus_sigma, mean_minus_sigma = get_sigma_range(
            result, median_age_range
        )
        mean_pms.append([mean_plus_sigma, mean_minus_sigma])

    if flag_mcq14_comp:
        result1 = mdf1.groupby('KIC')['age'].agg([
            ('mean_age', 'mean'),
            ('median_age', 'median'),
            ('+1sigma', lambda x: x.mean() + x.std()),
            ('-1sigma', lambda x: x.mean() - x.std())
        ]).reset_index()

        mean_pms1 = []
        for median_age_range in median_age_ranges:
            mean_plus_sigma, mean_minus_sigma = get_sigma_range(
                result, median_age_range
            )
            mean_pms1.append([mean_plus_sigma, mean_minus_sigma])

    # SAMPLES from the age posteriors
    plt.close("all")
    set_style('science')
    fig, ax = plt.subplots()

    bw = 200
    bins = np.arange(0, (MAXAGE+2*bw)/1e3, bw/1e3)

    ax.hist(mdf.age, bins=bins, color='lightgray', density=False, zorder=1,
            label='all')

    sel_gyro_ok = mdf.KIC.isin(skdf.KIC)
    if flag_mcq14_comp:
        sel_gyro_ok1 = mdf1.KIC.astype(str).isin(skdf1.KIC.astype(str))

    ax.hist(mdf[sel_gyro_ok].age, bins=bins, color='k',
            density=False, zorder=2, label='gyro applicable')

    ax.legend(loc='best', fontsize='small')

    xmin = 0
    xmax = MAXAGE
    ax.update({
        'xlabel': 'Age [Myr]',
        'ylabel': 'Count (10x over-rep)',
        'xlim': [xmin, xmax],
    })
    outpath = os.path.join(
        outdir,
        f'hist_samples_field_gyro_ages_{cache_id}_maxage{MAXAGE}{santosstr}.png'
    )
    savefig(fig, outpath, writepdf=1, dpi=400)

    #################################################
    # ok... now how about the subset that are good? #
    #################################################
    def get_poisson_uncertainties(df, bins):
        bin_counts, _ = np.histogram(df.age/1e3, bins=bins)
        # correction for 10x sample/overcount per star
        bin_counts = bin_counts / 10
        total_counts = len(df) / 10
        poisson_uncertainties = np.sqrt(bin_counts) / total_counts
        return poisson_uncertainties

    plt.close("all")
    set_style('science')
    factor = 0.9
    fig, axs = plt.subplots(ncols=3, figsize=(factor*5.5, factor*2.3),
                            constrained_layout=True)

    koi_df = get_koi_data('cumulative-KOI', grazing_is_ok=1)
    koi_df['kepid'] = koi_df['kepid'].astype(str)
    skoi_df = koi_df[koi_df['flag_is_ok_planetcand']]

    sel_planets = mdf.KIC.isin(skoi_df.kepid)

    N = int(len(mdf)/10)

    l0_0 = 'P$_{\mathrm{rot}}$ '+f'({N})'
    axs[0].hist(mdf.age/1e3, bins=bins, color='C0', histtype='step',
                weights=np.ones(len(mdf))/len(mdf), zorder=-5, label=l0_0,
                alpha=0.4, linewidth=0.5)

    N = int(len(mdf[sel_gyro_ok])/10)
    l0_1 = f'Gyro applicable ({N})'
    heights, bin_edges, _  = axs[0].hist(
        mdf[sel_gyro_ok].age/1e3, bins=bins, color='C0', histtype='step',
        weights=np.ones(len(mdf[sel_gyro_ok]))/len(mdf[sel_gyro_ok]), zorder=-3,
        alpha=1, label=l0_1
    )

    ynorm = heights[0]
    l2_1 = f'Kepler stars ({N})'
    if flag_mcq14_comp:
        l2_1 = 'Santos P$_{\mathrm{rot}}$' + f' ({N})'
    _  = axs[2].hist(
        mdf[sel_gyro_ok].age/1e3, bins=bins, color='C0', histtype='step',
        weights=np.ones(len(mdf[sel_gyro_ok]))/len(mdf[sel_gyro_ok]), zorder=-1,
        alpha=1, label=l2_1
    )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    poisson_uncertainties = get_poisson_uncertainties(mdf[sel_gyro_ok], bins)
    axs[0].errorbar(
        bin_centers, heights, yerr=poisson_uncertainties, marker='o',
        elinewidth=0.7, capsize=1, lw=0, mew=0.5, color='C0', markersize=0,
        zorder=-3, alpha=0.8
    )
    axs[2].errorbar(
        bin_centers, heights, yerr=poisson_uncertainties, marker='o',
        elinewidth=0.7, capsize=1, lw=0, mew=0.5, color='C0', markersize=0,
        zorder=-1, alpha=0.8
    )
    fit_line_and_print_results(bin_centers, heights, poisson_uncertainties)

    axs[0].errorbar(
        [1,2,3], [0.018,0.018,0.018], xerr=np.array(mean_pms).T/1e3, marker='o',
        elinewidth=0.7, capsize=1, lw=0, mew=0.5, color='C0', markersize=0,
        zorder=-3, alpha=1
    )
    axs[0].text(3, 0.0195, 'stat.\nuncert.', ha='center', va='bottom',
                fontsize='xx-small', zorder=5,
                transform=axs[0].transData,
                fontdict={'fontstyle':'normal'}, color='C0')


    ##########################################
    # calculate ratios of "middle" and "old" bins to young bin for all stars.
    n_yb = len(mdf[sel_gyro_ok][
        (mdf[sel_gyro_ok].age > 0) & (mdf[sel_gyro_ok].age <= 1000)
    ])/10
    n_mb = len(mdf[sel_gyro_ok][
        (mdf[sel_gyro_ok].age > 1000) & (mdf[sel_gyro_ok].age <= 2000)
    ])/10
    n_ob = len(mdf[sel_gyro_ok][
        (mdf[sel_gyro_ok].age > 2000) & (mdf[sel_gyro_ok].age <= 3000)
    ])/10
    if not preciseagesonly and not flag_mcq14_comp and not dropfraclongrot:
        ulkvp(f'{mcqstr}ratiombtoybstars', np.round(n_mb/n_yb, 1))
        ulkvp(f'{mcqstr}ratiombtoybstars', np.round(n_mb/n_yb, 1))
        ulkvp(f'{mcqstr}ratioobtoybstars', np.round(n_ob/n_yb, 1))

    n_youngg = len(mdf[sel_gyro_ok][
        (mdf[sel_gyro_ok].age > 0) & (mdf[sel_gyro_ok].age <= 300)
    ])/10
    n_oldd = len(mdf[sel_gyro_ok][
        (mdf[sel_gyro_ok].age > 2700) & (mdf[sel_gyro_ok].age <= 3000)
    ])/10
    ratiosfr = n_oldd/n_youngg
    if not preciseagesonly and not flag_mcq14_comp and not dropfraclongrot:
        ulkvp(f'{mcqstr}ratiosfr', f"{ratiosfr:.2f}")

    σ_oldd = n_oldd**0.5/n_oldd
    σ_youngg = n_youngg**0.5/n_youngg
    unc_ratio = np.sqrt(σ_oldd**2 + σ_youngg**2) * ratiosfr
    if not preciseagesonly and not flag_mcq14_comp and not dropfraclongrot:
        ulkvp(f'{mcqstr}uncratiosfr', f"{unc_ratio:.2f}")
    print(f'preciseagesonly: {preciseagesonly}, '
          f'flag_mcq14_comp: {flag_mcq14_comp}, '
          f'dropfraclongrot: {dropfraclongrot}.'
    )
    print(f'ratiosfr +/- uncratiosfr: {ratiosfr:.2f} +/- {unc_ratio:.2f}')

    ##########################################

    axs[0].legend(loc='best', fontsize='x-small')

    N = int(len(mdf[sel_planets])/10)
    if flag_mcq14_comp:
        N = int(len(mdf1)/10)

    l1_0 = 'P$_{\mathrm{rot}}$ '+f'({N})'
    if not flag_mcq14_comp:
        axs[1].hist(mdf[sel_planets].age/1e3, bins=bins, color='sienna',
                    histtype='step',
                    weights=np.ones(len(mdf[sel_planets]))/len(mdf[sel_planets]),
                    zorder=-5,
                    label=l1_0, alpha=0.4, linewidth=0.5)
    else:
        axs[1].hist(mdf1.age/1e3, bins=bins, color='darkorange',
                    histtype='step',
                    weights=np.ones(len(mdf1))/len(mdf1),
                    zorder=-5,
                    label=l1_0, alpha=0.4, linewidth=0.5)

    psel = sel_gyro_ok & sel_planets
    N = int(len(mdf[psel])/10)
    if flag_mcq14_comp:
        N = int(len(mdf1[sel_gyro_ok1])/10)

    l1_1 = f'Gyro applicable ' + f"({N})"
    if not flag_mcq14_comp:
        heights, _, _ = axs[1].hist(
            mdf[psel].age/1e3, bins=bins, histtype='step',
            weights=np.ones(len(mdf[psel]))/len(mdf[psel]), color='sienna', alpha=0.9,
            zorder=-3, label=l1_1
        )
    else:
        heights, _, _ = axs[1].hist(
            mdf1[sel_gyro_ok1].age/1e3, bins=bins, histtype='step',
            weights=np.ones(len(mdf1[sel_gyro_ok1]))/len(mdf1[sel_gyro_ok1]),
            color='darkorange', alpha=0.9, zorder=-3, label=l1_1
        )

    l2_2 = f'KOI hosts ({N})'
    if flag_mcq14_comp:
        l2_2 = 'McQuillan P$_{\mathrm{rot}}$' + f' ({N})'

    if not flag_mcq14_comp:
        heights, _, _ = axs[2].hist(
            mdf[psel].age/1e3, bins=bins, histtype='step',
            weights=np.ones(len(mdf[psel]))/len(mdf[psel]), color='sienna', alpha=0.9,
            zorder=-2, label=l2_2
        )
        poisson_uncertainties = get_poisson_uncertainties(mdf[psel], bins)
    else:
        heights, _, _ = axs[2].hist(
            mdf1[sel_gyro_ok1].age/1e3, bins=bins, histtype='step',
            weights=np.ones(len(mdf1[sel_gyro_ok1]))/len(mdf1[sel_gyro_ok1]),
            color='darkorange', alpha=0.9, zorder=-2, label=l2_2
        )
        poisson_uncertainties = get_poisson_uncertainties(mdf1[sel_gyro_ok1], bins)

    _c = 'sienna' if not flag_mcq14_comp else 'darkorange'
    axs[1].errorbar(
        bin_centers, heights, yerr=poisson_uncertainties, marker='o',
        elinewidth=0.7, capsize=1, lw=0, mew=0.5, color=_c, markersize=0,
        zorder=-3, alpha=0.8
    )
    axs[2].errorbar(
        bin_centers, heights, yerr=poisson_uncertainties, marker='o',
        elinewidth=0.7, capsize=1, lw=0, mew=0.5, color=_c, markersize=0,
        zorder=-2, alpha=0.8
    )
    print(f'preciseagesonly={preciseagesonly}')
    #planets
    #fit_line_and_print_results(bin_centers, heights, poisson_uncertainties)

    ##########################################
    # calculate ratios of "middle" and "old" bins to young bin for all selected
    # planets.
    n_yb = len(mdf[psel][
        (mdf[psel].age > 0) & (mdf[psel].age <= 1000)
    ])/10
    n_mb = len(mdf[psel][
        (mdf[psel].age > 1000) & (mdf[psel].age <= 2000)
    ])/10
    n_ob = len(mdf[psel][
        (mdf[psel].age > 2000) & (mdf[psel].age <= 3000)
    ])/10
    if not preciseagesonly and not flag_mcq14_comp and not dropfraclongrot:
        ulkvp(f'{mcqstr}ratiombtoybplanets', np.round(n_mb/n_yb, 1))
        ulkvp(f'{mcqstr}ratioobtoybplanets', np.round(n_ob/n_yb, 1))
    ##########################################

    # completeness gradient
    for ax in axs:
        add_gradient_patch(ax, 3.3, 4.1, 0, 0.1)

    xmin = 0
    xmax = MAXAGE-20 if MAXAGE < 4000 else MAXAGE
    if not preciseagesonly:
        teffstr = '$T_{\mathrm{eff}} \in [3800,6200]\,\mathrm{K}$'
        teffstr = '(3800—6200 K)'
    else:
        teffstr = '$T_{\mathrm{eff}} \in [4400,5400]\,\mathrm{K}$'
        teffstr = '(4400—5400 K)'

    for ix, ax in enumerate(axs):
        if ix == 0:
            ax.update({
            #'ylabel': '$N$/$N_{\mathrm{max}}$\n'+f'{teffstr}',
            'ylabel': 'Probability Density\n'+f'{teffstr}',
            'yticklabels': [0, 0.02, 0.04, 0.06, 0.08],
            })
        else:
            ax.update({
            'yticklabels': [],
            })
        if ix == 1:
            #ax.set_xlabel('Age from Rotation [Gigayears]')
            ax.set_xlabel('Age [Gigayears]')
        ax.update({
            #'xlabel': '$t_{\mathrm{gyro}}$ [Gyr]',
            'xlim': [xmin/1e3, (xmax)/1e3],
            'ylim': [0, 0.086],
            'yticks': [0, 0.02, 0.04, 0.06, 0.08],
        })
        if MAXAGE < 4000:
            ax.set_xticks([0, 1, 2, 3])
        else:
            ax.set_xticks([0, 1, 2, 3, 4])

    txt = 'Kepler stars'
    if flag_mcq14_comp:
        txt = 'Santos P$_{\mathrm{rot}}$'
    axs[0].text(.05, .95, txt, ha='left', va='top',
                fontsize='small', zorder=5, transform=axs[0].transAxes,
                fontdict={'fontstyle':'normal'}, color='C0')

    if not flag_mcq14_comp:
        txt = 'KOI hosts'
        axs[1].text(.05, .95, txt, ha='left', va='top',
                    fontsize='small', zorder=5, transform=axs[1].transAxes,
                    fontdict={'fontstyle':'normal'}, color='sienna')
    else:
        txt = 'McQuillan P$_{\mathrm{rot}}$'
        axs[1].text(.05, .95, txt, ha='left', va='top',
                    fontsize='small', zorder=5, transform=axs[1].transAxes,
                    fontdict={'fontstyle':'normal'}, color='darkorange')
    #axs[2].text(.05, .95, 'Comparison', ha='left', va='top',
    #            fontsize='large', zorder=5, transform=axs[2].transAxes,
    #            fontdict={'fontstyle':'normal'}, color='k')

    SHOW_MOR2019 = 1
    if SHOW_MOR2019:
        from scipy.interpolate import make_interp_spline
        csvpath = join(DATADIR, "literature", 'Mor_2019_sfr_vs_age.csv')
        data = pd.read_csv(csvpath)
        age_gyr = data['age_gyr'].values
        sfr = data['sfr_msun_per_gyr_per_pcsq'].values
        x_new = np.linspace(0, 10, 100)
        spl = make_interp_spline(age_gyr, sfr, k=3)
        y_new = spl(x_new)
        factor = 167 #spl(0.7)/ynorm
        l2_3 = 'Mor+19'
        print(42*'~')
        print(f'{l2_3}: {factor:.1f}')
        axs[2].plot(
            x_new, y_new/factor, c='k', lw=0.5, zorder=-10, ls='--',
            label=l2_3
        )

    SHOW_RUIZLARA2020 = 1
    if SHOW_RUIZLARA2020:
        from scipy.interpolate import make_interp_spline
        csvpath = join(DATADIR, "literature",
                       'RuizLara_2020_sfr_vs_age.csv')
        data = pd.read_csv(csvpath)
        age_gyr = data['age_gyr'].values
        sfr = data['sfr_au'].values
        x_new = np.linspace(0.05, 10, 100)
        spl = make_interp_spline(age_gyr, sfr, k=3)
        y_new = spl(x_new)
        factor = 18 # 0.9*spl(0.8)/ynorm
        l2_4 = 'Ruiz-Lara+20'
        print(42*'~')
        print(f'{l2_4}: {factor:.1f}')
        axs[2].plot(
            x_new, y_new/factor, c='k', lw=0.5, zorder=-10, ls=':',
            label=l2_4, alpha=0.7
        )

    SHOW_BERGER2020 = 0
    if SHOW_BERGER2020:
        # Berger+2020: Gaia-Kepler stellar properties catalog.
        fitspath = join(DATADIR, 'literature', 'Berger_2020_t2_secret.fits')
        hl = fits.open(fitspath)
        b20t2_df = Table(hl[1].data).to_pandas()
        #NOTE: might want to resample uncs???
        heights, _, _ = axs[2].hist(
            b20t2_df.Age, bins=bins, histtype='step',
            weights=np.ones(len(b20t2_df))*0.06/5531,
            color='gray', alpha=0.9,
            zorder=-2, label='Berger+20'
        )


    # AESTHETIC HAD WEIRD ISSUES...
    # Hide the right and top spines
    SHOW_RIGHTTOP = 1
    if not SHOW_RIGHTTOP:
        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            # Only show ticks on the left and bottom spines
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='C0', lw=0.5, alpha=0.4),
                    Line2D([0], [0], color='C0', lw=1, alpha=1.0 ) ]
    axs[0].legend(custom_lines, [l0_0, l0_1], fontsize='xx-small',
                  borderaxespad=0.9, borderpad=0.5, framealpha=0,
                  loc='lower right')
    _c = 'sienna' if not flag_mcq14_comp else 'darkorange'
    custom_lines = [Line2D([0], [0], color=_c, lw=0.5, alpha=0.4),
                    Line2D([0], [0], color=_c, lw=1, alpha=1.0 ) ]
    axs[1].legend(custom_lines, [l1_0, l1_1], fontsize='xx-small',
                  borderaxespad=0.9, borderpad=0.5, framealpha=0,
                  loc='lower right')
    custom_lines = [
        Line2D([0], [0], color='C0', lw=1, alpha=1.0),
        Line2D([0], [0], color=_c, lw=1, alpha=1.0),
        Line2D([0], [0], color='k', lw=0.5, alpha=1.0, ls='--'),
        Line2D([0], [0], color='k', lw=0.5, alpha=0.7, ls=':'),
    ]
    axs[2].legend(custom_lines, [l2_1, l2_2, l2_3, l2_4], fontsize='xx-small',
                  borderaxespad=0.9, borderpad=0.5, framealpha=0,
                  loc='lower right')

    s = ''
    if preciseagesonly:
        s += '_preciseagesonly'
    if mcqstr != '':
        mcqstr = "_"+mcqstr
    if dropfraclongrot:
        s += '_dropfraclongrot'

    outpath = os.path.join(
        outdir,
        f'comp_hist_samples_koi_gyro_ages_{cache_id}_maxage{MAXAGE}{santosstr}{s}{mcqstr}.png'
    )
    fig.tight_layout(h_pad=2)
    savefig(fig, outpath, writepdf=1, dpi=400)

    ###########################################
    ###########################################

    from scipy import stats
    sel_age = mdf.age < 3e9

    star_ages = nparr(
        mdf.loc[sel_gyro_ok & sel_age, 'age']
    )
    plhost_ages = nparr(
        mdf.loc[sel_gyro_ok & sel_planets & sel_age, 'age']
    )

    from gyrojo.stats import (
        bootstrap_ks_2samp, crossvalidate_ks_2samp
    )
    p_value_mean, p_value_std, p_value_median, p_value_ci = (
        crossvalidate_ks_2samp(star_ages, plhost_ages)
    )

    txt = (f'preciseagesonly={bool(preciseagesonly)}: '
           f'med(log10p)={np.log10(p_value_median):.1f}, '
           f'<log10p value>={np.log10(p_value_mean):.1f}, '
           f'log10(σ_pval)={np.log10(p_value_std):.1f}, '
           f'2.5-97.5% CI (log10p): '
           f'{np.log10(p_value_ci[0]):.1f} - {np.log10(p_value_ci[1]):.1f} 2-sample KS')
    print(42*'-')
    print(txt)
    print(42*'-')

    ##########################################
    # end the interesting plot               #
    ##########################################

    # begin "merged" version of this plot...
    plt.close("all")
    set_style('science')
    fig, ax = plt.subplots(figsize=(0.9*3, 0.9*3))

    N = int(len(mdf[sel_gyro_ok])/10)
    l0_1 = f'{N} Kepler targets with '+'P$_{\mathrm{rot}}$ & gyro applicable'
    heights, bin_edges, _  = ax.hist(
        mdf[sel_gyro_ok].age/1e3, bins=bins, color='C0', histtype='step',
        weights=np.ones(len(mdf[sel_gyro_ok]))/len(mdf[sel_gyro_ok]), zorder=2,
        alpha=0.8, label=l0_1
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    poisson_uncertainties = get_poisson_uncertainties(mdf[sel_gyro_ok], bins)
    ax.errorbar(
        bin_centers, heights, yerr=poisson_uncertainties, marker='o',
        elinewidth=0.7, capsize=1, lw=0, mew=0.5, color='C0', markersize=0,
        zorder=5, alpha=0.8
    )

    N = int(len(mdf[psel])/10)
    l1_1 = f'{N} KOIs with '+'P$_{\mathrm{rot}}$ & gyro applicable'
    heights, _, _ = ax.hist(
        mdf[psel].age/1e3, bins=bins, histtype='step',
        weights=np.ones(len(mdf[psel]))/len(mdf[psel]), color='C1', alpha=0.8,
        zorder=2, label=l1_1
    )
    poisson_uncertainties = get_poisson_uncertainties(mdf[psel], bins)
    ax.errorbar(
        bin_centers, heights, yerr=poisson_uncertainties, marker='o',
        elinewidth=0.7, capsize=1, lw=0, mew=0.5, color='C1', markersize=0,
        zorder=1, alpha=0.8
    )

    ax.legend(loc='best', fontsize='x-small')

    xmin = 0
    xmax = MAXAGE
    ax.update({
        #'xlabel': '$t_{\mathrm{gyro}}$ [Gyr]',
        'xlabel': 'Age from Rotation [Gigayears]',
        'ylabel': f'Fraction of Sample',
        'xlim': [xmin/1e3, (xmax+20)/1e3],
        'ylim': [0, 0.072],
        'yticks': [0, 0.02, 0.04, 0.06],
        'yticklabels': [0, 0.02, 0.04, 0.06],
    })
    if MAXAGE < 4000:
        ax.set_xticks([0, 1, 2, 3])
    if MAXAGE < 4000:
        ax.set_xticks([0, 1, 2, 3])

    #axs[1].set_yticklabels([])

    # AESTHETIC HAD WEIRD ISSUES...
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='C0', alpha=0.9, lw=1),
                    Line2D([0], [0], color='C1', alpha=0.9, lw=1) ]
    ax.legend(custom_lines, [l0_0, l0_1], fontsize='x-small',
              borderaxespad=2.0, borderpad=0.8, framealpha=0,
              loc='lower right')

    outpath = os.path.join(
        outdir,
        f'merged_hist_samples_koi_gyro_ages_{cache_id}_maxage{MAXAGE}{santosstr}.png'
    )
    fig.tight_layout()
    savefig(fig, outpath, writepdf=1, dpi=400)

    ##########


def plot_field_gyro_posteriors(outdir, cache_id, sampleid):

    from gyrointerp.helpers import get_summary_statistics

    from gyrointerp.paths import CACHEDIR
    csvdir = join(CACHEDIR, cache_id)
    writedir = join(CACHEDIR, "samples_"+cache_id)
    if not os.path.exists(writedir): os.mkdir(writedir)
    csvpaths = glob(join(csvdir, "*posterior.csv"))
    assert len(csvpaths) > 0
    N_post = len(csvpaths)
    print(f"Got {N_post} posteriors...")

    cachecsvpath = os.path.join(outdir, f"{cache_id}_gyro_ages.csv")
    if os.path.exists(cachecsvpath):
        df = pd.read_csv(cachecsvpath, dtype={'KIC':str})

    else:
        # plot all stars
        plt.close("all")
        set_style('clean')
        fig, ax = plt.subplots()

        kic_names = []
        summaries = []

        for ix, csvpath in enumerate(csvpaths):

            if ix % 100 == 0:
                print(f"{datetime.utcnow().isoformat()}: {ix}/{N_post}")

            kic_name = os.path.basename(csvpath).split("_")[0]
            kic_names.append(kic_name)

            df = pd.read_csv(csvpath)
            t_post = np.array(df.age_post)
            age_grid = np.array(df.age_grid)

            d = get_summary_statistics(age_grid, t_post)

            # draw 10 samples from each posterior, and write them...
            N = 10
            df = pd.DataFrame({'age':age_grid, 'p':t_post})
            try:
                outcsv = join(
                    writedir,
                    os.path.basename(csvpath).replace('posterior','posterior_samples')
                )
                sample_df = df.sample(n=N, replace=True, weights=df.p)
                sample_df.age.to_csv(outcsv, index=False)
            except ValueError:
                # some stars had adopted_teff>6200, adopted_teff<3800, or nan adopted_teff.
                pass
            if ix % 100 == 0:
                print(f"{datetime.utcnow().isoformat()}: wrote {outcsv}")

            summaries.append(d)

            zorder = ix
            ax.plot(age_grid, 1e3*t_post/np.trapz(t_post, age_grid), alpha=0.1,
                    lw=0.3, c='k', zorder=zorder)

        xmin = 0
        xmax = 6000
        ax.update({
            'xlabel': 'Age [Myr]',
            'ylabel': 'Probability ($10^{-3}\,$Myr$^{-1}$)',
            'xlim': [xmin, xmax],
            'ylim': [-0.5, 10.5]
        })
        outpath = os.path.join(outdir, f'posteriors_verification.png')
        savefig(fig, outpath, writepdf=1, dpi=400)

        df = pd.DataFrame(summaries, index=kic_names)
        df['KIC'] = kic_names
        df['KIC'] = df['KIC'].astype(str)
        df.to_csv(cachecsvpath, index=False)
        print(f"Wrote {cachecsvpath}")

    kdf = get_kicstar_data(sampleid)
    kdf['KIC'] = kdf['KIC'].astype(str)

    df = df.rename(
        {k:f'gyro_{k}' for k in df.columns if k != 'KIC'},
        axis='columns'
    )

    mdf = df.merge(kdf, how='inner', on='KIC')
    assert len(mdf) == len(df)

    csvpath = os.path.join(outdir, f"{cache_id}_gyro_ages_X_GDR3_S19_S21_B20.csv")
    mdf = mdf.sort_values(by='gyro_median')
    mdf.to_csv(csvpath, index=False)
    print(f"Wrote {csvpath}")

    cols = ['KIC', 'gyro_median', 'gyro_+1sigma', 'gyro_-1sigma',
            'gyro_+2sigma', 'gyro_-2sigma', "gyro_+1sigmapct", "gyro_-1sigmapct", "Prot",
            "adopted_Teff"]
    sel_2s = mdf['gyro_median'] + mdf['gyro_+2sigma'] < 1000

    print(mdf[sel_2s][cols])


def get_starnames(koi_ids):
    starnames = []
    for koi_id in koi_ids:
        if " " in koi_id:
            starname = koi_id.split(" ")[0]
        if "." in koi_id:
            starname = koi_id.split(".")[0]
        starnames.append(starname)
    return starnames

def find_duplicates(arr):
    seen = set()
    duplicates = set()

    for item in arr:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)

    result = np.array([item in duplicates for item in arr])
    return result

def plot_multis_vs_age(outdir, teffcondition='allteff'):

    # get data
    _df, d, st_ages = get_age_results(
        whichtype='gyro', grazing_is_ok=1
    )
    df = pd.DataFrame(d)

    sel = np.ones(len(df)).astype(bool)
    #sel = (df['rp']/df['rp_err1'] > 0.1) & (df['rp']/df['rp_err2'] > 0.1)
    # >33% radii
    #sel = (df['rp']/df['rp_err1'] > 3) & (df['rp']/df['rp_err2'] > 3)
    # >25% radii
    #sel = (df['rp']/df['rp_err1'] > 4) & (df['rp']/df['rp_err2'] > 4)
    # >20% radii
    #sel = (df['rp']/df['rp_err1'] > 5) & (df['rp']/df['rp_err2'] > 5)
    #AGE_MAX = 10**9.5 #(3.2 gyr)
    AGE_MAX = 3e9

    sel &= df['age'] < AGE_MAX

    if teffcondition == 'allteff':
        pass
    elif teffcondition == 'teffgt5000':
        sel &= df['adopted_Teff'] >= 5000
    elif teffcondition == 'tefflt5000':
        sel &= df['adopted_Teff'] < 5000

    df = df[sel]
    st_ages = st_ages[st_ages < AGE_MAX]

    df['age_pcterravg'] = np.nanmean(
        [df['age_pcterr1'], df['age_pcterr2']], axis=0
    )
    df['age_pcterrmax'] = np.max(
        [df['age_pcterr1'], df['age_pcterr2']], axis=0
    )
    df['age_errmean'] = np.nanmean(
        [df['age_err1'], df['age_err2']], axis=0
    )
    print(df.describe())

    df['starname'] = get_starnames(np.array(df['pl_name']))
    df['ismulti'] = find_duplicates(np.array(df['starname']))

    multidf = df[df.ismulti]
    multidf = multidf.sort_values(by=['age','pl_name'])

    N = len(np.unique(multidf.starname))

    plt.close("all")
    set_style("clean")
    fig, ax = plt.subplots(figsize=(3.3, 0.1*N))

    for ix, starname in enumerate(list(multidf.starname.drop_duplicates())):

        sel = (multidf.starname == starname)
        sdf = multidf[sel]

        min_period = np.min(sdf.period)
        norm_periods = nparr(sdf.period) / min_period

        sizes = nparr(sdf.rp)

        age = nparr(sdf.age)[0]
        age_errmean = nparr(sdf.age_errmean)[0]

        cmap = mpl.cm.viridis
        ax.scatter(
            norm_periods, -ix*np.ones_like(norm_periods),
            c=age*np.ones_like(norm_periods), cmap=cmap, linewidths=0.1,
            s=sizes
        )

        props = dict(boxstyle='square', facecolor='white', alpha=0.95, pad=0.15,
                     linewidth=0)
        # plot the observed ticks. x coords are axes, y coords are data
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)

        txt = (
            f'{starname.replace("ler","")}, '+'$P_{\mathrm{i}}$='+f'{min_period:.1f} d, '+
            '$t_\mathrm{g}$=' + f"{age/1e9:.2f} "+"$\pm$"+f"{age_errmean/1e9:.2f} Gy"
        )
        ax.text(
            0.99, -ix, txt, transform=trans, bbox=props, ha='right',
            va='center', fontsize='xx-small'
        )

        #bounds = np.arange(-0.5, 3.5, 1)
        #ticks = (np.arange(-1,3)+1)
        #norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    ymin, ymax = ax.get_ylim()
    vlines = [3/2, 4/2, 3]
    for vline in vlines:
        ax.vlines(
            vline, ymin, ymax, colors='darkgray', alpha=1,
            linestyles=':', zorder=-2, linewidths=0.6
        )
    ax.set_ylim((ymin, ymax))

    ax.set_xlabel("P/P_inner")
    ax.set_ylabel("")
    #ax.set_xlim([ 6300, 3700 ])
    #ax.set_xlim([0.5,10.5])
    ax.set_xscale('log')
    ax.set_yticklabels([])

    outpath = os.path.join(outdir, f'multis_age_sorted.png')
    savefig(fig, outpath)


def plot_gyroage_vs_teff(outdir, yscale='linear', showerrs=0, showplanets=0):

    # get data

    # stars
    sampleid = 'Santos19_Santos21_dquality'
    kicdf = get_kicstar_data(sampleid)
    kicdf = kicdf[kicdf['flag_is_gyro_applicable']]

    # planets
    koidf, _, _ = get_age_results(whichtype='gyro_li', COMPARE_AGE_UNCS=0,
                                  grazing_is_ok=1, drop_highruwe=0)

    # make plot
    plt.close('all')
    set_style('clean')

    fig, ax = plt.subplots(figsize=(4,3))

    yerr = nparr(
        [nparr(kicdf['gyro_-1sigma']), nparr(kicdf['gyro_+1sigma'])]
    ).reshape((2, len(kicdf)))
    yval = nparr(kicdf['gyro_median'])
    xval = nparr(kicdf['adopted_Teff'])
    meanpct = np.nanmean(nparr(
        [nparr(kicdf['gyro_-1sigmapct']), nparr(kicdf['gyro_+1sigmapct'])]
    ).reshape((2, len(kicdf))), axis=0)
    if showerrs:
        if not showplanets:
            ax.errorbar(
                xval, yval, yerr=yerr,
                marker='o', elinewidth=0.05, capsize=0, lw=0, mew=0.5, color='k',
                markersize=0, zorder=5, alpha=0.5
            )
    else:
        if not showplanets:
            ax.errorbar(
                xval, yval, yerr=yerr,
                marker='o', elinewidth=0.0, capsize=0, lw=0, mew=0., color='k',
                markersize=0.3, zorder=5, alpha=1
            )
    if showplanets:
        yerr = nparr(
            [nparr(koidf['gyro_-1sigma']), nparr(koidf['gyro_+1sigma'])]
        ).reshape((2, len(koidf)))
        yval = nparr(koidf['gyro_median'])
        xval = nparr(koidf['adopted_Teff'])
        if showerrs:
            ax.errorbar(
                xval, yval, yerr=yerr,
                marker='o', elinewidth=0.25, capsize=0, lw=0, mew=0.5, color='k',
                markersize=0, zorder=5, alpha=1
            )
        else:
            ax.errorbar(
                xval, yval, yerr=yerr,
                marker='o', elinewidth=0.0, capsize=0, lw=0, mew=0.5, color='k',
                markersize=2, zorder=5, alpha=1
            )

    ax.update({
        'xlabel': 'Effective Temperature [K]',
        'ylabel': r't$_{\rm gyro}$ [Myr]',
        'yscale': yscale,
        'xlim': ax.get_xlim()[::-1]
    })


    # set naming options
    if showerrs:
        s = 'errs'
    else:
        s = 'medvals'
    if showplanets:
        s += '_showplanets'

    s += f'_{yscale}'

    bn = 'gyroage_vs_teff'
    outpath = join(outdir, f'{bn}_{s}.png')
    savefig(fig, outpath, dpi=400)


def get_vtang(df):
    # Tricky unit conversions, but this works, where v_Tang is km/s, and rest
    # of quantities are mas.
    # NOTE: these two are ~equivalent.
    df['v_T'] = (
        4.74 *
        (1 / df.dr3_parallax)
        *
        np.sqrt( df.dr3_pmra**2 + df.dr3_pmdec**2 )
    )

    #  v_pmra = propermotion_to_kms(df.dr3_pmra, df.dr3_parallax)
    #  v_pmdec = propermotion_to_kms(df.dr3_pmdec, df.dr3_parallax)
    #  df['v_T_2'] = np.sqrt( v_pmdec**2 + v_pmra**2 )

    return df


def plot_st_params(outdir, xkey='dr3_bp_rp', ykey='M_G', vtangcut=None):

    # get data

    # stars (w/ Prot)
    sampleid = 'Santos19_Santos21_dquality'
    kicdf = get_kicstar_data(sampleid)

    # planets (for gyro)
    koidf, _, _ = get_age_results(whichtype='gyro_li', COMPARE_AGE_UNCS=0,
                                  grazing_is_ok=1, drop_highruwe=0)

    # get KIC target stars
    from gyrojo.getters import (
        get_cleaned_gaiadr3_X_kepler_supplemented_dataframe
    )
    cgk_df = get_cleaned_gaiadr3_X_kepler_supplemented_dataframe()

    #  # if you wanted all KOIs, including FPs
    #  koi_df = get_koi_data('cumulative-KOI', grazing_is_ok=1)
    #  koi_df['kepid'] = koi_df['kepid'].astype(str)
    #  # REQUIRE "flag_is_ok_planetcand"
    #  skoi_df = koi_df[koi_df['flag_is_ok_planetcand']]

    # make plot
    plt.close('all')
    set_style('clean')

    fig, ax = plt.subplots(figsize=(2.5,2.7))

    # Stars with Santos+ rotation period reported (...and good Gaia matches)
    sel = select_by_quality_bits(kicdf, [4, 5], [0, 0])

    dfs = [
       #cgk_df,
        kicdf[sel],
        kicdf[kicdf['flag_is_gyro_applicable']],
        koidf
    ]

    colors = [
        #'lightgray',
        'silver',
        'dimgray',
        'yellow'
    ]
    zorders = [
        #-1,
        0,
        1,
        2
    ]
    sizes = [
        #0.1,
        0.3,
        0.8,
        1.5
    ]
    labels = [
        #'KIC',
        'Has $P_\mathrm{rot}$',
        '...& gyro applicable',
        '...& KOI host',
    ]
    rasterized = [
        #True,
        True,
        True,
        False
    ]

    for _i, (df, c, z, l, s, r) in enumerate(zip(
        dfs, colors, zorders, labels, sizes, rasterized
    )):

        if isinstance(vtangcut, str):
            df = get_vtang(df)
            if vtangcut == 'thindisk':
                df = df[(df["v_T"] < 40)]
            elif vtangcut == 'thickdisk':
                df = df[(df["v_T"] > 60) & (df["v_T"] < 150)]
            elif vtangcut == 'halo':
                df = df[(df["v_T"] > 200)]

        yval = nparr(df[ykey])
        xval = nparr(df[xkey])
        kicids = nparr(df['KIC'])
        if _i != len(dfs) - 1:
            ax.scatter(
                xval, yval,
                marker='o', c=c, zorder=z, s=s, linewidths=0,
                label=l+f" ($N$={len(np.unique(kicids))})", rasterized=r
            )
        else:
            ax.scatter(
                xval, yval,
                marker='o', c=c, zorder=z, s=s, linewidths=0.1,
                label=l+f" ($N$={len(np.unique(kicids))})", rasterized=r,
                edgecolors='k'
            )

    if ykey == 'M_G' and xkey == 'dr3_bp_rp':
        csvpath = join(DATADIR, "interim", f"MG_bprp_locus_coeffs_poly.csv")
        coeffs = pd.read_csv(csvpath).values.flatten()
        _bprp = np.linspace(0.5, 3, 1000)
        poly_vals = np.polyval(coeffs, _bprp)
        #ax.plot(bprp, poly_vals, zorder=5, c='C1', lw=0.5)
        show_bounds = 0
        if show_bounds:
            ax.plot(_bprp, poly_vals-1, zorder=5, c='C1', lw=0.5)
            ax.plot(_bprp, poly_vals+1, zorder=5, c='C1', lw=0.5)


    if ykey == 'adopted_logg' and xkey == 'adopted_Teff':
        xerr = dfs[1].adopted_Teff_err.mean()
        yerr = dfs[1].adopted_logg_err.mean()
        ax.errorbar(
            3500, 4.2, xerr=xerr, yerr=yerr,
            marker='o', elinewidth=0.8, capsize=1.2, lw=0, mew=0.5,
            color='dimgray', markersize=0, zorder=5, alpha=1
        )
        bbox = dict(facecolor='white', alpha=1, pad=0, edgecolor='white')
        ax.text(3500, 4.27, 'mean\nuncert.', ha='center', va='top',
                fontsize='x-small', bbox=bbox, zorder=49, color='k')

        from gyrojo.locus_definer import constrained_polynomial_function
        selfn = 'manual'
        csvpath = join(DATADIR, "interim", f"logg_teff_locus_coeffs_{selfn}.csv")
        coeffs = pd.read_csv(csvpath).values.flatten()
        _teff = np.linspace(3801,6199,1000)
        if selfn in ['simple', 'complex']:
            ax.plot(_teff, constrained_polynomial_function(_teff, coeffs),
                    zorder=5, c='C1', lw=0.5)
        elif selfn == 'manual':
            csvpath = join(DATADIR, "interim", f"logg_teff_locus_coeffs_simple.csv")
            _coeffs = pd.read_csv(csvpath).values.flatten()
            _y = constrained_polynomial_function(_teff, _coeffs, selfn='simple')
            p = constrained_polynomial_function(_teff, coeffs, selfn=selfn)
            # 'top' (logg too low)
            _y0 = p - 0.12
            y0 = np.maximum(_y, _y0)
            # if you wanted to cut on 'bottom' (logg too high)
            y1 = p + 0.1
            ax.plot(_teff, y0, zorder=5, c='C1', lw=0.5)
           # ax.plot(_teff, y1, zorder=5, c='C1', lw=0.5)
            ax.plot(_teff, p, zorder=5, c='C1', lw=0.2, alpha=0.3)

        # overplot isochrones: MIST v1.2 with rotation
        #colors = ['cyan', 'hotpink', 'lime', 'magenta']
        #colors = plt.cm.Spectral(np.linspace(0, 1, 3))
        #colors = ['cyan', 'lime', 'magenta']
        colors = ['k','k','k']
        linestyles = ['--', '-', '-.']
        csvnames = [
            "MIST_iso_664e4474d6bb1_1gyr.iso.cmd",
            "MIST_iso_662b04a781d06_3gyr.iso.cmd",
            #"MIST_iso_662b02fe7d746_4gyr.iso.cmd",
            #"MIST_iso_662b0653ce559_5gyr.iso.cmd",
            #"MIST_iso_662b0684a43eb_6gyr.iso.cmd",
            "MIST_iso_664e448e18d47_10gyr.iso.cmd"
        ]
        csvpaths = [join(DATADIR, "literature", n) for n in csvnames]
        for ix, (csvpath, c, ls) in enumerate(zip(csvpaths, colors, linestyles)):
            mistdf = pd.read_csv(csvpath, delim_whitespace=True, comment='#')
            mist_teff = 10**mistdf.log_Teff
            mist_logg = mistdf.log_g
            sel = (mist_teff > 4000)
            ax.plot(mist_teff[sel], mist_logg[sel], zorder=5, c=c, lw=0.2,
                    alpha=1, ls=ls)
            bbox = dict(facecolor='white', alpha=1, pad=0, edgecolor='white')
            if ix == 1:
                ax.text(7500, 3.8, '1 Gyr', ha='left', va='center',
                        fontsize='x-small', bbox=bbox, zorder=49, color=c)
            if ix == 1:
                ax.text(6250, 3.8, '3 Gyr', ha='center', va='center',
                        fontsize='x-small', bbox=bbox, zorder=49, color=c)
            elif ix == 2:
                ax.text(5400, 3.8, '10 Gyr', ha='left', va='center',
                        fontsize='x-small', bbox=bbox, zorder=49, color=c)


        #print(constrained_polynomial_function(np.array([5070]), coeffs))

    if xkey == 'dr3_bp_rp':
        xlabel = '$G_\mathrm{BP}-G_{\mathrm{RP}}$ [mag]'
        xlim = [0.1, 3.05]
    elif xkey == 'adopted_Teff':
        xlabel = '$T_\mathrm{eff}$ [K]'
        xlim = [7600, 2500]

    if ykey == 'M_G':
        ylabel = '$M_\mathrm{G}$ [mag]'
        ylim = [12.5, -1]
    elif ykey == 'adopted_logg':
        ylabel = '$\log g$ [dex]'
        ylim = [5.4, 3.4]
    elif ykey == 'dr3_phot_g_mean_mag':
        ylabel = '$G$ [mag]'
        ylim = [21, 5]

    ax.update({
        'xlabel': xlabel,
        'ylabel': ylabel,
        'ylim': ylim,
        'xlim': xlim
    })
    if ykey == 'M_G':
        ax.set_yticks([12, 8, 4, 0])
    elif ykey == 'adopted_logg':
        ax.set_yticks([5, 4.5, 4, 3.5])

    ax.legend(
        loc='lower left', fontsize='x-small',
        markerscale=3,
        framealpha=0,
        #handletextpad=0.3, framealpha=0, borderaxespad=0, borderpad=0,
        #handlelength=1.6#, bbox_to_anchor=(0.97, 0.97)
        handletextpad=0.1, borderaxespad=0.5, borderpad=0.5
    )

    #ax.grid(linestyle=':', linewidth=0.5, color='gray', alpha=0.7, zorder=-1)

    # set naming options
    s = f'_{ykey}_vs_{xkey}'
    if isinstance(vtangcut, str):
        s += f'_{vtangcut}'

    outpath = join(outdir, f'st_params{s}.png')
    savefig(fig, outpath, dpi=400)



def get_Prot_model(age, Teff_obs):

    # Get model Prots
    age = int(age)

    Prot_mu = slow_sequence(Teff_obs, age, verbose=False)

    y_grid = np.linspace(-14, 6, 1000)
    Prot_resid = slow_sequence_residual(age, y_grid=y_grid, teff_grid=Teff_obs)

    Prot_mod = []

    for ix, _ in enumerate(Teff_obs):

        dProt = np.random.choice(
            y_grid, size=1, replace=True,
            p=Prot_resid[:,ix]/np.sum(Prot_resid[:,ix])
        )

        Prot_mod.append(Prot_mu[ix] + dProt)

    Prot_mod = np.array(Prot_mod)

    return Prot_mod





def plot_gyromodeldispersion(outdir):
    # similar to _plot_slow_sequence_residual from gyrointerp

    from matplotlib.colors import LogNorm, Normalize
    from matplotlib import cm

    np.random.seed(42)

    ymin, ymax = 0, 30
    teffmin, teffmax = 3800, 6200
    N = int(1e4)
    #N = int(5e3)
    Teffs = np.linspace(teffmin, teffmax, N)
    ages = [100, 500, 1000, 2000]

    Nteffbins, Nprotbins = (
        int(1.5*int((teffmax-teffmin)/20)), int(1.5*int((ymax-ymin)/0.2))
    )
    print(f"Npoints: {N}, Nteff, Nprot: {Nteffbins}, {Nprotbins}, product: {Nteffbins*Nprotbins}")

    Prots = []
    Protmodels = []
    for age in ages:

        _Prot = slow_sequence(Teffs, age)
        Prots.append(_Prot)

        Protmodel = get_Prot_model(age, Teffs)
        Protmodels.append(Protmodel.flatten())

    ####

    set_style("clean")
    fig, ax = plt.subplots(figsize=(3,3))

    agelabels = ['100 Myr', '500 Myr', '1 Gyr', '2 Gyr']
    yvals = [8.5, 12.5, 16.5, 19.5, 28]
    linestyles = ['solid', 'dotted', 'dashdot', 'dashed', 'solid']
    colormaps = [cm.Blues, cm.Oranges, cm.Greens, cm.Purples, cm.Greys]
    colors = ['C0', 'C1', 'C2', 'purple', 'darkgray']
    zorders = [2,3,4,5,6][::-1]

    for age, ls, yval, al, Prot, Protmodel, _c, _cm, zo in zip(
        ages, linestyles, yvals, agelabels, Prots, Protmodels,
        colors,
        colormaps,
        zorders
    ):

        # plot mean lines
        color = 'k'
        ax.plot(
            Teffs[5:-5], Prot[5:-5], color=color, linewidth=1,
            zorder=10, alpha=0.9, ls=ls
        )

        SHOW_POINTS = 1
        SHOW_HIST2D = 0

        if SHOW_POINTS:
            ax.scatter(
                Teffs, Protmodel, linewidths=0,
                marker='o', color=_c, zorder=zo, s=0.4, rasterized=True

            )

        if SHOW_HIST2D:
            H, _, _ = np.histogram2d(
                Teffs, Protmodel, bins=[Nteffbins, Nprotbins],
                range=[[teffmin, teffmax], [ymin, ymax]]
            )

            #alpha_mask = 1/(H)
            #alpha_mask[alpha_mask > 1] = 1
            #alpha_mask[alpha_mask < 0.5] = 0.5

            norm = LogNorm()
            h, xedges, yedges, img = ax.hist2d(
                Teffs, Protmodel, bins=[Nteffbins, Nprotbins],
                range=[[teffmin, teffmax], [ymin, ymax]],
                #density=True,
                cmin=1,
                cmap=_cm,
                norm=norm,
                alpha=0.8,
                #alpha=alpha_mask,
                zorder=zo
            )
            h[pd.isnull(h)] = 0
            assert np.all(h==H)

        # annotate means
        bbox = dict(facecolor='white', alpha=1, pad=0, edgecolor='white')
        ax.text(3680, yval, al, ha='right', va='center', fontsize='x-small',
                bbox=bbox, zorder=49, color=_c)

    txt = r"Model: ${\tt gyro-interp}$"
    ax.text(0.97, 0.97, txt, transform=ax.transAxes,
            ha='right',va='top', color='k')

    ax.set_xlabel("Effective Temperature [K]")
    ax.set_ylabel("Rotation Period [days]")
    ax.set_xlim([ 6300, 3700 ])
    ax.set_ylim([ -1, 23 ])

    s = ''

    outpath = os.path.join(outdir, f'gyromodeldispersion{s}.png')
    savefig(fig, outpath)


def plot_liagefloor_vs_teff(outdir):

    # get data
    sys.path.append('/Users/luke/Dropbox/proj/eagles')

    # import the EWLi prediction model from the main EAGLES code
    from eagles import AT2EWm
    from eagles import eAT2EWm
    from eagles import get_li_age

    # set up a an equally spaced set of log temperatures between 3000 and 6500 K
    Teff_model = np.linspace(np.log10(3800), np.log10(6200), int(1e2))

    twosig_age_lowerlimits = []
    for ix, _Teff in enumerate(Teff_model):
        if ix % 10 == 0:
            print(f"{ix}/{len(Teff_model)}")
        LiEW = np.array([-20])   #if one were to adopt LiEW > 20mA required
        eLiEW = np.array([20])
        prior = 1 # linear age prior
        lagesmin = 6.0
        lagesmax = 10.1
        lApkmin = np.log10(5)+6
        nAge = 820
        lAges, llike, lprob, p, chisq = (
            get_li_age(LiEW, eLiEW, np.array([10**_Teff]), lagesmax=lagesmax,
                       lagesmin=lagesmin, lApkmin=lApkmin, z=0.0, nAge=nAge,
                       prior=prior)
        )
        twosig_age_lowerlimit = p[4]
        twosig_age_lowerlimits.append(twosig_age_lowerlimit)

    twosig_age_lowerlimits = nparr(twosig_age_lowerlimits)

    # make plot
    plt.close('all')
    set_style('clean')

    fig, ax = plt.subplots(figsize=(4,3))

    ax.plot(
        10**Teff_model, (10**twosig_age_lowerlimits)/1e9, c='k', lw=1
    )
    ax.update({
        'xlabel': 'Effective Temperature [K]',
        'ylabel': '2σ $t_\mathrm{Li}$ lower limit [Gyr]',
        'yscale': 'linear',
        'title': f'EW={LiEW[0]}$\pm${eLiEW[0]}mA implies...',
        'xlim': ax.get_xlim()[::-1]
    })

    # set naming options
    s = ''

    outpath = os.path.join(outdir, f'liagefloor_vs_teff.png')
    savefig(fig, outpath, dpi=400)


def plot_perioddiff_vs_period(outdir, xkey='Prot', ykey=None, ylim=None,
                              dx=0.25, dy=0.125):

    from gyrojo.prot_uncertainties import get_empirical_prot_uncertainties

    assert isinstance(ykey, str)

    xlabeldict = {
        'Prot': 'Santos $P_\mathrm{rot}$ [days]'
    }
    ylabeldict = {
        'Prot': 'Santos $P_\mathrm{rot}$',
        'm14_Prot': "M14 $P_\mathrm{rot}$",
        'r23_ProtGPS': 'R23$_\mathrm{GPS}$ $P_\mathrm{rot}$',
        'r23_ProtFin': 'R23$_\mathrm{Fin}$ $P_\mathrm{rot}$',
    }

    df = get_prot_metacatalog()

    # make plot
    plt.close('all')
    set_style('clean')

    fig, ax = plt.subplots(figsize=(3.5,2.5))

    xval = df[xkey]
    yval = df[xkey] - df[ykey]

    SHOW_POINTS = 0
    if SHOW_POINTS:
        ax.scatter(
            xval, yval,
            c='lightgray', s=0.25, linewidths=0, zorder=1
        )
    else:
        # Create the 2D histogram
        hist, xedges, yedges, im = plt.hist2d(
            xval, yval,
            bins=[np.arange(-1, 51, dx),
                  np.arange(-12, 12, dy)],
            zorder=-1
        )

        # Create a custom colormap with white color for zero values
        cmap = plt.cm.YlGnBu
        cmaplist = [cmap(i) for i in list(range(cmap.N))[:-100]]
        cmaplist[0] = (1.0, 1.0, 1.0, 1.0)  # Set the color for zero values to white
        cmap = mcolors.LinearSegmentedColormap.from_list('Custom YlGnBu', cmaplist, cmap.N)

        ## Apply log scaling to the colorbar
        norm = mcolors.LogNorm(vmin=0.9, vmax=np.max(hist))
        #norm = mcolors.Normalize(vmin=1, vmax=7)
        im.set_norm(norm)

        # Update the colormap of the plot
        im.set_cmap(cmap)

        show_colorbar = 1
        if show_colorbar:
            axins1 = inset_axes(ax, width="20%", height="2%", loc='lower right',
                                borderpad=2.5)

            cb = fig.colorbar(im, cax=axins1, orientation="horizontal",
                              norm=norm)
            #cb.set_ticks([1,4,7,10])
            #cb.set_ticks([1,10])
            cb.ax.tick_params(labelsize='small')
            cb.ax.tick_params(size=1, which='both') # shrink the ticks
            #cb.ax.yaxis.set_ticks_position('left')
            cb.ax.xaxis.set_label_position('top')
            cb.set_label(
                "$N_\mathrm{stars}$", fontsize='small', weight='normal',
                bbox={'facecolor':'white', 'edgecolor':'none', 'pad': 2}
            )


    pcts, alphas, lss = [1, 5], [0.8, 0.4], ['-.', ':']
    if 'r23' in ykey:
        pcts, alphas = [20, 40], [0.8, 0.4]
    for _ix, (pct, alpha, ls) in enumerate(zip(pcts, alphas, lss)):
        ax.plot(
            [0, 50], (pct/100)*np.array([0,50]), lw=0.5,
            label="$\Delta P/P=$"f"{pct}%", color=f'C{_ix}',
            alpha=alpha, ls=ls
        )
        ax.plot(
            [0, 50], -(pct/100)*np.array([0,50]), lw=0.5,
            color=f'C{_ix}', alpha=alpha, ls=ls
        )

    _prot = np.linspace(0.1, 50, 1000)
    prot_err = 2**(0.5) * get_empirical_prot_uncertainties(_prot)
    if 'r23' not in ykey:
        ax.plot(_prot, prot_err, lw=0.5, color='darkgray',
                label='Empirical $\sigma_{\mathrm{P}}/P$', zorder=99)
        ax.plot(_prot, -prot_err, lw=0.5, color='darkgray', zorder=99)

    # zero line
    ax.plot([-100,100], [0,0], ls='-', c='k', lw=0.5, alpha=0.5, zorder=0)

    # Calculate the absolute difference and add it as a new column
    df[f'diff_{ykey}'] = df[xkey] - df[ykey]

    # Bin the data and calculate median and ±1 sigma values
    bins = np.arange(1, 51, 1)
    binned_data = pd.cut(df[xkey], bins)
    grouped_data = df.groupby(binned_data)
    y_medians = grouped_data[f'diff_{ykey}'].median()
    y_q16 = grouped_data[f'diff_{ykey}'].quantile(0.16)
    y_q84 = grouped_data[f'diff_{ykey}'].quantile(0.84)

    # Overplot the median and ±1 sigma values
    x_mids = (bins[:-1] + bins[1:]) / 2
    x_err = np.ones_like(x_mids) / 2
    y_err = np.vstack([y_medians - y_q16, y_q84 - y_medians])
    ax.errorbar(x_mids, y_medians, xerr=x_err, yerr=y_err, fmt='o',
                color='k', elinewidth=0.5, capsize=0, lw=0,
                mew=0.5, markersize=0, zorder=9999)

    ax.set_xlabel(xlabeldict[xkey])
    ax.set_ylabel(f"{ylabeldict[xkey]} - {ylabeldict[ykey]} [days]")
    #ax.set_ylim([-2.2, 2.2])

    ax.legend(
        loc='upper left', fontsize='x-small',
        #markerscale=3,
        framealpha=0,
        #handletextpad=0.3, framealpha=0, borderaxespad=0, borderpad=0,
        #handlelength=1.6#, bbox_to_anchor=(0.97, 0.97)
        handletextpad=0.1, borderaxespad=1.5, borderpad=0.5
    )

    if not isinstance(ylim, list):
        ax.set_ylim([-12, 12])
        ax.set_yticks([-10,-5,0,5,10])
        ax.set_yticklabels([-10,-5,0,5,10])
    else:
        ax.set_ylim(ylim)
        ax.set_yticks([-4,-2,0,2,4])
        ax.set_yticklabels([-4,-2,0,2,4])
    ax.set_xlim([-2,52])

    outdir = join(RESULTSDIR, "perioddiff_vs_period")
    if not os.path.exists(outdir): os.mkdir(outdir)

    outpath = join(outdir, f'perioddiff_vs_period_diff{xkey}-{ykey}_vs_{xkey}.png')
    savefig(fig, outpath)
