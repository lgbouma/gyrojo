"""
Catch-all file for plotting scripts.  Contents:

    plot_koi_mean_prot_teff
    plot_star_Prot_Teff
    plot_reinhold_2015

    plot_li_vs_teff

    plot_gyroage_vs_teff
    plot_st_params

    plot_koi_gyro_posteriors
    plot_li_gyro_posteriors
    plot_field_gyro_posteriors

    plot_hist_field_gyro_ages

    plot_rp_vs_age
    plot_rp_vs_porb_binage
    plot_rp_ks_test

    plot_age_comparison

    plot_multis_vs_age

    plot_sub_praesepe_selection_cut

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
from numpy import array as nparr

from astropy.table import Table
from astropy.io import fits

from gyrojo.paths import DATADIR, RESULTSDIR, LOCALDIR, CACHEDIR
from gyrojo.getters import (
    get_gyro_data, get_li_data, get_age_results,
    get_kicstar_data, get_koi_data
)
from gyrojo.papertools import update_latex_key_value_pair as ulkvp

from gyrointerp.models import (
    reference_cluster_slow_sequence
)

from aesthetic.plot import set_style, savefig

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


def get_planet_class_labels(df, OFFSET=0):
    # given a dataframe with keys "rp" and "period", return a dataframe with a
    # "pl_class" key 

    df['pl_class'] = ''

    sel = df['rp'] <= 1
    df.loc[sel, 'pl_class'] = 'Earths'

    sel = df['rp'] >= 6
    df.loc[sel, 'pl_class'] = 'Sub-Saturns'

    sel = df['rp'] >= 10
    df.loc[sel, 'pl_class'] = 'Jupiters'

    # van eylen+2018
    m = -0.09
    a = 0.37
    fn_Rmod = lambda log10Pmod: 10**(m*log10Pmod + a)

    R_mod = fn_Rmod(np.log10(df['period'])) + OFFSET

    sel = (df['rp'] < 6) & (df['rp'] > R_mod)
    df.loc[sel, 'pl_class'] = 'Mini-Neptunes'

    sel = (df['rp'] > 1) & (df['rp'] <= R_mod)
    df.loc[sel, 'pl_class'] = 'Super-Earths'

    return df


############
# plotters #
############
def plot_koi_mean_prot_teff(outdir, sampleid='koi_X_S19S21dquality', drop_grazing=1):
    # For KOIs

    df = get_gyro_data(sampleid, drop_grazing=drop_grazing)

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
    ages = ['120 My', '670 My', '1 Gy', '2.6 Gy', '4 Gy']
    yvals = [9.8,14.8,16.7,20,28]

    _Teff = np.linspace(3800, 6200, int(1e3))
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot', 'solid']
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
            _Teff, _Prot, color=color, linewidth=1, zorder=10, alpha=0.4, ls=ls
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
    if drop_grazing:
        s += "dropgrazing"
    else:
        s += "keepgrazing"
    outpath = os.path.join(outdir, f'koi_mean_prot_teff_{sampleid}_{s}.png')
    savefig(fig, outpath)


def plot_star_Prot_Teff(outdir, sampleid):
    # For KIC / all Santos stars

    assert sampleid in [
        'Santos19_Santos21_all', 'Santos19_Santos21_logg',
        'Santos19_Santos21_clean0', 'teff_age_prot_seed42_nstar20000',
        'Santos19_Santos21_dquality'
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

    Teffs = nparr(df.adopted_Teff)
    Teff_errs = nparr(df.adopted_Teff_err)
    Prots = np.round(nparr(df.Prot), 4)
    if sampleid == 'Santos19_Santos21_all':
        # as reported; probably underestimated?
        Prot_errs = nparr(df.E_Prot)
    elif sampleid == 'Santos19_Santos21_clean0':
        # empirical uncs
        Prot_errs = nparr(df.Prot_err)
    elif sampleid == 'teff_age_prot_seed42_nstar20000':
        Prot_errs = nparr(df.Prot_err)

    set_style("clean")
    fig, ax = plt.subplots(figsize=(3,3))

    model_ids = ['120-Myr', 'Praesepe', 'NGC-6811', '2.6-Gyr', 'M67']
    ages = ['120 My', '670 My', '1 Gy', '2.6 Gy', '4 Gy']
    yvals = [9.8,14.8,16.7,20,28]

    _Teff = np.linspace(3800, 6200, int(1e3))
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot', 'solid']
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
            _Teff, _Prot, color=color, linewidth=1, zorder=10, alpha=0.4, ls=ls
        )

        bbox = dict(facecolor='white', alpha=1, pad=0, edgecolor='white')
        ax.text(3680, yval, age, ha='right', va='center', fontsize='x-small',
                bbox=bbox, zorder=49)

    print(f"Mean Teff error is {np.nanmean(Teff_errs):.1f} K")

    #N_reported_periods = df['N_reported_periods']
    #sel = (N_reported_periods >= 2)

    ax.errorbar(
        Teffs, Prots, #xerr=Teff_errs,
        yerr=np.zeros(len(Prots)),
        marker='o', elinewidth=0., capsize=0, lw=0, mew=0., color='k',
        markersize=0.5, zorder=5
    )

    ## only one reported period
    #sel = (N_reported_periods == 1)
    #ax.errorbar(
    #    Teffs[sel], Prots[sel], #xerr=Teff_errs,
    #    yerr=Prot_errs[sel],
    #    marker='o', elinewidth=0.5, capsize=0, lw=0, mew=0.5, color='lightgray',
    #    markersize=1, zorder=4
    #)

    txt = (
        #"$N_\mathrm{p}$ = " + f"{n_pl}\n"
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


def plot_li_vs_teff(outdir, sampleid='koi_X_S19S21dquality', yscale=None,
                    limodel='eagles', show_dispersion=0):

    #df = get_li_data('all')
    df = get_li_data(sampleid)

    n_pl = len(np.unique(df.kepoi_name))
    n_st = len(np.unique(df.kepid))

    Teffs = nparr(df.adopted_Teff)
    Teff_errs = nparr(df.adopted_Teff_err)

    # CKS-Young lithium dataset
    IRON_OFFSET = 10 #TODO FIXME CALIBRATE
    li_ew = df['Fitted_Li_EW_mA'] - IRON_OFFSET
    li_ew_perr = df['Fitted_Li_EW_mA_perr']
    li_ew_merr = df['Fitted_Li_EW_mA_merr']
    CUTOFF = 10 # mA: consider anything less a nondetection
    upperlim = li_ew - li_ew_merr < CUTOFF
    det = ~upperlim
    li_ew_upper_lims = li_ew_perr[upperlim]

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
        tstep = 0.002
        Teff_model = np.arange(3.4772, 3.8130, tstep)

        if not show_dispersion:
            ages = [120, 300, 670, 1000, 4000]
        else:
            ages = [120, 670, 4000]
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

    linestyles = ['solid', 'dotted', 'dashed', 'dashdot', 'solid']
    for ix, (li_model, ls, age) in enumerate(zip(li_models, linestyles, ages)):
        color = 'k'
        agestr = f'{age/1e3:.2f} Gyr'
        ax.plot(
            Teff_model, li_model, color=color, linewidth=1, zorder=10,
            alpha=0.4, ls=ls, label=agestr
        )

        if show_dispersion :
            ax.fill_between(
                Teff_model,
                li_model-li_model_dispersion[ix],
                li_model+li_model_dispersion[ix],
                alpha=0.3
            )

    print(f"Mean Teff error is {np.nanmean(Teff_errs):.1f} K")

    yerr = np.array(
        [li_ew_merr[det], li_ew_perr[det]]
    ).reshape((2, len(li_ew[det])))

    ax.errorbar(
        Teffs[det], li_ew[det], #xerr=Teff_errs,
        yerr=yerr,
        marker='o', elinewidth=0.5, capsize=0, lw=0, mew=0.5, color='k',
        markersize=1, zorder=5
    )

    ax.scatter(
        Teffs[upperlim], li_ew_upper_lims,
        marker='$\downarrow$', s=2, color='k', zorder=4,
        linewidths=0
    )

    leg = ax.legend(loc='upper right', handletextpad=0.3, fontsize='small',
                    framealpha=0, borderaxespad=0, borderpad=0,
                    handlelength=1.6, bbox_to_anchor=(0.97, 0.97))

    txt = (
        "$N_\mathrm{p}$ = " + f"{n_pl}\n"
        "$N_\mathrm{s}$ = " + f"{n_st}"
    )
    ax.text(0.03, 0.97, txt, transform=ax.transAxes,
            ha='left',va='top', color='k')

    ax.set_ylabel('Li$_{6708}$ EW [m$\mathrm{\AA}$]')
    ax.set_xlabel('Effective Temperature [K]')

    if yscale == 'log':
        ax.set_yscale('log')
        ax.set_ylim([1, 300])

    ax.set_xlim([ 6300, 3700 ])
    ax.set_ylim([ -5, 270 ])

    s = f'_{sampleid}'
    s += f"_{limodel}"
    if yscale == 'log':
        s += "_logy"
    if show_dispersion:
        s += "_showdispersion"

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

    ##TODO: what if you sampled instead?
    #import IPython; IPython.embed()


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
    _df, d, st_ages = get_age_results(whichtype='gyro', drop_grazing=0)
    df = pd.DataFrame(d)

    # >33% radii
    #sel = (df['rp']/df['rp_err1'] > 3) & (df['rp']/df['rp_err2'] > 3)
    # >25% radii
    #sel = (df['rp']/df['rp_err1'] > 4) & (df['rp']/df['rp_err2'] > 4)
    # >20% radii
    sel = (df['rp']/df['rp_err1'] > 5) & (df['rp']/df['rp_err2'] > 5)
    AGE_MAX = 10**9.5 #(3.2 gyr)
    #AGE_MAX = 3e9

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


def plot_sub_praesepe_selection_cut(mdf5, poly_order=7):
    """
    make figure showing which stars have rotation periods that are being
    selected via "sub-praesepe"
    (called from drivers/build_koi_table.py)
    """

    plt.close("all")
    fig, ax = plt.subplots(figsize=(4,3))

    csvpath = join(
        DATADIR, 'interim', 'slow_sequence_manual_selection',
        'Praesepe_slow_sequence.csv'
    )
    df_prae = pd.read_csv(csvpath)

    teff_model = np.arange(3800, 6200+1, 1)
    prot_model = reference_cluster_slow_sequence(
        teff_model, "Praesepe", poly_order=poly_order
    )

    ax.scatter(
        df_prae["Teff"], df_prae["Prot"], c='C0', marker='o', zorder=4,
        s=3, linewidths=0.5, label='Praesepe'
    )
    ax.plot(
        teff_model, prot_model, c='C0', ls='-', zorder=3, lw=1
    )

    period_cols = ['s19_Prot', 's21_Prot', 'm14_Prot', 'm15_Prot']

    for period_col in period_cols:
        # all KOIs
        if period_col == 's19_Prot':
            ax.scatter(
                mdf5["adopted_Teff"], mdf5[period_col], c='darkgray', marker='o',
                zorder=0, s=1, linewidths=0, label='KOIs'
            )
        else:
            # legend label trick
            ax.scatter(
                mdf5["adopted_Teff"], mdf5[period_col], c='darkgray', marker='o',
                zorder=0, s=1, linewidths=0
            )

        # KOIs below Praesepe
        sel_teff_range = (
            (mdf5["adopted_Teff"] >= 3800)
            &
            (mdf5["adopted_Teff"] <= 6200)
        )

        period_val = mdf5[period_col][sel_teff_range]
        teff_val = mdf5["adopted_Teff"][sel_teff_range]

        sel_below_Praesepe = (
            period_val < reference_cluster_slow_sequence(
                teff_val, "Praesepe", poly_order=poly_order
            )
        )

        if period_col == 's19_Prot':
            ax.scatter(
                teff_val[sel_below_Praesepe], period_val[sel_below_Praesepe],
                c='C1', marker='o', zorder=4, linewidths=0.5, s=2,
                label='KOIs below Praesepe'
            )
        else:
            # legend label trick
            ax.scatter(
                teff_val[sel_below_Praesepe], period_val[sel_below_Praesepe],
                c='C1', marker='o', zorder=4, linewidths=0.5, s=2,
            )

    ax.legend(loc='upper left', fontsize='xx-small', framealpha=1)

    ax.update({
        'xlabel': 'B+20 Teff',
        'ylabel': 'Prot [M14,M15,S19,S21]',
        'ylim': [0,20],
        'xlim': [6500, 3600]
    })

    outdir = join(RESULTSDIR, 'debug')
    if not os.path.exists(outdir): os.mkdir(outdir)
    outpath = join(
        outdir, f"koi_table_sub_praesepe_selection_cut_poly{poly_order}.png"
    )
    fig.savefig(outpath, dpi=350, bbox_inches='tight')


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


def plot_process_koi_li_posteriors(outdir, cache_id, li_method='eagles'):

    from gyrointerp.helpers import get_summary_statistics

    csvdir = os.path.join(RESULTSDIR, cache_id)
    if li_method == 'baffles':
        csvpaths = glob(os.path.join(csvdir, "*lithium.csv"))
        raise NotImplementedError('need a way to propagate Li EWs...')
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
        ewpath = join(csvdir, f"{kepoi_name}.csv")
        assert os.path.exists(ewpath)
        kepoi_names.append(kepoi_name)

        df = pd.read_csv(csvpath, names=['age_grid','age_post'], comment='#')
        ewdf = pd.read_csv(ewpath)

        t_post = np.array(df.age_post)

        if li_method == 'baffles':
            age_grid = np.array(df.age_grid) # already in myr
        elif li_method == 'eagles':
            age_grid = 10**np.array(df.age_grid) / (1e6) # convert to myr

        d = get_summary_statistics(age_grid, t_post)
        # write the Li EWs that eagles / baffles actually used
        d['LiEW'] = ewdf.LiEW.iloc[0]
        d['eLiEW'] = ewdf.eLiEW.iloc[0]
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

    _csvpath = join(
        DATADIR, "interim", "koi_jump_getter_koi_X_S19S21dquality.csv"
    )
    kdf = pd.read_csv(_csvpath)

    _mdf = df.merge(kdf, how='inner', on='kepoi_name')

    # Check if "li_median" agree for duplicate kepoi_name entries...
    result = _mdf.groupby('kepoi_name')['li_median'].apply(
        lambda x: x.nunique() == 1
    )
    all_match = result.all()
    assert all_match

    # If true, drop duplicates
    mdf = _mdf[~_mdf.duplicated('kepoi_name', keep='first')]

    assert len(mdf) == len(df)

    # Write lithium result contents
    csvpath = join(
        outdir, f"{li_method}_koi_lithium_ages_X_S19S21_dquality.csv"
    )
    mdf = mdf.sort_values(by='li_median')
    mdf.to_csv(csvpath, index=False)
    print(f"Wrote {csvpath}")

    cols = ['kepoi_name', 'kepler_name', 'li_median', 'li_+1sigma', 'li_-1sigma',
            'li_+2sigma', 'li_-2sigma', "li_+1sigmapct", "li_-1sigmapct", "koi_prad"]
    sel_2s = mdf['li_median'] + mdf['li_+2sigma'] < 1000

    print(mdf[sel_2s][cols])



def plot_age_comparison(outdir, logscale=1, iso_v_gyroli=0, ratio_v_gyroli=0,
                        hist_age_unc_ratio=0):

    # get data
    df = get_joint_results(COMPARE_AGE_UNCS=1)

    # plot all stars
    plt.close("all")
    set_style('clean')
    fig, ax = plt.subplots()

    sel = (
        (~pd.isnull(df['adopted_age_median']))
        &
        (~pd.isnull(df['P22S_age-iso']))
    )

    if iso_v_gyroli:

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

    if ratio_v_gyroli:

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

        gyroli_m1sig = np.abs(1e6*df.loc[sel, 'adopted_age_-1sigma'])
        gyroli_p1sig = 1e6*df.loc[sel, 'adopted_age_+1sigma']
        iso_m1sig = np.abs(1e9*df.loc[sel, 'P22S_e_age-iso'])
        iso_p1sig = 1e9*df.loc[sel, 'P22S_E_age-iso']

        iso_gyroli_p1sig_ratio = iso_p1sig / gyroli_p1sig
        iso_gyroli_m1sig_ratio = iso_m1sig / gyroli_m1sig

        bins = np.logspace(-1.5,3,10)
        ax.hist(iso_gyroli_p1sig_ratio, bins=bins, color='C0', alpha=0.5,
                density=False, histtype='step')
        ax.hist(iso_gyroli_m1sig_ratio, bins=bins, color='C1', alpha=0.5,
                density=False, histtype='step')


    n_st = len(np.unique(df[sel].kepoi_number_str))
    n_pl = len(np.unique(df[sel].kepoi_name))
    txt = (
        "$N_\mathrm{p}$ = " + f"{n_pl}\n"
        "$N_\mathrm{s}$ = " + f"{n_st}"
    )
    if iso_v_gyroli:
        xloc, yloc, va, ha = 0.97, 0.03, 'bottom', 'right'
    if ratio_v_gyroli:
        xloc, yloc, va, ha = 0.97, 0.97, 'top', 'right'
    if hist_age_unc_ratio:
        xloc, yloc, va, ha = 0.05, 0.97, 'top', 'left'

    ax.text(xloc, yloc, txt, transform=ax.transAxes, ha=ha, va=va, color='k')

    if hist_age_unc_ratio:
        ax.text(0.97, 0.97, '+1σ', transform=ax.transAxes, ha='right',
                va='top', color='C0')
        ax.text(0.97, 0.92, '-1σ', transform=ax.transAxes, ha='right',
                va='top', color='C1')

    if iso_v_gyroli:
        ax.set_xlabel("Age Gyro+Li [yr]")
        ax.set_ylabel("Age Iso [yr] (Petigura+22)")
    if ratio_v_gyroli:
        ax.set_xlabel("Age Gyro+Li [yr]")
        ax.set_ylabel("Age Iso / Age Gyro+Li")
    if hist_age_unc_ratio:
        ax.set_xlabel(r"σ$_{\rm Iso}^{\rm Petigura\!+\!\!22}$ / σ$_{\rm Gyro+Li}^{\rm This\ Work}$")
        ax.set_ylabel("Count")

    if logscale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    if iso_v_gyroli:
        ax.set_xlim([3e7, 20e9])
        ax.set_ylim([3e7, 20e9])

    s = ''
    if logscale:
        s += '_logscale'
    if iso_v_gyroli:
        s += '_iso_v_gyroli'
    if ratio_v_gyroli:
        s += '_ratio_v_gyroli'
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


def plot_hist_field_gyro_ages(outdir, cache_id, MAXAGE=4000, datestr='20240405'):

    from gyrointerp.paths import CACHEDIR
    csvdir = join(CACHEDIR, f"samples_field_gyro_posteriors_{datestr}")

    mergedcsv = join(csvdir, f'merged_{cache_id}_samples_{datestr}.csv')
    if not os.path.exists(mergedcsv):

        csvpaths = glob(join(csvdir, "*samples.csv"))
        assert len(csvpaths) > 0
        N_post_samples = 10*len(csvpaths)

        df_list = []
        for f in csvpaths:

            bn = os.path.basename(f)
            kic_id = bn.split("_")[0]

            this_df = pd.read_csv(f)
            this_df['KIC'] = kic_id

            df_list.append(this_df)

        mdf = pd.concat(df_list)
        mdf.to_csv(mergedcsv, index=False)
        print(f"Wrote {mergedcsv}")

    else:
        mdf = pd.read_csv(mergedcsv)
        N_post_samples = len(mdf)

    print(f"Got {N_post_samples} posterior samples...")

    kdf = get_gyro_data("Santos19_Santos21_dquality", drop_grazing=0)
    skdf = kdf[kdf.flag_is_gyro_applicable]
    skdf['KIC'] = skdf.KIC.astype(str)
    mdf['KIC'] = mdf.KIC.astype(str)
    sel_gyro_ok = mdf.KIC.isin(skdf.KIC)

    # SAMPLES from the age posteriors
    plt.close("all")
    set_style('clean')
    fig, ax = plt.subplots()

    bw = 200
    bins = np.arange(0, (MAXAGE+2*bw)/1e3, bw/1e3)

    ax.hist(mdf.age, bins=bins, color='lightgray', density=False, zorder=1,
            label='all')

    ax.hist(mdf[sel_gyro_ok].age, bins=bins, color='C0',
            density=False, zorder=2, label='gyro applicable')

    ax.legend(loc='best', fontsize='small')

    xmin = 0
    xmax = MAXAGE
    ax.update({
        'xlabel': 'Age [Myr]',
        'ylabel': 'Count (10x over-rep)',
        'xlim': [xmin, xmax],
    })
    outpath = os.path.join(outdir, f'hist_samples_field_gyro_ages_{cache_id}_maxage{MAXAGE}.png')
    savefig(fig, outpath, writepdf=1, dpi=400)

    #################################################
    # ok... now how about the subset that are good? #
    #################################################
    plt.close("all")
    set_style('clean')
    fig, axs = plt.subplots(ncols=2, figsize=(0.9*6, 0.9*3), constrained_layout=True)

    koi_df = get_koi_data('cumulative-KOI', drop_grazing=0)
    koi_df['kepid'] = koi_df['kepid'].astype(str)
    skoi_df = koi_df[koi_df['flag_is_ok_planetcand']]
    sel_planets = mdf.KIC.isin(skoi_df.kepid)

    N = int(len(mdf)/10)
    l0_0 = f'{N} w/ '+'P$_{\mathrm{rot}}$'
    axs[0].hist(mdf.age/1e3, bins=bins, color='lightgray',
                histtype='step',
                weights=np.ones(len(mdf))/len(mdf),
                zorder=1, label=l0_0, alpha=0.6)
    N = int(len(mdf[sel_gyro_ok])/10)
    l0_1 = f'{N} w/ '+'P$_{\mathrm{rot}}$ & gyro applicable'
    axs[0].hist(mdf[sel_gyro_ok].age/1e3, bins=bins, color='C0',
                histtype='step',
                weights=np.ones(len(mdf[sel_gyro_ok]))/len(mdf[sel_gyro_ok]),
                zorder=2, alpha=0.81,
                label=l0_1)

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
    ulkvp('ratiombtoybstars', np.round(n_mb/n_yb, 1))
    ulkvp('ratiombtoybstars', np.round(n_mb/n_yb, 1))
    ulkvp('ratioobtoybstars', np.round(n_ob/n_yb, 1))
    ##########################################

    axs[0].legend(loc='best', fontsize='xx-small')

    N = int(len(mdf[sel_planets])/10)
    l1_0 = f'{N} w/ '+'P$_{\mathrm{rot}}$'
    axs[1].hist(mdf[sel_planets].age/1e3, bins=bins, color='lightgray',
                histtype='step',
                weights=np.ones(len(mdf[sel_planets]))/len(mdf[sel_planets]),
                zorder=1,
                label=l1_0, alpha=0.6)
    psel = sel_gyro_ok & sel_planets
    N = int(len(mdf[psel])/10)
    l1_1 = f'{N} w/ '+'P$_{\mathrm{rot}}$ & gyro applicable'
    axs[1].hist(mdf[psel].age/1e3, bins=bins,
                histtype='step',
                weights=np.ones(len(mdf[psel]))/len(mdf[psel]),
                color='C0', alpha=0.81, zorder=2,
                label=l1_1)

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
    ulkvp('ratiombtoybplanets', np.round(n_mb/n_yb, 1))
    ulkvp('ratioobtoybplanets', np.round(n_ob/n_yb, 1))
    ##########################################

    xmin = 0
    xmax = MAXAGE
    axs[0].update({
        'xlabel': '$t_{\mathrm{gyro}}$ [Gyr]',
        'ylabel': f'Fraction',
        'xlim': [xmin/1e3, (xmax-20)/1e3],
        'ylim': [0, 0.065],
        #'title': 'Kepler stars'
    })
    axs[0].text(.08, .97, 'Kepler targets', ha='left', va='top',
                fontsize='large', zorder=5, transform=axs[0].transAxes,
                fontdict={'fontstyle':'oblique'})
    if MAXAGE < 4000:
        axs[0].set_xticks([0, 1, 2, 3])

    axs[1].update({
        'xlabel': '$t_{\mathrm{gyro}}$ [Gyr]',
        'ylabel': f'Fraction',
        'xlim': [xmin/1e3, (xmax-20)/1e3],
        'ylim': [0, 0.065],
        #'title': 'KOI host stars'
    })
    axs[1].text(.08, .97, 'KOI host stars', ha='left', va='top',
                fontsize='large', zorder=5, transform=axs[1].transAxes,
                fontdict={'fontstyle':'oblique'})
    if MAXAGE < 4000:
        axs[1].set_xticks([0, 1, 2, 3])

    #axs[1].set_yticklabels([])

    # AESTHETIC HAD WEIRD ISSUES...
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='lightgray', lw=1, alpha=0.6),
                    Line2D([0], [0], color='C0', alpha=0.81, lw=1) ]
    axs[0].legend(custom_lines, [l0_0, l0_1], fontsize='xx-small',
                  borderaxespad=2.0, borderpad=0.8, framealpha=0,
                  loc='lower right')
    axs[1].legend(custom_lines, [l1_0, l1_1], fontsize='xx-small',
                  borderaxespad=2.0, borderpad=0.8, framealpha=0,
                  loc='lower right')

    outpath = os.path.join(outdir, f'hist_samples_koi_gyro_ages_{cache_id}_maxage{MAXAGE}.png')
    fig.tight_layout()
    savefig(fig, outpath, writepdf=1, dpi=400)

    ###########################################
    ###########################################

    from scipy import stats
    sel_age = mdf.age < 3.2e9
    D, p_value = stats.ks_2samp(
        mdf.loc[sel_gyro_ok & sel_age, 'age'],
        mdf.loc[sel_gyro_ok & sel_planets & sel_age, 'age']
    )

    txt = f'D={D:.2f}, p={p_value:.2e} 2-sample KS'
    print(42*'-')
    print(txt)
    print(42*'-')

    ##########################################
    # end the interesting plot               #
    ##########################################

    # ok, now just plot the histogram of the median values...
    csvpath = join(RESULTSDIR, "field_gyro_posteriors_20230529",
                   "field_gyro_posteriors_20230529_gyro_ages_X_GDR3_S19_S21_B20.csv")
    df = pd.read_csv(csvpath)

    plt.close("all")
    set_style('clean')
    fig, ax = plt.subplots()

    ax.hist(df['median'], bins=bins, color='lightgray', density=True)

    xmin = 0
    xmax = MAXAGE
    ax.update({
        'xlabel': 'Age [Myr]',
        'ylabel': 'Fraction',
        'xlim': [xmin, xmax],
    })
    outpath = os.path.join(
        outdir,
        f'hist_medianvals_field_gyro_ages_{cache_id}_maxage{MAXAGE}.png'
    )
    savefig(fig, outpath, writepdf=1, dpi=400)


def plot_field_gyro_posteriors(outdir, cache_id):

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
        xmax = 4000
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

    sampleid = 'Santos19_Santos21_all'
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
    _df, d, st_ages = get_age_results(whichtype='gyro', drop_grazing=0)
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
    if sampleid == 'Santos19_Santos21_dquality':
        kicdf = kicdf[kicdf['flag_is_gyro_applicable']]

    # planets
    koidf, _, _ = get_age_results(whichtype='gyro_li', COMPARE_AGE_UNCS=0,
                                  drop_grazing=0, drop_highruwe=0)

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
        ax.errorbar(
            xval, yval, yerr=yerr,
            marker='o', elinewidth=0.05, capsize=0, lw=0, mew=0.5, color='k',
            markersize=0, zorder=5, alpha=0.5
        )
    else:
        ax.errorbar(
            xval, yval, yerr=yerr,
            marker='o', elinewidth=0.0, capsize=0, lw=0, mew=0.5, color='k',
            markersize=0.5, zorder=5, alpha=0.5
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
                marker='o', elinewidth=0.5, capsize=0, lw=0, mew=0.5, color='C0',
                markersize=0, zorder=5, alpha=1
            )
        else:
            ax.errorbar(
                xval, yval, yerr=yerr,
                marker='o', elinewidth=0.0, capsize=0, lw=0, mew=0.5, color='C0',
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


def plot_st_params(outdir, xkey='dr3_bp_rp', ykey='M_G'):

    # get data

    # stars (w/ Prot)
    sampleid = 'Santos19_Santos21_dquality'
    kicdf = get_kicstar_data(sampleid)

    # planets (for gyro)
    koidf, _, _ = get_age_results(whichtype='gyro_li', COMPARE_AGE_UNCS=0,
                                  drop_grazing=0, drop_highruwe=0)

    # get KIC target stars
    from gyrojo.getters import (
        get_cleaned_gaiadr3_X_kepler_supplemented_dataframe
    )
    cgk_df = get_cleaned_gaiadr3_X_kepler_supplemented_dataframe()

    #  # if you wanted all KOIs
    #  koi_df = get_koi_data('cumulative-KOI', drop_grazing=0)
    #  koi_df['kepid'] = koi_df['kepid'].astype(str)
    #  # REQUIRE "flag_is_ok_planetcand"
    #  skoi_df = koi_df[koi_df['flag_is_ok_planetcand']]

    # make plot
    plt.close('all')
    set_style('clean')

    fig, ax = plt.subplots(figsize=(3,3))

    dfs = [
        cgk_df,
        kicdf,
        kicdf[kicdf['flag_is_gyro_applicable']],
        koidf
    ]
    colors = [
        'lightgray',
        'gray',
        'k',
        'C0'
    ]
    zorders = [
        -1,
        0,
        1,
        2
    ]
    sizes = [
        0.1,
        0.3,
        1,
        1.5
    ]
    labels = [
        'KIC',
        '...w/ Prot',
        '...& gyro applicable',
        '...& KOI',
    ]
    rasterized = [
        True,
        True,
        True,
        False
    ]

    for df, c, z, l, s, r in zip(
        dfs, colors, zorders, labels, sizes, rasterized
    ):

        yval = nparr(df[ykey])
        xval = nparr(df[xkey])
        ax.scatter(
            xval, yval,
            marker='o', c=c, zorder=z, s=s, linewidths=0,
            label=l+f" ($N$={len(xval)})", rasterized=r
        )

    if xkey == 'dr3_bp_rp':
        xlabel = '$G_\mathrm{BP}-G_{\mathrm{RP}}$ [mag]'
        xlim = [0.1, 4.05]
    elif xkey == 'adopted_Teff':
        xlabel = '$T_\mathrm{eff,adopted}$ [K]'
        xlim = [7500, 2500]

    if ykey == 'M_G':
        ylabel = '$M_\mathrm{G}$ [mag]'
        ylim = [15.5, -3.5]
    elif ykey == 'adopted_logg':
        ylabel = '$(\log g)_\mathrm{adopted}$ [dex]'
        ylim = [6.5, 2.5]
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
        ax.set_yticks([15, 10, 5, 0])

    ax.legend(
        loc='lower left', fontsize='x-small',
        markerscale=3,
        framealpha=0,
        #handletextpad=0.3, framealpha=0, borderaxespad=0, borderpad=0,
        #handlelength=1.6#, bbox_to_anchor=(0.97, 0.97)
        handletextpad=0.1, borderaxespad=0.5, borderpad=0.5
    )


    # set naming options
    s = f'_{ykey}_vs_{xkey}'

    outpath = join(outdir, f'st_params{s}.png')
    savefig(fig, outpath, dpi=400)
