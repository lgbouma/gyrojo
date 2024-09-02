"""
Make scatter plot of size versus period (colored by age).
"""

# Standard imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
import os
from astropy import units as u
from numpy import array as nparr
from matplotlib import rcParams

#plt.rcParams["font.family"] = "Times New Roman"

# If you want to run the code, you'll need to do:
# `git clone https://github.com/lgbouma/cdips; cd cdips; python setup.py install`
# `git clone https://github.com/lgbouma/aesthetic; cd aesthetic; python setup.py install`
from cdips.utils import today_YYYYMMDD
from aesthetic.plot import savefig, format_ax, set_style

from getters import get_nea_data, get_auxiliary_data

# This "VER" string caches the NASA exoplanet archive `ps` table at a
# particular date, in the YYYYMMDD format.
VER = '20230627' # could be today_YYYYMMDD()
VER = '20240415' # could be today_YYYYMMDD()

def plot_rp_vs_period_scatter(
    showlegend=1, showarchetypes=1, showss=1, colorbyage=0,
    verbose=0, add_allkep=0, add_plnames=0, showauxiliarysample=0, s=None,
    add_david2021=0, showNEA=1, showclusterplanets=1, NEAselfn='strictyoung',
    selfntxt=0, N_colors=6, arial_font=0, nograzingnoruwe=1
):
    """
    Plot planetary parameters versus ages. By default, it writes the plots to
    '../results/rp_vs_period_scatter/' from wherever you put this script.
    (See `outdir` parameter below).

    Options (all boolean):

        showauxiliarysample: 0 or a string identifying the auxiliary sample.

        showlegend: whether to overplot a legend.

        colorbydisc: whether to color by the discovery method.

        showarchetypes: whether to show "Hot Jupiter", "Cold Jupiter" etc
        labels for talks.

        showss: whether to show the solar system planets.

        colorbyage: whether to color the points by their ages.

        verbose: if True, prints out more information about the youngest
        planets from the NASA exoplanet archive.

        add_kep1627: adds a special star for Kepler 1627.

        add_allkep: adds special symbols for the recent Kepler systems in open
        clusters: 'Kepler-52', 'Kepler-968', 'Kepler-1627', 'KOI-7368'

        add_plnames: if True, shows tiny texts for the age-dated planets.
    """

    set_style("clean")
    if arial_font:
        rcParams['font.family'] = 'Arial'

    paramdict, ea_df, sdf, sel, s0, s1 = get_nea_data(
        VER, colorbyage, NEAselfn=NEAselfn
    )
    mp = paramdict['mp']
    rp = paramdict['rp']
    age = paramdict['age']
    period = paramdict['period']
    discoverymethod = paramdict['discoverymethod']
    pl_name = paramdict['pl_name']

    if showauxiliarysample:
        auxparamdict = (
            get_auxiliary_data(showauxiliarysample)
        )
        a_period = auxparamdict['period']
        a_rp = auxparamdict['rp']
        a_age = auxparamdict['age']
        a_pl_name = auxparamdict['pl_name']
        a_starquality = auxparamdict['starquality']
        a_planetquality = auxparamdict['planetquality']
        assert pd.isnull(a_rp).sum() == 0
        assert pd.isnull(a_period).sum() == 0

        # update radii of any overlaps to match NEA
        _sel = nparr(a_pl_name.isin(pl_name[s1]))
        _isel = nparr(pl_name[s1].isin(a_pl_name))
        assert _sel.sum() == _isel.sum()

        # Create a dictionary mapping planet names to their corresponding rp values
        rp_dict = dict(zip(pl_name[s1], rp[s1]))
        a_rp_dict = dict(zip(a_pl_name, a_rp))

        # Iterate over a_rp_dict and update values based on matching planet names in rp_dict
        for planet_name, rp_value in a_rp_dict.items():
            if planet_name in rp_dict:
                a_rp_dict[planet_name] = rp_dict[planet_name]

        # Convert the updated a_rp_dict back to a Series if needed
        a_rp = np.array(pd.Series(a_rp_dict))

        print(42*'!')
        print(NEAselfn)
        print(len(a_period))
        print(a_pl_name[_sel])
        print(42*'!')

    #
    # Plot age vs rp. (age is on y axis b/c it has the error bars, and I at
    # least skimmed the footnotes of Hogg 2010).
    #
    #fig,ax = plt.subplots(figsize=(1.3*4,1.3*3))
    #fig,ax = plt.subplots(figsize=(1.1*4,1.1*3))
    fig,ax = plt.subplots(figsize=(0.8*4,0.8*3))

    if showNEA:
        ax.scatter(period[s0], rp[s0],
                   color='darkgray', s=1., zorder=1, marker='o', linewidth=0,
                   alpha=1, rasterized=0)

    if add_david2021:
        ddf = pd.read_csv('../data/David2021_fig2_lt_9pt25.csv')
        ax.scatter(ddf.pl_period, ddf.pl_radius,
                   color='darkgray', s=2, zorder=1, marker='o', linewidth=0,
                   alpha=1, rasterized=True)

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    if colorbyage:
        if showclusterplanets or add_allkep:
            axins1 = inset_axes(ax, width="3%", height="25%", loc='lower right',
                                borderpad=1.0)
    cmap = mpl.cm.get_cmap('magma_r', N_colors)
    #cmap = mpl.cm.get_cmap('magma', N_colors)
    #cmap = mpl.cm.get_cmap('YlGnBu', N_colors)
    #cmap = mpl.cm.get_cmap('OrRd_r', N_colors)
    #cmap = mpl.cm.get_cmap('viridis', N_colors)
    #cmap = mpl.cm.get_cmap('PuBuGn', N_colors)
    bounds = np.arange(7.0,9.0,0.01)
    norm = mpl.colors.LogNorm(vmin=1e7, vmax=1e9)

    if s is None:
        s = 36
        s = 18
        #s = 15
    if showauxiliarysample:
        s = 11
        #s = 15

    if not showauxiliarysample:
        alpha = 1
    else:
        alpha = 1

    if colorbyage:

        # draw the colored points
        if showclusterplanets:
            _p = ax.scatter(
                period[s1], rp[s1],
                c=age[s1], alpha=alpha, zorder=2, s=s, edgecolors='k',
                marker='o', cmap=cmap, linewidths=0.2, norm=norm
            )
            N = len(period[s1])
            print(42*'~!')
            print(f'selfn={NEAselfn}, N={N} planets from NEA')

        if selfntxt:
            if NEAselfn == 'strictyoung':
                txt = '$t$/$\sigma_t$>3'
            elif NEAselfn == 'anyyoung':
                txt = '$t$+$2\sigma_t$<10$^9$yr'
            ax.text(0.98, 0.98, txt, ha='right', va='top', fontsize='medium',
                    #fontdict={'fontstyle':'oblique'},
                    zorder=5,
                    transform=ax.transAxes)


        if add_plnames:
            bbox = dict(facecolor='white', alpha=0.8, pad=0, edgecolor='white',
                       lw=0)
            for _x,_y,_s in zip(period[s1],rp[s1],pl_name[s1]):
                ax.text(_x, _y, _s, ha='right', va='bottom', fontsize=1.5,
                        bbox=bbox, zorder=49)

        if add_allkep:
            # Kepler-52 and Kepler-968
            namelist = ['Kepler-52', 'Kepler-968', 'Kepler-1627', 'KOI-7368',
                        'KOI-7913 A', 'Kepler-1643']
            ages = [3.5e8, 3.5e8, 3.8e7, 3.8e7, 3.8e7, 3.8e7]
            markers = ['d','*','o','o','o','o']
            sizes = [75, 100, 36, 36, 36, 36]
            #markers = ['*','*','*', '*', '*', '*']
            #sizes = [80, 80, 80, 80, 80, 80]

            # namelist = ['Kepler-1627', 'KOI-7368',
            #             'KOI-7913 A', 'Kepler-1643']
            # ages = [3.8e7, 3.8e7, 3.8e7, 3.8e7]
            # markers = ['o'] * 4
            # #sizes = [180] * 4
            # sizes = [36] * 4

            for n, a, m, _s in zip(namelist, ages, markers, sizes):
                sel = ea_df.hostname == n

                _sdf = ea_df[sel]
                _rp = _sdf.pl_rade
                _per= _sdf.pl_orbper
                print(n, _rp, _per)
                if n in ['KOI-7368', 'KOI-7913 A', 'Kepler-1643', 'Kepler-1627']:
                    # Exoplanet archive correctly updated these
                    continue

                _age = np.ones(len(_sdf))*a

                if showclusterplanets:
                    ax.scatter(
                        _per, _rp,
                        c=_age, alpha=1, zorder=2, s=_s, edgecolors='k',
                        marker=m, cmap=cmap, linewidths=0.2, norm=norm
                    )
                else:
                    _p = ax.scatter(
                        _per, _rp,
                        c=_age, alpha=1, zorder=2, s=_s, edgecolors='k',
                        marker=m, cmap=cmap, linewidths=0.2, norm=norm
                    )

                if add_plnames:
                    print(np.array(_sdf.pl_name), np.array(_per),
                          np.array(_rp))
                    for __n, __per, __rp in zip(
                        np.array(_sdf.pl_name), np.array(_per), np.array(_rp)
                    ):
                        ax.text(__per, __rp, __n, ha='right', va='bottom',
                                fontsize=1.5, bbox=bbox, zorder=49)

    if showauxiliarysample:

        age_boundaries = np.linspace(7, 9, N_colors+1)
        #age_boundaries = [9,10] #FIXME
        age_bins = [
            (lo,hi) for lo,hi in zip(age_boundaries[:-1], age_boundaries[1:])
        ]

        N = 0
        for age_bin, zord in zip(age_bins, range(len(age_bins))):

            lo, hi = 10**age_bin[0], 10**age_bin[1]
            sel = (a_age >= lo) & (a_age < hi)

            print(f"{lo}-{hi}: N = {len(a_rp[sel])} KOIs.")

            if nograzingnoruwe:
            # draw the colored points
                _p = ax.scatter(
                    a_period[sel], a_rp[sel],
                    c=a_age[sel], alpha=1, s=4.5*s, edgecolors='k',
                    marker='*', cmap=cmap, linewidths=0.2, norm=norm, zorder=10-zord
                    #c=a_age[sel], alpha=1, s=s, edgecolors='k',
                    #marker='o', cmap=cmap, linewidths=0.2, norm=norm, zorder=10-zord
                )
            else:
                from gyrojo.getters import select_by_quality_bits
                _df = pd.DataFrame({'flag_gyro_quality': a_starquality})
                highruwe = select_by_quality_bits(_df, [7], [1])
                grazing = (a_planetquality.astype(str) == '4')

                sel0 = (
                    sel & ~highruwe & ~grazing
                )
                sel1 = (
                    sel & (highruwe | grazing)
                )

                _p = ax.scatter(
                    a_period[sel0], a_rp[sel0],
                    c=a_age[sel0], alpha=1, s=4.5*s, edgecolors='k',
                    marker='*', cmap=cmap, linewidths=0.2, norm=norm,
                    zorder=10-zord
                )
                _ = ax.scatter(
                    a_period[sel1], a_rp[sel1],
                    c=a_age[sel1], alpha=0.6, s=1.5*s, edgecolors='gray',
                    marker='X', cmap=cmap, linewidths=0.2, norm=norm,
                    zorder=2-zord
                )

            N += len(a_period[sel])
        print(f'selfn={NEAselfn}, N={N} planets from Bouma2024')

        if add_plnames:
            bbox = dict(facecolor='white', alpha=0.7, pad=0, edgecolor='white',
                        lw=0)
            for _x,_y,_s in zip(a_period, a_rp, a_pl_name):
                ax.text(_x, _y, _s, ha='right', va='bottom', fontsize=1.5,
                        bbox=bbox, zorder=49)


    if colorbyage or showauxiliarysample:
        if showclusterplanets or add_allkep:
            cb = fig.colorbar(_p, cax=axins1, orientation="vertical",
                              extend="neither")
            cb.set_ticks([1e7,1e8,1e9])
            cb.ax.tick_params(labelsize='small')
            cb.ax.tick_params(size=0, which='both') # remove the ticks
            cb.ax.yaxis.set_ticks_position('left')
            cb.ax.yaxis.set_label_position('left')
            cb.set_label("Age [years]", fontsize='small', weight='normal')


    if showarchetypes:
        # P, Mp, Rp
        ARCHETYPEDICT = {
            'hot Jupiters': [3, 3*380, 2*1*11.8],
            'cold Jupiters': [365*1.6, 3*380, 2*1*11.8],
            'super-Earths': [8, 1.5, 1.25],
            'mini-Neptunes': [25, 10, 3]
        }

        for k,v in ARCHETYPEDICT.items():
            txt = k
            _x,_y = v[0], v[2]
            bbox = dict(facecolor='white', alpha=0.95, pad=0, edgecolor='white')
            ax.text(_x, _y, txt, va='center', ha='center',
                    color='k', bbox=bbox,
                    fontsize='xx-small')

    if showss:
        # P[day], Mp[Me], Rp[Re]
        SSDICT = {
            'Jupiter': [4332.8, 317.8, 11.21, 'J'],
            'Saturn': [10755.7, 95.2, 9.151, 'S'],
            'Neptune': [60190.0, 17.15, 3.865, 'N'],
            'Uranus': [30687, 14.54, 3.981, 'U'],
            'Earth': [365.3, 1, 1, 'E'],
            'Venus': [224.7, 0.815, 0.950, 'V'],
            'Mars': [687.0, 0.1074, 0.532, 'M'],
        }

        for k,v in SSDICT.items():
            txt = k
            _x,_y = v[0], v[2]
            _m = v[3]

            ax.scatter(_x, _y, color='k', s=9, zorder=1000,
                       marker='$\mathrm{'+_m+'}$',
                       linewidth=0, alpha=1)
            ax.scatter(_x, _y, color='white', s=22, zorder=999,
                       marker='o', edgecolors='k',
                       linewidth=0.2, alpha=1)

    ax.set_xlabel('Orbital period [days]', weight='normal', fontsize='medium')
    ax.set_ylabel('Planet radius [Earths]', weight='normal', fontsize='medium')

    if showss:
        ax.set_xlim([0.1, 110000])
    else:
        ax.set_xlim([0.1, 1100])

    ax.set_ylim([0.35,25])
    #ax.set_ylim([0.35,5]) #FIXME

    format_ax(ax)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_yticks([1,2,4,10])
    ax.set_yticklabels([1,2,4,10], fontsize='medium')
    ax.set_xticks([0.1,1,10,100,1000])
    ax.set_xticklabels([0.1,1,10,100,1000], fontsize='medium')


    s = ''
    if showss:
        s += '_showss'
    if showarchetypes:
        s += '_showarchetypes'
    if colorbyage:
        s += '_colorbyage'
    if add_allkep:
        s += '_showallkep'
    if add_plnames:
        s += '_showplnames'
    if showauxiliarysample:
        s += f'_showaux-{showauxiliarysample}'
    if add_david2021:
        s += f'_addDavid2021'
    if not showclusterplanets:
        s += f'_KeplerClusterPlanetsOnly'
    if isinstance(NEAselfn, str):
        s += f'_{NEAselfn}'
    if nograzingnoruwe:
        s += f'_nograzingnoruwe'
    else:
        s += f'_allowsgrazingandhighruwe'

    outdir = '../../results/rp_vs_period_scatter/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    outpath = (
        os.path.join(
            outdir, f'rp_vs_period_scatter_{VER}{s}.png'
        )
    )

    savefig(fig, outpath)


if __name__=='__main__':

    # defaults, "state of the art" according to NEA with cleaning by LGB
    nograzingnoruwe = 0
    N_colors = 6
    for add_plnames in [0,1]:
        for NEAselfn in ['anyyoung','strictyoung']:
            plot_rp_vs_period_scatter(
                showarchetypes=0, showss=0, colorbyage=1, NEAselfn=NEAselfn,
                verbose=1, add_allkep=0, add_plnames=add_plnames,
                selfntxt=0, N_colors=N_colors,
                nograzingnoruwe=nograzingnoruwe
            )
        plot_rp_vs_period_scatter(
            showarchetypes=0, showss=0, colorbyage=1,
            NEAselfn='anyyoung', showauxiliarysample='gyro_anyyoung',
            verbose=1, add_allkep=0, add_plnames=add_plnames,
            selfntxt=0, N_colors=N_colors,
            nograzingnoruwe=nograzingnoruwe
        )
        plot_rp_vs_period_scatter(
            showarchetypes=0, showss=0, colorbyage=1,
            NEAselfn='strictyoung', showauxiliarysample='gyro_selsnrupper3',
            verbose=1, add_allkep=0, add_plnames=add_plnames,
            selfntxt=0, N_colors=N_colors,
            nograzingnoruwe=nograzingnoruwe
        )

    assert 0

    # show isochronally young planets from David+2021
    # (for jason curtis)
    plot_rp_vs_period_scatter(
        showarchetypes=0, showss=0, colorbyage=1, verbose=1, add_allkep=1,
        add_plnames=0, add_david2021=1, showNEA=0, N_colors=N_colors
    )
    plot_rp_vs_period_scatter(
        showarchetypes=0, showss=0, colorbyage=1, verbose=1, add_allkep=1,
        add_plnames=0, add_david2021=1, showNEA=0, showclusterplanets=0,
        N_colors=N_colors
    )
    plot_rp_vs_period_scatter(
        showarchetypes=0, showss=0, colorbyage=1, verbose=1, add_allkep=0,
        add_plnames=0, add_david2021=1, showNEA=0, showclusterplanets=0,
        N_colors=N_colors
    )

    assert 0


