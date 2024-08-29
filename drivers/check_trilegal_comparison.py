"""
Contents:
    plot_projections
    plot_age_histogram
"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from astropy.io import fits, ascii
from astropy.table import Table
import os
from os.path import join
from gyrojo.paths import LOCALDIR, PAPERDIR, DATADIR, TABLEDIR, RESULTSDIR
from gyrojo.getters import select_by_quality_bits
from gyrointerp.helpers import prepend_colstr, left_merge
from aesthetic.plot import set_style, savefig
from collections import Counter
from operator import itemgetter
from numpy import array as nparr

from astropy.coordinates import SkyCoord
import astropy.units as u

from mpl_toolkits.mplot3d import Axes3D

from astropy.coordinates import Galactocentric
import astropy.coordinates as coord
_ = coord.galactocentric_frame_defaults.set('v4.0')

# Sun's position in the galactic cartesian frame
sun_x_pc =  - 8122  # pc

from gyrojo.trilegal import (
    random_skycoords_within_area, galactic_to_xyz, get_trilegal
)

def plot_projections(
    df: pd.DataFrame,
    x_key: str = 'galx',
    y_key: str = 'galy',
    z_key: str = 'galz',
    istrilegal: bool = True,
    kepfield: bool = True
    ) -> None:
    """Plot projections of galactic positions in a 3-panel layout.

    Args:
        df (pd.DataFrame): DataFrame containing galactic positions.
        x_key (str): Column name for x-coordinate. Defaults to 'galx'.
        y_key (str): Column name for y-coordinate. Defaults to 'galy'.
        z_key (str): Column name for z-coordinate. Defaults to 'galz'.
    """
    # Define the mosaic layout
    fig, axs = plt.subplot_mosaic(
        """
        aab
        aac
        """,
        figsize=(3, 2),
        constrained_layout=True
    )

    _sun_x = 0 # -sun_x_pc/1e3
    dr = 4.1

    c = 'C1' if istrilegal else 'C0'

    # Top-down view (Y vs X)
    ax = axs['a']
    ax.scatter(df[x_key], df[y_key], c=c, s=0.5, alpha=0.6, linewidths=0)
    ax.set_xlabel('X-X$_\odot$ (kpc)')
    ax.set_ylabel('Y (kpc)')
    ax.set_xlim([_sun_x - dr, _sun_x + dr])
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.set_ylim([-dr, dr])

    # Z vs X view
    ax = axs['b']
    ax.scatter(df[x_key], df[z_key], c=c, s=0.5, alpha=0.6, linewidths=0)
    ax.set_xlabel('X-X$_\odot$ (kpc)')
    ax.set_ylabel('Z (kpc)')
    ax.set_xlim([_sun_x - dr, _sun_x + dr])
    ax.set_xticks([-4, 0, 4])
    ax.set_ylim([-dr/3, dr/3])

    # Z vs Y view
    ax = axs['c']
    ax.scatter(df[y_key], df[z_key], c=c, s=0.5, alpha=0.6, linewidths=0)
    ax.set_xlabel('Y (kpc)')
    ax.set_ylabel('Z (kpc)')
    ax.set_xlim([-dr, dr])
    ax.set_xticks([-4, 0, 4])
    ax.set_ylim([-dr/3, dr/3])

    # Save the figure
    t = 'trilegal_' if istrilegal else 'realdata_'
    s = ''
    if not kepfield:
        s += '_b0'
    outdir = join(RESULTSDIR, "trilegal")
    outpath = join(outdir, f'{t}xyzproj{s}.png')
    savefig(fig, outpath, writepdf=0, dpi=400)


def plot_age_histogram(df, const_sfr=1, kepfield=1):

    # histogram
    plt.close("all")
    set_style("science")
    fig, ax = plt.subplots(figsize=(3,3))

    bins = np.arange(0, 5.2, 0.2)
    heights, bin_edges, _ = ax.hist(
        df.Age, bins=bins, color='k', histtype='step',
        #weights=np.ones(len(df))/len(df)
    )
    ax.set_xlabel('age [gyr]')
    ax.set_xlim([0, 5.1])
    #ax.set_ylim([0, 0.05])
    ax.set_ylabel('rel. count')

    outdir = join(RESULTSDIR, "trilegal")
    if not os.path.exists(outdir): os.mkdir(outdir)

    s = ''
    if const_sfr:
        s += '_constsfr'
    else:
        s += '_twostepsfr'
    if not kepfield:
        s += '_b0'

    outpath = join(outdir, f'trilegal_agehist{s}.png')
    savefig(fig, outpath, writepdf=0, dpi=400)


def _get_realdata():

    # Bouma 2024 gyro ages...
    csvpath = join(TABLEDIR, "table_star_gyro_allcols.csv")
    bdf = pd.read_csv(csvpath)
    bdf = bdf[bdf.dr3_parallax > 0]
    from earhart.physicalpositions import calculate_XYZ_given_RADECPLX
    bdf['x'], bdf['y'], bdf['z'] = calculate_XYZ_given_RADECPLX(
        nparr(bdf.dr3_ra), nparr(bdf.dr3_dec), nparr(bdf.dr3_parallax)
    )
    bdf['galx'] = (bdf['x'] - sun_x_pc)/1e3
    bdf['galy'] = bdf['y']/1e3
    bdf['galz'] = bdf['z']/1e3

    return bdf


def run_viz(kepfield=1, const_sfr=1):

    age_scale = 0.02
    if not kepfield and not const_sfr:
        age_scale = 0.015
    df = get_trilegal(kepfield=kepfield, const_sfr=const_sfr,
                      age_scale=age_scale)
    bdf = _get_realdata()

    plot_age_histogram(df, const_sfr=const_sfr, kepfield=kepfield)

    plot_projections(df, kepfield=kepfield)

    plot_projections(bdf, istrilegal=False)


if __name__ == "__main__":
    run_viz(const_sfr=0, kepfield=0)
    run_viz(const_sfr=1, kepfield=0)
    run_viz(const_sfr=0, kepfield=1)
    run_viz(const_sfr=1, kepfield=1)
