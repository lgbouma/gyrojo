"""
Contents:
    get_trilegal

helpers:
    random_skycoords_within_area
    galactic_to_xyz
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

def get_trilegal(kepfield=1, const_sfr=1, age_scale=0.02):
    """
    Args:

        kepfield: if true, b=13.5, else b=0

        const_sfr: if true, constant SFR, else two-step

    Trilegal settings:
    l = 76.32, b=13.5.
    Field area 10 deg2
    Kepler band systesm
    Limiting magnitude in 1st filter 16th mag
    Distance mod resolution 0.05 mag
    Default dust extinction calibration
    Sun at x=8122, z=20.8
    CONSTANT SFR by default.
    """
    litdir = join(DATADIR, 'literature')
    if not const_sfr:

        if kepfield:
            # "SFR given by 2 step SFR + Fuhrman's AMR + alpha-enh" "see paper" with with age(yr) = 0.735*t + 0
            datpaths = [
                join(litdir, 'trilegal_kepler_output842802126267.dat'),
                join(litdir, 'trilegal_kepler_output680553906670.dat')
            ]

        else:
            # best way to see what the "2 step SFR" actually means is to look
            # in the plane....
            datpaths = [
                join(litdir, 'trilegal_b0_2step_1sqdeg_output900538207338.dat'),
                join(litdir, 'trilegal_b0_2step_1sqdeg_output736684026503.dat'),
                join(litdir, 'trilegal_b0_2step_1sqdeg_output237029785829.dat'),
                join(litdir, 'trilegal_b0_2step_1sqdeg_output228497039134.dat'),
            ]

    elif const_sfr:

        if kepfield:
            # it's a constant SFR.
            datpaths = [
                join(litdir, 'trilegal_kepler_output69896305286.dat'),
                join(litdir, 'trilegal_kepler_output602727815582.dat')
            ]

        else:
            datpaths = [
                join(litdir, 'trilegal_b0_1sqdeg_output610774677553.dat'),
                join(litdir, 'trilegal_b0_1sqdeg_output346658447132.dat'),
                join(litdir, 'trilegal_b0_1sqdeg_output821294504989.dat'),
            ]

    df = pd.concat((pd.read_csv(f, sep='\s+') for f in datpaths))

    # log10age grid has resolution of +/- 0.02.
    np.random.seed(3141)
    eps = np.random.normal(loc=0, scale=age_scale, size=len(df))
    df['logAge'] += eps
    df['Age'] = 10**df.logAge / 1e9

    # m-M0 has scale of 0.05 dex.
    eps = np.random.normal(loc=0, scale=0.03, size=len(df))
    df['m-M0'] += eps
    df['distance_pc'] = 10 * 10**(df['m-M0']/5)

    sel = (
        (df.Mact < 1.2) &
        (df.Mact > 0.5) &
        (df.distance_pc < 3000) &
        (df.Age < 5)
    )

    df = df[sel]

    if kepfield:
        l, b = 76.32, 13.5
    else:
        l, b = 76.32, 0

    ls, bs = random_skycoords_within_area(l, b, 10, len(df))
    df['l'] = ls
    df['b'] = bs

    x,y,z = galactic_to_xyz(nparr(ls), nparr(bs), nparr(df['distance_pc']))
    df['galx'] = (x - sun_x_pc) / 1e3
    df['galy'] = y / 1e3
    df['galz'] = z / 1e3

    return df


def random_skycoords_within_area(
    l: float, b: float, area_deg2: float = 10.0, N: int = 1
    ):
    """
    Generate a list of random SkyCoord objects within a specified area around
    given Galactic coordinates.

    Args:
        l (float): Galactic longitude in degrees.

        b (float): Galactic latitude in degrees.

        area_deg2 (float): Area in square degrees around the center coordinates
            to draw from. Default is 10.0.

        N (int): Number of random SkyCoord points to generate. Default is 1.

    Returns:
        tuple of ls and bs lists, length N.
    """
    # Define the central SkyCoord
    c = SkyCoord(l=l * u.deg, b=b * u.deg, frame='galactic')

    # Calculate the radius in degrees for the specified area (10 square degrees)
    radius_deg = np.sqrt(area_deg2 / np.pi)

    np.random.seed(42)
    # Randomly sample l and b offsets within the specified area
    l_offset = np.random.uniform(-radius_deg, radius_deg, size=N)
    b_offset = np.random.uniform(-radius_deg, radius_deg, size=N)
    # Create the new SkyCoord with

    ls = [l + _l_offset for _l_offset in l_offset]
    bs = [b + _b_offset for _b_offset in b_offset]

    return ls, bs


def galactic_to_xyz(l, b, distance):
    # Create a SkyCoord object with the given galactic coordinates and distance
    galactic_coord = SkyCoord(l=l*u.degree, b=b*u.degree,
                              distance=distance*u.pc, frame='galactic')

    # Convert to Galactocentric coordinates
    galactocentric = galactic_coord.transform_to('galactocentric')

    # Extract X, Y, Z coordinates
    x = galactocentric.x.value
    y = galactocentric.y.value
    z = galactocentric.z.value

    return x, y, z


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


