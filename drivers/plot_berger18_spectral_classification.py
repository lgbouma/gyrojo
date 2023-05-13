"""
Plot Teff vs logg, colored by stellar classification, to get an idea
for what cuts might be reasonable to omit subgiants from the
Berger+2020 stellar parameter catalog.  (Note: the *classification*
comes from Berger+2018 based on a solar-metallicity PARSEC isochrone
in the Teff versus radius planet, while the latest and greatest stellar
parameters, like the logg, come from Berger+2020).

Result: logg > 4.3 will omit majority of subgiants.
"""
import numpy as np, matplotlib.pyplot as plt, pandas as pd
import os
from astroquery.vizier import Vizier

from agetools.paths import DATADIR, RESULTSDIR

# pip install aesthetic
from aesthetic.plot import savefig

def prepend_colstr(colstr, df):
    # prepend a string, `colstr`, to all columns in a dataframe
    return df.rename(
        {c:colstr+c for c in df.columns}, axis='columns'
    )

def plot_berger18_berger20_spectral_classification(
    outdir, zorder='default'
):

    Vizier.ROW_LIMIT = -1

    outcsvpath = os.path.join(DATADIR, "interim", "cache",
                              "Berger18_Berger20_inner_xmatch.csv")
    if not os.path.exists(outcsvpath):
        # Berger+2018: radii, luminosities, and spectral classifications
        # based on PARSEC evolutionary models and TAMS definitions (see
        # Section 3.3 of the paper).
        print('downloading berger+18 catalog...')
        catalogs = Vizier.get_catalogs("J/ApJ/866/99")
        df_b18 = catalogs[0].to_pandas()
        print('download successful.')
        df_b18 = prepend_colstr('b18_', df_b18)

        # Berger+2020: Gaia-Kepler stellar properties catalog.
        print('downloading berger+20 catalog...')
        catalogs = Vizier.get_catalogs("J/AJ/159/280")
        df_b20 = catalogs[1].to_pandas() # output parameters
        print('download successful.')
        df_b20 = prepend_colstr('b20_', df_b20)

        df = df_b18.merge(
            df_b20, left_on='b18_KIC', right_on='b20_KIC', how='inner'
        )
        df.to_csv(outcsvpath, index=False)
    else:
        df = pd.read_csv(outcsvpath)

    fig, ax = plt.subplots(figsize=(8,6))

    # iterate over dwarfs, subgiants, giants
    evols = [0,1,2]
    labels = ['B+18 dwarfs', 'B+18 subgiants', 'B+18 giants']
    if zorder == 'default':
        zorders = [0,1,2]
    elif zorder == 'dwarfsontop':
        zorders = [2,1,0]

    for ix, Evol, label, _z in zip(
        range(len(evols)), evols, labels, zorders
    ):

        sel = df.b18_Evol == Evol

        ax.scatter(
            df[sel].b20_Teff, df[sel].b20_logg, s=0.5, c=f'C{ix}',
            label=label, zorder=_z
        )

    ax.update({
        'xlabel': 'B+20 Teff [K]',
        'ylabel': 'B+20 logg',
        'xlim': [7200, 3100],
        'ylim': [3.8, 5],
    })
    ax.legend(loc='best', fontsize='xx-small')

    outpath = os.path.join(
        outdir,
        f'berger18_berger20_spectral_classification_{zorder}.png'
    )
    savefig(fig, outpath)


if __name__ == "__main__":

    outdir = os.path.join(RESULTSDIR, 'berger18_berger20_spectral_classification')
    if not os.path.exists(outdir): os.mkdir(outdir)
    plot_berger18_berger20_spectral_classification(
        outdir, zorder='default'
    )
    plot_berger18_berger20_spectral_classification(
        outdir, zorder='dwarfsontop'
    )
