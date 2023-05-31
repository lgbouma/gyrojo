"""
For field stars within a given sampleid (e.g., "Santos19_Santos21_all"),
calculate the gyro posteriors.
"""

import numpy as np, matplotlib.pyplot as plt, pandas as pd
import os, pickle
from os.path import join
from numpy import array as nparr

from gyrojo.paths import DATADIR, LOCALDIR, RESULTSDIR
from gyrojo.getters import (
    get_kicstar_data
)


from gyrointerp.helpers import prepend_colstr, left_merge
from gyrointerp.gyro_posterior import gyro_age_posterior_list

def calc_field_gyro_posteriors():

    sampleid = 'Santos19_Santos21_all'

    df = get_kicstar_data(sampleid)

    # Run gyro analysis for all of them -- flags for which stars it is actually
    # _valid_ for will follow afterward.  This means stars outside the
    # 3800-6200K range, subgiants, and known binaries are INCLUDED at this
    # stage.
    Teffs = nparr(df.adopted_Teff)
    Teff_errs = nparr(df.adopted_Teff_err)
    Prots = np.round(nparr(df.Prot), 4)
    AS_REPORTED = 0
    if AS_REPORTED:
        # as reported; probably underestimated?
        Prot_errs = nparr(df.E_Prot)
    else:
        # empirical uncertainty estimate
        Prot_errs = nparr(df.Prot_err)

    object_ids = nparr(df.KIC).astype(str)

    PLOT_INTERIM = 1
    if PLOT_INTERIM:
        # visualize adopted prot vs teffs
        outdir = join(RESULTSDIR, "field_mean_prot_teff")
        if not os.path.exists(outdir): os.mkdir(outdir)
        outpath = join(outdir, f"{sampleid}_prot_teff.png")
        plt.close("all")
        fig, ax = plt.subplots()
        ax.errorbar(
            Teffs, Prots, xerr=Teff_errs, yerr=Prot_errs,
            marker='o', elinewidth=0.5, capsize=0, lw=0, mew=0.5, color='k',
            markersize=1, zorder=5
        )
        ax.update({
            'xlabel': 'Adopted Teff [K; mostly B+20]',
            'ylabel': 'Prot [day]',
            'title': (
                f'{sampleid} Prot vs Teff, no cleaning'
            ),
            'xlim': [6500, 3500],
            'ylim': [-1, 45]
        })
        fig.savefig(outpath, bbox_inches='tight', dpi=400)

        ax.set_ylim([-1, 25])

        outpath = join(outdir, "step0_mean_prot_adoptedteff_zoomylim.png")
        fig.savefig(outpath, bbox_inches='tight', dpi=400)

    DO_CALCULATION = 1

    if DO_CALCULATION:
        cache_id = "field_gyro_posteriors_20230529"
        gyro_age_posterior_list(
            cache_id, Prots, Teffs, Prot_errs=Prot_errs, Teff_errs=Teff_errs,
            star_ids=object_ids, age_grid=np.linspace(0, 4000, 500),
            N_grid='default', bounds_error='4gyrlimit',
            interp_method='pchip_m67'
        )


if __name__ == "__main__":
    calc_field_gyro_posteriors()
