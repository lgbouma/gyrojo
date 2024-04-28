"""
For KOIs passing "step 0" from `build_koi_table.py`, calculate the gyro
posteriors using gyro-interp.

"Step 0" means:
finite teff and logg, 3800-6200K, logg>4, not FP, not grazing, finite MES,
MES>10, period not flagged by Mazeh+15.
"""

import numpy as np, matplotlib.pyplot as plt, pandas as pd
import os, pickle
from os.path import join
from numpy import array as nparr

from gyrojo.paths import DATADIR, LOCALDIR, RESULTSDIR

from gyrointerp.helpers import prepend_colstr, left_merge
from gyrointerp.gyro_posterior import gyro_age_posterior_list

def calc_koi_gyro_posteriors():

    csvpath = os.path.join(DATADIR, "interim",
                           "koi_table_X_GDR3_B20_S19_S21_M14_M15.csv")
    df = pd.read_csv(csvpath)

    # Step0: 3800-6200K, logg>4, not FP, not grazing, finite MES, MES>10,
    # period not flagged by Mazeh+15.  869 KOIs meet this designation.
    sdf = df[df.flag_selected_step0]

    # 864 of these have periods that "agree" (meaning either only one period
    # was provided, or if multiple periods were provided then they agree to
    # within 10% of the mean period.)
    #
    # The missing 5 were:
	#      kepoi_name  s19_Prot  s21_Prot  m14_Prot  m15_Prot
	# 2329  K02703.01     56.37       NaN       NaN     27.15
	# 2482  K02803.01       NaN      8.37       NaN     16.84
	# 2509  K02883.01     33.52       NaN       NaN     40.77
	# 2547  K02821.01       NaN     15.53       NaN     31.12
	# 2956  K02996.01       NaN     23.54       NaN     28.51
    # 
    # By inspecting the DVR reports at NASA's exoplanet archive:
    # 
    # KOI-2703: is 27 days.  (doesn't matter tho, based on gyrointerp range)
    # KOI-2803: I cannot tell, either 8 or 16 looks fine.
    # KOI-2883: I think this disagreement is just bc it's in the 33-41 range.
    # KOI-2821: I think I prefer 30 days, but not obvious.
    # KOI-2996: similar to KOI-2883.
    #
    # So... manual inspection yields nothing.  Just omit them.

    sel = sdf.flag_reported_periods_agree
    ssdf = sdf[sel]

    Teffs = nparr(ssdf.adopted_Teff)
    Teff_errs = nparr(ssdf.adopted_Teff_err)
    Prots = np.round(nparr(ssdf.mean_period), 4)

    # To estimate the period uncertainties... for cases with matching periods,
    # by how much do they differ?
    period_cols = ['s19_Prot', 's21_Prot', 'm14_Prot', 'm15_Prot']
    _mean_periods, _dperiods = [], []
    for ix, r in ssdf.iterrows():
        mean_period = r['mean_period']
        for period_col in period_cols:
            dperiod = r[period_col] - mean_period
            if ~pd.isnull(dperiod):
                _mean_periods.append(mean_period)
                _dperiods.append(np.abs(dperiod))

    # Based on the plot (below), adopt 1% for <15 days, 2% out to 20 days, 4%
    # 20-30 days, and 5% at >30 days.
    Prot_errs = np.ones(len(Prots))
    Prot_errs[Prots<=15] = 0.01*Prots[Prots<=15]
    Prot_errs[(Prots>15) & (Prots<=20)] = 0.02*Prots[(Prots>15) & (Prots<=20)]
    Prot_errs[(Prots>20) & (Prots<=25)] = 0.03*Prots[(Prots>20) & (Prots<=25)]
    Prot_errs[(Prots>25) & (Prots<=30)] = 0.04*Prots[(Prots>25) & (Prots<=30)]
    Prot_errs[Prots>30] = 0.05*Prots[Prots>30]

    object_ids = nparr(ssdf.kepoi_name).astype(str)

    PLOT_INTERIM = 1
    if PLOT_INTERIM:
        # plot1: estimate period uncertainties
        outdir = join(RESULTSDIR, "koi_period_uncertainty_estimate")
        if not os.path.exists(outdir): os.mkdir(outdir)
        outpath = join(outdir, "perioddiff_vs_period.png")
        plt.close("all")
        fig, ax = plt.subplots()
        ax.scatter(_mean_periods, _dperiods, s=2, c='k')
        _period = np.linspace(0.1,45,1000)
        lss = [':','--','-.','-',':']
        for pct, ls in zip(np.arange(0.01, 0.06, 0.01), lss):
            ax.plot(_period, pct*_period, zorder=-1, lw=0.5, ls=ls,
                    label=f'{int(100*pct)}%')
        ax.legend(loc='upper left', fontsize='small')
        ax.update({
            'xlabel': '<Prot> [day]',
            'ylabel': '|P-<Prot>| [day]',
            'xlim': [-1, 45],
            'title': "period differences in S19/S21/M14/M15 vs. each other"
        })

        # plot2: to visualize adopted prot vs teffs
        outdir = join(RESULTSDIR, "koi_mean_prot_teff")
        if not os.path.exists(outdir): os.mkdir(outdir)
        outpath = join(outdir, "step0_mean_prot_adoptedteff.png")
        plt.close("all")
        fig, ax = plt.subplots()
        ax.errorbar(
            Teffs, Prots, xerr=Teff_errs, yerr=Prot_errs,
            marker='o', elinewidth=0.5, capsize=0, lw=0, mew=0.5, color='k',
            markersize=3, zorder=5
        )
        ax.update({
            'xlabel': 'Adopted Teff [K; mostly B+20]',
            'ylabel': '<Prot> [day]',
            'title': (
                'step0: finite teff and logg, 3800-6200K, logg>4, not FP,\n '
                'not grazing, finite MES, MES>10, no M15 flag'
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
        cache_id = "koi_gyro_posteriors_20230208"
        gyro_age_posterior_list(
            cache_id, Prots, Teffs, Prot_errs=Prot_errs, Teff_errs=Teff_errs,
            star_ids=object_ids, age_grid=np.linspace(0, 4000, 500),
            N_grid='default', bounds_error='4gyrlimit',
            interp_method='pchip_m67'
        )


if __name__ == "__main__":
    calc_koi_gyro_posteriors()
