import os
from os.path import join
import numpy as np, pandas as pd
from numpy import array as nparr
from astropy import units as u, constants as c
from gyrojo.getters import (
    get_gyro_data, get_age_results, get_koi_data
)
from gyrojo.paths import TABLEDIR
from gyrojo.papertools import update_latex_key_value_pair as ulkvp
from scipy.stats import gamma

def completeness_correction(snr):

    k = 17.56  # shape parameter
    theta = 0.49  # scale parameter (θ)
    l = 1  # location parameter is not directly used in gammacdf but can adjust the input

    return gamma.cdf(snr, k, scale=theta, loc=l)

def get_star_and_planet_dataframes():
    # stars (with rotation, & gyro applicable)
    _df = get_gyro_data("Santos19_Santos21_dquality", drop_grazing=1, drop_highruwe=1)
    sdf = _df[_df.flag_is_gyro_applicable]

    # planets
    df, _, _ = get_age_results(whichtype='gyro', drop_grazing=1)

    df.loc[df.KIC == '7335514', 'koi_smass'] = 0.879 # KOI-7368 missing stellar mass and sma
    df.loc[df.KIC == '7335514', 'koi_srad'] = 0.874 # KOI-7368 missing stellar radius
    sma_AU = ((
        c.G * nparr(df.koi_smass)*u.Msun / (4*np.pi**2) *
        (nparr(df.adopted_period) * u.day)**2
    )**(1/3)).to(u.AU).value
    resid = df['koi_sma'] - sma_AU
    df.loc[df.koi_sma.isnull(), 'koi_sma'] = sma_AU[df.koi_sma.isnull()]

    # calculate geometric weights
    df['w_geom'] = (1 / 0.8 ) * (
        (nparr(df['koi_sma'])*u.AU)
        /
        (nparr(df['koi_srad'])*u.Rsun)
    ).cgs.value

    # calculate completeness correction
    # TODO: warning: this is a sketchy hack.  the MES != the SNR as defined by
    # Fulton et al... but for a zeroth order pass, this is OK.
    snr = nparr(df.koi_max_mult_ev)
    df['w_det'] = (
        1/completeness_correction(snr)
    )

    return sdf, df

def calc_simple_bin_occ():

    sdf, df = get_star_and_planet_dataframes()

    # define bins
    age_binedges = [0,1000,2000,3000]
    rp_binedges = [0,1.8,4]
    period_binedges = [0.1,10,100]
    #rp_binedges = [0,1.8,6]
    #period_binedges = [0.01,10,1000]

    age_bins = [
        (lo, hi) for lo,hi in zip(age_binedges[:-1], age_binedges[1:])
    ]
    rp_bins = [
        (lo, hi) for lo,hi in zip(rp_binedges[:-1], rp_binedges[1:])
    ]
    period_bins = [
        (lo, hi) for lo,hi in zip(period_binedges[:-1], period_binedges[1:])
    ]

    countdict = {}
    occdict = {}

    for alo,ahi in age_bins:

        sel_st = (sdf['median'] > alo) & (sdf['median'] <= ahi)
        N_st = len(sdf[sel_st])

        agekey = f"{alo}to{ahi}Myr"
        countdict[agekey] = {}
        countdict[agekey]['Nst'] = N_st
        occdict[agekey] = {}

        N_pl_tot = 0
        for rlo,rhi in rp_bins:
            for plo,phi in period_bins:

                sel_pl = (
                    (df['gyro_median'] >= alo) & (df['gyro_median'] < ahi)
                    &
                    (df['adopted_rp'] >= rlo) & (df['adopted_rp'] < rhi)
                    &
                    (df['adopted_period'] >= plo) & (df['adopted_period'] < phi)
                )
                N_pl = len(df[sel_pl])
                N_pl_tot += N_pl

                binkey = f'Npl_Rp{rlo}to{rhi}_P{plo}to{phi}'
                countdict[agekey][binkey] = N_pl

                NPPS = N_pl / N_st

                wNPPS = np.sum(df[sel_pl].w_geom * df[sel_pl].w_det) / N_st
                binkey = f'wNPPS_Rp{rlo}to{rhi}_P{plo}to{phi}'
                occdict[agekey][binkey] = np.round(wNPPS, 2)

                abs_unc = wNPPS * np.sqrt(N_pl) / N_pl
                binkey = f'auncwNPPS_Rp{rlo}to{rhi}_P{plo}to{phi}'
                occdict[agekey][binkey] = np.round(abs_unc, 2)

        countdict[agekey]['Npl'] = N_pl_tot

    countdf = pd.DataFrame(countdict)
    occdf = pd.DataFrame(occdict)
    print(42*'-')
    print(countdf)
    print(42*'-')
    print(occdf)
    print(42*'-')

    outcsv = join(TABLEDIR, 'simple_bin_counts.csv')
    countdf.to_csv(outcsv, index=False)
    print(f"Wrote {outcsv}")

    outcsv = join(TABLEDIR, 'simple_bin_occurrence.csv')
    occdf.to_csv(outcsv, index=False)
    print(f"Wrote {outcsv}")


def main():
    calc_simple_bin_occ()


if __name__ == "__main__":
    main()
