"""
Given gyro and lithium posteriors, multiply them.
"""
import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from os.path import join
from glob import glob
from numpy import array as nparr
from agetools.paths import LOCALDIR, DATADIR, RESULTSDIR
from copy import deepcopy

from scipy.interpolate import interp1d

from agetools.getters import get_li_data, get_gyro_data

from gyrointerp.helpers import get_summary_statistics
from aesthetic.plot import set_style, savefig

def calc_koi_joint_posteriors(sampleid, overwrite=0):

    # gyro data: 862 KOIs if "all" (past step0, +agreeing periods).
    # 73 if cutting down to the "sel-2s" sample.
    kdf = get_gyro_data(sampleid)

    # lithium data: for the `sel_2s` sample, 68 KOIs with finite Li values
    # (meaning: spectra exist, and calc_koi_lithium_posteriors yielded
    # posteriors for them, potentially based on non-detections of the line).
    li_df = get_li_data(sampleid)
    li_df = li_df[~pd.isnull(li_df.Fitted_Li_EW_mA)]

    # The "joint posteriors" are the intersection of the lithium and
    # gyro samples.  (Note that if only lithium, or only gyro is available,
    # then those should be adopted by themselves!)
    sel = li_df.kepoi_name.isin(kdf.kepoi_name)
    li_df = li_df[sel]

    for ix, r in kdf.iterrows():

        kepoi_name = r['kepoi_name']

        # load the gyro posterior
        gyrocachedir = join(LOCALDIR, "gyrointerp",
                            "koi_gyro_posteriors_20230208")
        gyropath = glob(join(gyrocachedir, f"{kepoi_name}*posterior.csv"))
        assert len(gyropath) == 1
        gyropath = gyropath[0]
        # keys: age_grid, age_post
        gdf = pd.read_csv(gyropath)

        # load the lithium posterior
        licachedir = join(RESULTSDIR, "koi_lithium_posteriors_20230208")
        lipath = join(licachedir, f"{kepoi_name}_lithium.csv")
        if os.path.exists(lipath):
            # keys: age_grid, age_post
            ldf = pd.read_csv(lipath, names=['age_grid','age_post'])
            HIRES_data_exist = 1
        else:
            HIRES_data_exist = 0

        outdir = join(RESULTSDIR, "koi_gyro_X_lithium_posteriors_20230208")
        if not os.path.exists(outdir): os.mkdir(outdir)
        outpath = join(outdir, f"{kepoi_name}_gyroXlithium_posterior.csv")

        age_grid = nparr(gdf.age_grid)
        gyro_post = nparr(gdf.age_post)

        if not HIRES_data_exist:
            gdf['HIRES_data_exist'] = HIRES_data_exist
            gdf.to_csv(outpath, index=False)

            # cache output values
            d = get_summary_statistics(age_grid, gyro_post)
            _outpath = join(outdir, f"{kepoi_name}_gyroXlithium_summary.csv")
            summary_df = pd.DataFrame(d, index=[0])
            summary_df['HIRES_data_exist'] = HIRES_data_exist
            summary_df.to_csv(_outpath, index=False)
            med = str(int(summary_df['median']))
            p1sig, m1sig = str(int(summary_df['+1sigma'])), str(int(summary_df['-1sigma']))

            # make summary plot
            _outpath = join(outdir, f"{kepoi_name}_gyroXlithium.png")
            if not os.path.exists(_outpath) and not overwrite:
                plt.close("all")
                set_style("clean")
                fig, ax = plt.subplots()
                ax.plot(age_grid, gyro_post, lw=1, c='k', ls=':', label='Rotation')
                ax.update({
                    'xlabel': 'Age [Myr]',
                    'ylabel': 'Probability ($10^{-3}\,$Myr$^{-1}$)',
                    'xlim': [0, 2000],
                })
                ax.legend(loc='best', fontsize='x-small', handletextpad=0.2,
                          borderaxespad=1., borderpad=0.5, fancybox=True,
                          framealpha=0.8, frameon=False)
                ax.set_title(
                    f'{kepoi_name} $'+med+'^{+'+p1sig+'}_{-'+m1sig+'}$ Myr (no HIRES)',
                    pad=-4, fontsize='small'
                )
                savefig(fig, _outpath, dpi=400, writepdf=1)

                print(f"Wrote {outpath} +plot, +cache, continue.")
            continue

        #
        # interpolate to match the same grid between the lithium and gyro
        # posteriors.  finally, compute the product.
        #
        _x, _y = nparr(ldf.age_grid), nparr(ldf.age_post)
        fn = interp1d(_x, _y, kind='linear', fill_value='extrapolate')
        li_post = fn(age_grid)
        li_post /= np.trapz(li_post, age_grid)

        assert np.isclose(np.trapz(gyro_post, age_grid), 1)
        assert np.isclose(np.trapz(li_post, age_grid), 1)

        joint_post = gyro_post * li_post

        joint_post /= np.trapz(joint_post, age_grid)

        # cache the output for both the joint and the lithium posteriors
        out_df = pd.DataFrame({'age_grid': age_grid, 'age_post': joint_post})
        out_df['HIRES_data_exist'] = HIRES_data_exist
        out_df.to_csv(outpath, index=False)
        print(f"Wrote {outpath}")

        # cache joint output values, and the lithium values alone
        d = get_summary_statistics(age_grid, joint_post)
        _outpath = join(outdir, f"{kepoi_name}_gyroXlithium_summary.csv")
        summary_df = pd.DataFrame(d, index=[0])
        summary_df['HIRES_data_exist'] = HIRES_data_exist
        summary_df.to_csv(_outpath, index=False)

        li_d = get_summary_statistics(age_grid, li_post)
        _outpath = join(outdir, f"{kepoi_name}_lithium_summary.csv")
        _summary_df = pd.DataFrame(li_d, index=[0])
        _summary_df['HIRES_data_exist'] = HIRES_data_exist
        _summary_df.to_csv(_outpath, index=False)

        # make summary plot
        _outpath = join(outdir, f"{kepoi_name}_gyroXlithium.png")
        if not os.path.exists(_outpath) and not overwrite:
            plt.close("all")
            set_style("clean")
            fig, ax = plt.subplots()
            ax.plot(age_grid, gyro_post, lw=1, c='C0', ls=':', label='Rotation',
                   zorder=3)
            ax.plot(age_grid, li_post, lw=1, c='C1', ls='--', label='Lithium',
                    zorder=2)
            ax.plot(age_grid, joint_post, lw=2, c='k', ls='-', label='Both',
                   zorder=1)
            ax.update({
                'xlabel': 'Age [Myr]',
                'ylabel': 'Probability ($10^{-3}\,$Myr$^{-1}$)',
                'xlim': [0, 2000],
            })
            ax.legend(loc='best', fontsize='x-small', handletextpad=0.2,
                      borderaxespad=1., borderpad=0.5, fancybox=True,
                      framealpha=0.8, frameon=False)

            if not np.all(pd.isnull(summary_df['median'])):
                med = str(int(summary_df['median']))
                p1sig, m1sig = str(int(summary_df['+1sigma'])), str(int(summary_df['-1sigma']))

                ax.set_title(
                    f'{kepoi_name} $'+med+'^{+'+p1sig+'}_{-'+m1sig+'}$ Myr',
                    pad=-4, fontsize='small'
                )
            else:
                ax.set_title(
                    f'{kepoi_name} NaN age',
                    pad=-4, fontsize='small'
                )

            savefig(fig, _outpath, dpi=400, writepdf=1)

    #
    # write the joint posteriors (and lithium, and gyro) to a summarized
    # dataframe.
    #
    outdf = deepcopy(kdf)
    cols = ['median', 'peak', 'mean', '+1sigma', '-1sigma', '+2sigma',
            '-2sigma', '+3sigma', '-3sigma', '+1sigmapct', '-1sigmapct']
    renamedict = {}
    for c in cols:
        renamedict[c] = 'gyro_'+c

    outdf = outdf.rename(renamedict, axis='columns')

    # incorporate the joint summary statistics...
    joint_df = pd.concat(
        [pd.read_csv(join(outdir, f"{kepoi_name}_gyroXlithium_summary.csv"))
         for kepoi_name in kdf.kepoi_name]
    )
    joint_df['kepoi_name'] = nparr(kdf.kepoi_name)
    for c in cols:
        renamedict[c] = 'joint_'+c

    joint_df = joint_df.rename(renamedict, axis='columns')
    outdf = outdf.merge(joint_df, how='left', on='kepoi_name')

    # incorporate the lithium summary statistics.  trickier, because
    # non-existent summary files means that HIRES data did not exist.
    li_dfs = []
    _kepoi_names = []
    for kepoi_name in kdf.kepoi_name:
        li_summ = join(outdir, f"{kepoi_name}_lithium_summary.csv")
        if not os.path.exists(li_summ):
            pass
        else:
            li_dfs.append(pd.read_csv(li_summ))
            _kepoi_names.append(kepoi_name)
    li_df = pd.concat(li_dfs)
    li_df['kepoi_name'] = _kepoi_names

    for c in cols:
        renamedict[c] = 'li_'+c

    li_df = li_df.rename(renamedict, axis='columns')

    outdf = outdf.merge(li_df, how='left', on='kepoi_name')

    selcols = ['kepler_name', 'kepoi_name', 'gyro_median', 'gyro_+1sigma',
               'gyro_-1sigma', 'li_median', 'li_+1sigma', 'li_-1sigma',
               'joint_median', 'joint_+1sigma', 'joint_-1sigma']
    pd.options.display.max_rows = 5000
    print(outdf[selcols].sort_values(by='joint_median'))

    outpath = join(outdir, f"{sampleid}_merged_joint_age_posteriors.csv")
    outdf.to_csv(outpath, index=False)
    print(f"Wrote {outpath}")


if __name__ == "__main__":
    calc_koi_joint_posteriors("all")
