"""
You've measured the lithium equivalent widths.  Use them, as well as B-V colors
calculated from the adopted_Teff column and assuming the Pecaut+Mamajek
calibration, to estimate the lithium posteriors.
"""

import os, sys, shutil
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from os.path import join
from glob import glob
from numpy import array as nparr

import baffles.baffles as baffles

from gyrojo.paths import DATADIR, RESULTSDIR
from gyrojo.getters import get_li_data

def calc_koi_lithium_posteriors(datestr, sampleid, li_method='baffles',
                                overwrite=1, IRON_OFFSET=7.5):

    sys.path.append('/Users/luke/Dropbox/proj/eagles')
    from eagles import main

    assert li_method in ['baffles', 'eagles']

    outbasedir = join(RESULTSDIR, f"koi_lithium_posteriors_{li_method}_{datestr}")
    if not os.path.exists(outbasedir):
        pass
    else:
        if overwrite:
            # NOTE: necessary bc otherwise updated gyro selxn fn doesnt propagate
            # and can end up with posteriors from earlier runs
            shutil.rmtree(outbasedir)
    os.mkdir(outbasedir)

    mldf = get_li_data(sampleid)

    write_ix = 0
    for ix, r in mldf.iterrows():

        print(79*'-')
        kepoi_name = r['kepoi_name']
        print(f'Beginning {kepoi_name}')

        li_ew = r['Fitted_Li_EW_mA'] - IRON_OFFSET
        li_ew_perr = r['Fitted_Li_EW_mA_perr']
        li_ew_merr = r['Fitted_Li_EW_mA_merr']

        li = 1.*np.round(li_ew, 3)
        li_err = int(np.max([abs(li_ew_perr), abs(li_ew_merr)]))

        outname = join(outbasedir, f"{kepoi_name}")

        if li_method == 'baffles':

            raise DeprecationWarning("Why are you using baffles?")

            # by default, assume we have a detection and that it is gaussian
            bv = r['B-V']
            bv_err = r['B-V_err']
            upperLim = False

            CUTOFF = 10 # mA: consider anything less a nondetection
            if li_ew - li_ew_merr < CUTOFF:
                upperLim = True
                li_err = None
                li = 2* li_ew_perr

            if li < 10 and upperLim:
                li = 10 # hard cut

            maxAge = 4000 # Myr  (for initial "all" analysis)

            print(f"{kepoi_name}: B-V = {bv}+/-{bv_err}, Li = {li}+/-{li_err}, upperLim={upperLim}")
            posterior = baffles.baffles_age(
                bv=bv, bv_err=bv_err, li=li, li_err=li_err, upperLim=upperLim,
                maxAge=maxAge, fileName=outname, pdfPage=None,
                showPlots=False, savePlots=True, savePostAsText=True
            )

        elif li_method == 'eagles':

            error_code = 0

            cachepath = join(outbasedir, "NO_LITHIUM_AGE_CASES.txt")
            if write_ix == 0:
                if os.path.exists(cachepath):
                    os.remove(cachepath)
                    print(f"Cleared {cachepath} failure case cache for eagles...")
                with open(cachepath, 'w') as f:
                    f.writelines(f"kepoi_name,error_code\n")

            if r['adopted_Teff'] < 3000 or r['adopted_Teff'] > 6500:
                error_code = 1

            if error_code >= 1:
                with open(cachepath, 'a') as f:
                    f.writelines(f"{kepoi_name},{error_code}\n")
                print(f"{kepoi_name} got code {error_code}; continue.")
                write_ix += 1
                continue

            # Otherwise, calculate eagles posterior...
            line = (
                f"{kepoi_name} "
                f"{r['adopted_Teff']} "
                f"{r['adopted_Teff_err']} "
                f"{li} "
                f"{li_err}"
            )
            input_path = join(outbasedir, f"{kepoi_name}.indat")
            with open(input_path, 'w') as f:
                f.writelines(line)

            output_path = join(outbasedir, f"{kepoi_name}")

            # Read data for a single star and estimate its age, (input_file
            # would have one row), based on a prior age probability that is
            # flat in linear age, saving the output plots
            args = [input_path, output_path,
                    '-s', '-p', '1', "--lagesmax", "10"]
            plt.close("all")
            main(args)
            plt.close("all")

        print(f"{kepoi_name}: done")

if __name__ == "__main__":

    # (deprecated)
    datestr = "20230208"
    sampleid = "all"
    li_method = 'baffles'

    # plausible alternative...  (upper Li limits treated but poor calibrators)
    datestr = "20240405"
    sampleid = "koi_X_S19S21dquality"
    li_method = 'baffles'

    # default?  (upper Li limits problematic)
    datestr = "20240405"
    sampleid = "koi_X_S19S21dquality"
    li_method = 'eagles'

    # default?  (upper Li limits problematic)
    datestr = "20240405"
    sampleid = "koi_X_JUMP"
    li_method = 'eagles'

    calc_koi_lithium_posteriors(datestr, sampleid, li_method=li_method)
