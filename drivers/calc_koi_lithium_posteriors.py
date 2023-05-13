"""
You've measured the lithium equivalent widths.  Use them, as well as B-V colors
calculated from the adopted_Teff column and assuming the Pecaut+Mamajek
calibration, to estimate the lithium posteriors.
"""

import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from os.path import join
from glob import glob
from numpy import array as nparr

import baffles.baffles as baffles

from agetools.paths import DATADIR, RESULTSDIR
from agetools.getters import get_li_data

def calc_koi_lithium_posteriors():

    datestr = "20230208"
    sampleid = "all"

    mldf = get_li_data(sampleid)

    for ix, r in mldf.iterrows():

        kepoi_name = r['kepoi_name']
        bv = r['B-V']
        bv_err = r['B-V_err']

        IRON_OFFSET = 10 #TODO FIXME CALIBRATE
        li_ew = r['Fitted_Li_EW_mA'] - IRON_OFFSET
        li_ew_perr = r['Fitted_Li_EW_mA_perr']
        li_ew_merr = r['Fitted_Li_EW_mA_merr']

        # by default, assume we have a detection and that it is gaussian
        upperLim = False
        li = 1.*li_ew
        li_err = int(np.mean([abs(li_ew_perr), abs(li_ew_merr)]))

        CUTOFF = 10 # mA: consider anything less a nondetection
        if li_ew - li_ew_merr < CUTOFF:
            upperLim = True
            li_err = None
            li = 2* li_ew_perr

        if li < 10 and upperLim:
            li = 10 # hard cut

        maxAge = 4000 # Myr  (for initial "all" analysis)

        outbasedir = join(RESULTSDIR, f"koi_lithium_posteriors_{datestr}")
        if not os.path.exists(outbasedir): os.mkdir(outbasedir)

        outname = join(outbasedir, f"{kepoi_name}")

        print(f"{kepoi_name}: B-V = {bv}+/-{bv_err}, Li = {li}+/-{li_err}, upperLim={upperLim}")
        posterior = baffles.baffles_age(
            bv=bv, bv_err=bv_err, li=li, li_err=li_err, upperLim=upperLim,
            maxAge=maxAge, fileName=outname, pdfPage=None,
            showPlots=False, savePlots=True, savePostAsText=True
        )
        print(f"{kepoi_name}: done")

if __name__ == "__main__":
    calc_koi_lithium_posteriors()
