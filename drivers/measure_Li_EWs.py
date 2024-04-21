"""
Driver script used to measure Li EWs on HIRES spectra.

By default, this is done on every star for which a spectrum was acquired off
JUMP -- i.e., the "koi_jump_getter_all.csv" sample, which was output by
"prepare_koi_jump_getter".  This means the 864 KOIs that pass "step0" and that
have consistent rotation period measurements.
"""
import os
from copy import deepcopy
from glob import glob
from gyrojo.paths import DATADIR, LOCALDIR
from os.path import join
from datetime import datetime
import pandas as pd, numpy as np

# https://github.com/lgbouma/cdips_followup
# cdips_followup has specmatch-emp as a dependency
from cdips_followup.spectools import get_Li_6708_EW

# metadata dataframe, contains the JUMP, Gaia, B20, S19/21, etc information
sampleid = "koi_X_S19S21dquality"
sampleid = "koi_X_JUMP"
csvpath = join(DATADIR, "interim", f"koi_jump_getter_{sampleid}.csv")
df = pd.read_csv(csvpath)

for ix, r in df.iterrows():

    # eg., CK06186
    idstring = r['name']

    if idstring == 'CK03377B':
        continue
        # odd edge case; the "primary" was observed for 730sec, this one is
        # only 60sec and causes the fitter to fail.  (and i think it's the same
        # star)

    filename = os.path.basename(r['filename'])
    spectrum_path = join(LOCALDIR, "gyrojo_HIRES_lithium", filename)
    assert os.path.exists(spectrum_path)

    # e.g., ij489.70.fits
    specname = deepcopy(filename)

    print(42*'-')
    print(f"{datetime.utcnow().isoformat()}: Start {idstring} {specname}.")

    outbasedir = os.path.join(DATADIR, 'interim', f'Li_EW_HIRES_{sampleid}')
    outdir = os.path.join(DATADIR, 'interim', f'Li_EW_HIRES_{sampleid}', idstring)

    for d in [outbasedir, outdir]:
        if not os.path.exists(d): os.mkdir(d)

    delta_wavs = [7.5]
    for delta_wav in delta_wavs:
        outname = (
            f"{idstring}_{specname.replace('.fits','')}_"
            f"Li_EW_deltawav{delta_wav:.1f}_xshift-find.png"
        )
        outpath = os.path.join(outdir, outname)
        if os.path.exists(outpath):
            print(f"Found {outpath}")
        else:
            get_Li_6708_EW(spectrum_path, wvsol_path=None, delta_wav=delta_wav,
                           outpath=outpath, xshift='find')

    print(f"{datetime.utcnow().isoformat()}: Finished {idstring} {specname}.")
