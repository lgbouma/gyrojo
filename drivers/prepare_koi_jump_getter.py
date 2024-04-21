"""
Crossmatch the gyro KOI sample against JUMP.  Needs access to the JUMP
"explore" SQL interface to run.  Cleans the initial SQL query results, and
constucts the list of necessary spectra, and makes an scp script that can be
used to pull the deblazed spectra from shrek.
"""

import numpy as np, matplotlib.pyplot as plt, pandas as pd
import os, pickle
from os.path import join

from gyrojo.paths import DATADIR, LOCALDIR, RESULTSDIR
from gyrojo.getters import get_gyro_data, get_koi_data

def prepare_koi_jump_getter(sampleid):

    print(20*'-')
    print(f"Preparing sampleid={sampleid}")

    # this CSV is output by `kepler_lithium_sample_getter.sql` already
    # sorted by name and counts.  there is no requirement on whether
    # iodine is in or out b/c it doesn't affect lithium.
    csvpath = join(
        DATADIR, "raw",
        "20240407_JUMP_kepler_lithium_sample_no_iodine_constraint.csv"
    )

    # initially grab only the highest count spectra.
    __jdf = pd.read_csv(csvpath)
    jdf = __jdf.drop_duplicates(subset='name', keep='first')

    # get all 9564 KOIs, including false positives...
    kdf = get_koi_data('cumulative-KOI', grazing_is_ok=1)
    assert len(kdf) == 9564
    # run the lithium analysis for any non-FP KOI with MES>10 for
    # which a HIRES spectrum exists
    kdf = kdf[kdf.flag_is_ok_planetcand]
    kdf = kdf.sort_values(by='kepoi_name')
    assert len(kdf) == 3307

    # search the KOI table for matches based on both kepid and
    # kepoi_name (the KOI identifier)
    matchrows = []
    ix = 0
    verbose = 0
    for _, r in kdf.iterrows():

        kepid = r['kepid']
        kepoi_name = r['kepoi_name']
        abbrev_kepoi_name = kepoi_name.split(".")[0]
        if verbose:
            print(ix, kepid, kepoi_name)

        # Check first for match based on Kepler identifier number.
        # If found, done.
        _sel = jdf.name.str.contains(str(kepid))
        if np.any(_sel):
            N = np.sum(_sel)
            assert N == 1
            _matchdf = jdf[_sel].reset_index(drop=True)
            matchname = _matchdf.name.iloc[0]
            if verbose:
                print(f'\t...got {N} Kepler ID match, {kepid} = {matchname}')
            _matchdf['kepid'] = kepid
            _matchdf['kepoi_name'] = kepoi_name
            matchrows.append(_matchdf)
            ix += 1
            continue

        # Check for match based on KOI identifier number
        # If found, done.
        _sel = jdf.name.str.contains(str(abbrev_kepoi_name))
        if np.any(_sel):
            N = np.sum(_sel)
            if N == 1:
                _matchdf = jdf[_sel].reset_index(drop=True)
                matchname = _matchdf.name.iloc[0]
                if verbose:
                    print(f'\t...got {N} KOI ID match, {abbrev_kepoi_name} = {matchname}')
            else:
                __matchdf = jdf[_sel].sort_values(
                    by='counts',
                    ascending=False
                ).reset_index(drop=True)
                _matchdf = __matchdf.head(n=1)
                printcols = ['name','ra','dec','utctime','observation_id','counts']
                if verbose:
                    print(f'\t...got {N} KOI ID matches, {abbrev_kepoi_name} =\n{__matchdf[printcols]}')
                    print(f'\t...taking first!')

            _matchdf['kepid'] = kepid
            _matchdf['kepoi_name'] = kepoi_name
            matchrows.append(_matchdf)
            ix += 1
            continue

        if verbose:
            print('\t... got no matches')
        ix += 1

    _jdf = pd.concat(matchrows)
    N_lithiumstars = len(np.unique(_jdf.kepid))
    N_lithiumplanets = len(_jdf)
    print(f"N_lithiumstars: {N_lithiumstars}")
    print(f"N_lithiumplanets: {N_lithiumplanets}")

    N_HIRES_hours = int(np.round(
        _jdf[~_jdf.duplicated('observation_id')].exposure_time.sum() /
        3600, 0
    ))

    from gyrojo.papertools import update_latex_key_value_pair as ulkvp
    ulkvp('nlithiumstars', N_lithiumstars)
    ulkvp('nhireshours', N_HIRES_hours)
    ulkvp('nlithiumplanets', N_lithiumplanets)

    mjdf = _jdf.merge(kdf, how='left', on='kepoi_name', suffixes=("_JUMP",""))
    assert len(mjdf) == len(_jdf)

    outpath = join(DATADIR, "interim", f"koi_jump_getter_{sampleid}.csv")
    mjdf.to_csv(outpath, index=False)
    print(f"Wrote {outpath}")

    outdir = join(LOCALDIR, "gyrojo_HIRES_lithium")
    if not os.path.exists(outdir): os.mkdir(outdir)

    # make the scp script to pull all the lithium data available for the entire
    # "step0" gyro sample.
    lines = []
    for fname in mjdf.filename:
        l = f"scp luke@cadence:{fname} {outdir}/. \n"
        lines.append(l)

    if sampleid == 'koi_X_JUMP':
        bash_script = "scp_HIRES_lithium_data.sh"
        with open(bash_script, "w") as f:
            f.writelines(lines)

        print(f"Wrote {bash_script}; now you must execute it.")


if __name__ == "__main__":
    prepare_koi_jump_getter('koi_X_JUMP')
    #prepare_koi_jump_getter('koi_X_S19S21dquality')
    #prepare_koi_jump_getter('deprecated_all')
    #prepare_koi_jump_getter('deprecated_sel_2s')
