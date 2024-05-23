"""
Draw samples from the gyro age posterior PDFs.
"""
from gyrointerp.helpers import sample_ages_from_pdf
from os.path import join
import os
import pandas as pd, numpy as np
from numpy import array as nparr
from glob import glob
from gyrointerp.paths import CACHEDIR

postdir = join(CACHEDIR, 'field_gyro_posteriors_20240521')
outdir = join(CACHEDIR, 'bigsamples_field_gyro_posteriors_20240521')
if not os.path.exists(outdir): os.mkdir(outdir)

csvpaths = glob(join(postdir, "*_posterior.csv"))

for ix, csvpath in enumerate(csvpaths):

    outcsvname = os.path.basename(csvpath).replace("_posterior", "_bigsamples")
    outcsvpath = join(outdir, outcsvname)
    if os.path.exists(outcsvpath):
        continue

    df = pd.read_csv(csvpath)

    age_pdf = nparr(df.age_post)

    if not np.all(np.isfinite(age_pdf)):
        #print(f'{ix}/{len(csvpaths)}:  Found nans; skip.')
        continue

    age_samples = sample_ages_from_pdf(nparr(df.age_grid), age_pdf)
    outdf = pd.DataFrame({'t_gyro': age_samples})

    outdf.to_csv(outcsvpath, index=False, float_format='%.1f')
    if ix % 100 == 0:
        print(f'{ix}/{len(csvpaths)}:  Wrote {outcsvpath}')
