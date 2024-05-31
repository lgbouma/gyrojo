"""
Draw samples from the gyro age posterior PDFs using multithreading.
"""
from gyrointerp.helpers import sample_ages_from_pdf
from os.path import join
import os
import pandas as pd, numpy as np
from numpy import array as nparr
from glob import glob
from gyrointerp.paths import CACHEDIR
from multiprocessing import Pool

postdir = join(CACHEDIR, 'field_gyro_posteriors_20240530')
outdir = join(CACHEDIR, 'bigsamples_field_gyro_posteriors_20240530')
if not os.path.exists(outdir): os.mkdir(outdir)

csvpaths = glob(join(postdir, "*_posterior.csv"))

def process_csvpath(csvpath):
    outcsvname = os.path.basename(csvpath).replace("_posterior", "_bigsamples")
    outcsvpath = join(outdir, outcsvname)
    if os.path.exists(outcsvpath):
        return

    df = pd.read_csv(csvpath)

    age_pdf = nparr(df.age_post)

    if not np.all(np.isfinite(age_pdf)):
        return

    age_samples = sample_ages_from_pdf(nparr(df.age_grid), age_pdf)
    outdf = pd.DataFrame({'t_gyro': age_samples})

    outdf.to_csv(outcsvpath, index=False, float_format='%.1f')
    print(f'Wrote {outcsvpath}')

if __name__ == '__main__':
    with Pool() as pool:
        pool.map(process_csvpath, csvpaths)
