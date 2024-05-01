import os
import gyrojo.plotting as ap
from gyrojo.paths import RESULTSDIR

# calculated in calc_field_gyro_posteriors.py
cache_id = "hist_field_gyro_ages_20240430"

outdir = os.path.join(RESULTSDIR, cache_id)
if not os.path.exists(outdir): os.mkdir(outdir)

maxages = [3200, 4000]
require_santosonly = [True, False]
for maxage in maxages:
    for s19s21only in require_santosonly:
        ap.plot_hist_field_gyro_ages(
            outdir, cache_id, MAXAGE=maxage, s19s21only=s19s21only
        )
