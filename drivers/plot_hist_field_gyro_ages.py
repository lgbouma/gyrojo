import os
import gyrojo.plotting as ap
from gyrojo.paths import RESULTSDIR

# calculated in calc_field_gyro_posteriors.py
cache_id = "hist_field_gyro_ages_20230529"

outdir = os.path.join(RESULTSDIR, cache_id)
if not os.path.exists(outdir): os.mkdir(outdir)

ap.plot_hist_field_gyro_ages(outdir, cache_id, MAXAGE=3200)
ap.plot_hist_field_gyro_ages(outdir, cache_id, MAXAGE=4000)
