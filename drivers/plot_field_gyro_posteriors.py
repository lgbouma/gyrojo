import os
import gyrojo.plotting as ap
from gyrojo.paths import RESULTSDIR

cache_id = "field_gyro_posteriors_20230529"

outdir = os.path.join(RESULTSDIR, cache_id)
if not os.path.exists(outdir): os.mkdir(outdir)

ap.plot_field_gyro_posteriors(outdir, cache_id)
