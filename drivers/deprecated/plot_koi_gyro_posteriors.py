import os
import gyrojo.plotting as ap
from gyrojo.paths import RESULTSDIR

cache_id = "koi_gyro_posteriors_20230208"

outdir = os.path.join(RESULTSDIR, cache_id)
if not os.path.exists(outdir): os.mkdir(outdir)

ap.plot_koi_gyro_posteriors(outdir, cache_id)
