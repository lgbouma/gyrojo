import os
import agetools.plotting as ap
from agetools.paths import RESULTSDIR

cache_id = "koi_lithium_posteriors_20230208"

outdir = os.path.join(RESULTSDIR, cache_id)
if not os.path.exists(outdir): os.mkdir(outdir)

ap.plot_koi_li_posteriors(outdir, cache_id)
