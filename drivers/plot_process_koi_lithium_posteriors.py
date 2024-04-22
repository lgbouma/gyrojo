import os
import gyrojo.plotting as ap
from gyrojo.paths import RESULTSDIR

li_method = 'eagles' # or "baffles"
datestr = "20240405"

cache_id = f"koi_lithium_posteriors_{li_method}_{datestr}"

outdir = os.path.join(RESULTSDIR, cache_id)

sampleid = 'koi_X_JUMP'
ap.plot_process_koi_li_posteriors(outdir, cache_id, sampleid, li_method=li_method)
