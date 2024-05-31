import os
import gyrojo.plotting as ap
from gyrojo.paths import RESULTSDIR

#sampleid = 'Santos19_Santos21_all'
#cache_id = "field_gyro_posteriors_20240405"
sampleid = 'Santos19_Santos21_litsupp_all'
cache_id = "field_gyro_posteriors_20240530"

outdir = os.path.join(RESULTSDIR, cache_id)
if not os.path.exists(outdir): os.mkdir(outdir)

ap.plot_field_gyro_posteriors(outdir, cache_id, sampleid)
