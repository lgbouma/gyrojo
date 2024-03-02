import os
from gyrojo.paths import DATADIR, RESULTSDIR
import gyrojo.plotting as ap

outdir = os.path.join(RESULTSDIR, 'multis_vs_age')
if not os.path.exists(outdir): os.mkdir(outdir)

ap.plot_multis_vs_age(outdir)

