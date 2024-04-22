import os
from gyrojo.paths import DATADIR, RESULTSDIR
import gyrojo.plotting as ap

outdir = os.path.join(RESULTSDIR, 'liagefloor_vs_teff')
if not os.path.exists(outdir): os.mkdir(outdir)

ap.plot_liagefloor_vs_teff(outdir)
