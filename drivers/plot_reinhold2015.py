import os
from gyrojo.paths import DATADIR, RESULTSDIR
import gyrojo.plotting as ap

outdir = os.path.join(RESULTSDIR, 'reinhold_2015')
if not os.path.exists(outdir): os.mkdir(outdir)

ap.plot_reinhold_2015(outdir)
