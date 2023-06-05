import os
from gyrojo.paths import DATADIR, RESULTSDIR
import gyrojo.plotting as ap

outdir = os.path.join(RESULTSDIR, 'rp_vs_porb_binage')
if not os.path.exists(outdir): os.mkdir(outdir)

ap.plot_rp_vs_porb_binage(outdir)
