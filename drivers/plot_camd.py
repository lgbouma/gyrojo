import os
from gyrojo.paths import DATADIR, RESULTSDIR
import gyrojo.plotting as ap

outdir = os.path.join(RESULTSDIR, 'camd')
if not os.path.exists(outdir): os.mkdir(outdir)

ap.plot_camd(outdir, xkey='dr3_bp_rp')
ap.plot_camd(outdir, xkey='adopted_Teff')
