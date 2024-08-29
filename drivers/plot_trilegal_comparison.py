import os
from gyrojo.paths import DATADIR, RESULTSDIR
import gyrojo.plotting as ap

outdir = os.path.join(RESULTSDIR, 'trilegal_comparison')
if not os.path.exists(outdir): os.mkdir(outdir)

ap.plot_threepanel_trilegal_comparison(outdir)

ap.plot_trilegal_comparison(outdir, const_sfr=0)
ap.plot_trilegal_comparison(outdir, const_sfr=1)
