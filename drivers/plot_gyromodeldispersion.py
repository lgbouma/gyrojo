import os
from gyrojo.paths import DATADIR, RESULTSDIR
import gyrojo.plotting as ap

outdir = os.path.join(RESULTSDIR, 'gyromodeldispersion')
if not os.path.exists(outdir): os.mkdir(outdir)

# logg vs Teff
ap.plot_gyromodeldispersion(outdir)


