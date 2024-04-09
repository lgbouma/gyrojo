import os
from gyrojo.paths import DATADIR, RESULTSDIR
import gyrojo.plotting as ap

outdir = os.path.join(RESULTSDIR, 'gyroage_vs_teff')
if not os.path.exists(outdir): os.mkdir(outdir)

# most useful
for yscale in ['linear', 'log']:
    for showerrs in [0, 1]:
        for showplanets in [0, 1]:
            ap.plot_gyroage_vs_teff(outdir, yscale=yscale, showerrs=showerrs,
                                    showplanets=showplanets)
