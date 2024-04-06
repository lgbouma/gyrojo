import os
from gyrojo.paths import DATADIR, RESULTSDIR
import gyrojo.plotting as ap

outdir = os.path.join(RESULTSDIR, 'li_vs_teff')
if not os.path.exists(outdir): os.mkdir(outdir)

ap.plot_li_vs_teff(outdir, yscale='linear', limodel='eagles', show_dispersion=1)

for yscale in ['linear', 'log']:
    for limodel in ['eagles', 'baffles']:
        ap.plot_li_vs_teff(outdir, yscale=yscale, limodel=limodel)
