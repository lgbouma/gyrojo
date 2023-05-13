import os
from agetools.paths import DATADIR, RESULTSDIR
import agetools.plotting as ap

outdir = os.path.join(RESULTSDIR, 'li_vs_teff')
if not os.path.exists(outdir): os.mkdir(outdir)

for yscale in ['linear', 'log']:
    ap.plot_li_vs_teff(outdir, yscale=yscale)
