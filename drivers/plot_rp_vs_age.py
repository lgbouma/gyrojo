import os
from agetools.paths import DATADIR, RESULTSDIR
import agetools.plotting as ap

outdir = os.path.join(RESULTSDIR, 'rp_vs_age')
if not os.path.exists(outdir): os.mkdir(outdir)

ap.plot_rp_vs_age(outdir, xscale='log', shortylim=1, lowmes=1)
ap.plot_rp_vs_age(outdir, xscale='linear', shortylim=1, lowmes=1)
ap.plot_rp_vs_age(outdir, xscale='log', shortylim=1)
ap.plot_rp_vs_age(outdir, xscale='linear', shortylim=1)
assert 0
ap.plot_rp_vs_age(outdir, xscale='log', elinewidth=0)
ap.plot_rp_vs_age(outdir, xscale='linear')
ap.plot_rp_vs_age(outdir, xscale='log')
