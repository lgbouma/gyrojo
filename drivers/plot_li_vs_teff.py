import os
from gyrojo.paths import DATADIR, RESULTSDIR
import gyrojo.plotting as ap

outdir = os.path.join(RESULTSDIR, 'li_vs_teff')
if not os.path.exists(outdir): os.mkdir(outdir)

# most useful
ap.plot_li_vs_teff(outdir, sampleid='koi_X_JUMP',
                   yscale='linear', limodel='eagles',
                   show_dispersion=0)
ap.plot_li_vs_teff(outdir, sampleid='koi_X_JUMP',
                   yscale='log', limodel='eagles',
                   show_dispersion=0)
ap.plot_li_vs_teff(outdir, sampleid='koi_X_S19S21dquality',
                   yscale='log', limodel='eagles',
                   show_dispersion=0)
ap.plot_li_vs_teff(outdir, sampleid='koi_X_S19S21dquality',
                   yscale='linear', limodel='eagles', show_dispersion=0,
                   nodata=1, show_dispersionpoints=1)

assert 0

# bonus options
for yscale in ['linear', 'log']:
    for limodel in ['eagles', 'baffles']:
        ap.plot_li_vs_teff(outdir, yscale=yscale, limodel=limodel)
