import os
from gyrojo.paths import DATADIR, RESULTSDIR
import gyrojo.plotting as ap

outdir = os.path.join(RESULTSDIR, 'st_params')
if not os.path.exists(outdir): os.mkdir(outdir)

# logg vs Teff
ap.plot_st_params(outdir, xkey='adopted_Teff', ykey='adopted_logg')

# CAMDs
ap.plot_st_params(outdir, xkey='dr3_bp_rp', ykey='M_G')
ap.plot_st_params(outdir, xkey='adopted_Teff', ykey='M_G')

# mKep vs Teff
ap.plot_st_params(outdir, xkey='adopted_Teff', ykey='dr3_phot_g_mean_mag')
