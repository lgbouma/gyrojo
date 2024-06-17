import os
from gyrojo.paths import DATADIR, RESULTSDIR
import gyrojo.plotting as gp

outdir = os.path.join(RESULTSDIR, 'star_Prot_Teff')
if not os.path.exists(outdir): os.mkdir(outdir)

gp.plot_star_Prot_Teff(outdir, 'teff_age_prot_seed42_nstar20000')

for sampleid in [
    "McQuillan2014only_dquality",
    "McQuillan2014only",
    'Santos19_Santos21_dquality',
    'Santos19_Santos21_litsupp_all',
    #'Santos19_Santos21_clean0'
]:
    gp.plot_star_Prot_Teff(outdir, sampleid)
