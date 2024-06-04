import os
from gyrojo.paths import DATADIR, RESULTSDIR
import gyrojo.plotting as ap

outdir = os.path.join(RESULTSDIR, 'perioddiff_vs_period')
if not os.path.exists(outdir): os.mkdir(outdir)

ap.plot_perioddiff_vs_period(outdir, ykey='m14_Prot', ylim=[-4.3, 4.3],
                             dx=0.25, dy=0.1)

ap.plot_perioddiff_vs_period(outdir, ykey='r23_ProtGPS', dx=0.5, dy=0.2)

ap.plot_perioddiff_vs_period(outdir, ykey='r23_ProtFin', dx=0.5, dy=0.2)

