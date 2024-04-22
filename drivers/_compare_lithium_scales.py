import pandas as pd, numpy as np, matplotlib.pyplot as plt
import os
from os.path import join
from gyrojo.paths import RESULTSDIR
from gyrojo.getters import get_li_data
from aesthetic.plot import set_style, savefig
from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1

CATALOG = "J/ApJ/855/115" # Berger+2018
catalogs = Vizier.get_catalogs(CATALOG)

bdf = catalogs[0].to_pandas()
bdf['KOI'] = bdf['KOI'].astype(str)

sampleid = 'koi_X_S19S21dquality' # TODO FIXME: once koi_X_JUMP done, rerun
df = get_li_data(sampleid)
df['matchstr'] =[s[2:-3] for s in df.kepoi_name.astype(str)]

dfcols = ('Fitted_Li_EW_mA,adopted_Teff,pl_name,matchstr,'
          'Fitted_Li_EW_mA_perr,Fitted_Li_EW_mA_merr,kepoi_name'.split(","))
bcols = 'KOI,EW_Li_,Teff,e_EW_Li_'.split(",")
mdf = df[dfcols].merge(bdf[bcols], how='inner', left_on='matchstr', right_on='KOI')





set_style("science")
fig, axs = plt.subplots(nrows=2, figsize=(3,6), sharex=True)
ax = axs[0]
xerr = np.array(
    [mdf['Fitted_Li_EW_mA_merr'], mdf['Fitted_Li_EW_mA_perr']]
).reshape((2, len(mdf)))
meanxerr = np.nanmean(xerr, axis=0)
yerr = np.array(mdf['e_EW_Li_'])
color = np.array(mdf.adopted_Teff)
sc = ax.scatter(mdf['Fitted_Li_EW_mA'], mdf['EW_Li_'], s=1, c=color,
                cmap='viridis', zorder=99)
a, b, c = ax.errorbar(
    mdf['Fitted_Li_EW_mA'], mdf['EW_Li_'],
    xerr=xerr, yerr=yerr,
    marker='o', elinewidth=0.1, capsize=0, lw=0, mew=0.5,
    markersize=1, zorder=5, c='k'
)
ax.plot( [0,200], [0,200], c='k', alpha=0.4, ls='--', zorder=-1)
ax.update({
    'ylabel': 'EW (mA Berger+18)'
})

ax = axs[1]
xval = mdf['Fitted_Li_EW_mA']
yval = mdf['Fitted_Li_EW_mA']-mdf['EW_Li_']
ax.scatter(xval, yval, s=1, c=color, cmap='viridis', zorder=99)
a,b,c = ax.errorbar(
    xval, yval,
    xerr=xerr, yerr=np.sqrt(meanxerr**2 + yerr**2),
    marker='o', elinewidth=0.1, capsize=0, lw=0, mew=0.5,
    markersize=1, zorder=5, c='k'
)
ax.plot( [0,200], [0,0], c='k', alpha=0.4, ls='--', zorder=-1)
ax.update({
    'xlabel': 'EW (mA, this work)',
    'ylabel': f'(Me-Berger)',
    'ylim': [-40, 40]
})

outdir = join(RESULTSDIR, "compare_lithium_scales")
if not os.path.exists(outdir): os.mkdir(outdir)
outpath = join(outdir, f'lithium_scale_comparison.png')

fig.tight_layout(w_pad = 0)

savefig(fig, outpath)




# cases for which Berger reported ~>20mA EWs, and I reported negative!
pcols = (
    'Fitted_Li_EW_mA,EW_Li_,Fitted_Li_EW_mA_perr,'
    'Fitted_Li_EW_mA_merr,e_EW_Li_,kepoi_name'.split(",")
)

sel = yval < -20
print(42*'-')
print(mdf[sel][pcols])

sel = yval > 20
print(42*'-')
print(mdf[sel][pcols])


sel = xval > 25
mean_offset = np.nanmean(yval[sel])
median_offset = np.nanmedian(yval[sel])
std_offset = np.nanstd(yval[sel])
print(42*'-')
print(f'mean_offset={mean_offset:.2f}, median_offset={median_offset:.2f} std_offset={std_offset:.2f}')
