import os
from os.path import join
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from gyrojo.getters import (
    get_gyro_data, get_age_results, get_koi_data
)
from gyrojo.papertools import update_latex_key_value_pair as ulkvp
from gyrojo.paths import PAPERDIR

# ...with age results
drop_highruwe = 0
drop_grazing = 0
_df, _, _ = get_age_results(
    whichtype='gyro', drop_grazing=drop_grazing, drop_highruwe=drop_highruwe
)

# ...kepler-52 is <1002myr at 2sigma lol.
sel = (_df['gyro_median'] + _df['gyro_+2sigma']  <= 1003)
N = len(_df[sel])

df = _df[sel]

selcols = (
    "kepoi_name,kepler_name,koi_disposition,gyro_median,"
    "gyro_+1sigma,gyro_-1sigma,adopted_rp,adopted_period,flag_ruwe_outlier,flag_koi_is_grazing"
).split(",")
pdf = df[selcols].sort_values(
    by=['gyro_median','kepler_name','kepoi_name']
)

for c in pdf.columns:
    if 'gyro' in c:
        pdf[c] = pdf[c].astype(int)
    if c == 'adopted_period':
        pdf[c] = np.round(pdf[c], 2)
    if 'flag' in c:
        pdf[c] = pdf[c].astype(int)

pcols = (
    "kepoi_name,kepler_name,gyro_median,"
    "gyro_+1sigma,gyro_-1sigma,adopted_rp,adopted_period,flag_ruwe_outlier,flag_koi_is_grazing"
).split(",")

print(42*'-')
print(pdf[pdf.koi_disposition == 'CONFIRMED'][pcols])
print(42*'-')
print(pdf[pdf.koi_disposition == 'CANDIDATE'][pcols])

# where are kepler 1627 and 1643???
#
# Kepler-1627: failed RUWE cut
# --> can merge that...
#
# Kepler-1643: Porb~Prot within 20%, and was a KOI, and so was treated in a
# special way in the Santos+ processing (see email thread: "Rotation period of
# Kepler-1643").  Quoting A. Santos: "From our current table of about 800 KOIs
# with Prot, we have 73 Porb~Prot.  From these 73, 16 have Prot<10 days."
# ... this is kind of a TODO...

pdf['t_gyro'] = pdf.apply(
    lambda row:
    "$"+
    f"{row['gyro_median']}"+
    "^{+"+
    f"{row['gyro_+1sigma']}"+
    "}_{-"+
    f"{row['gyro_-1sigma']}"
    +"} $",
    axis=1
)

# Drop the original gyro columns
pdf = pdf.drop(columns=['gyro_median', 'gyro_+1sigma', 'gyro_-1sigma'])

mapdict = {
    'kepoi_name': "KOI",
    "kepler_name": "Kepler",
    "t_gyro": r"$t_{\rm gyro}$",
    "adopted_rp": r"$R_{\rm p}$",
    "adopted_period": r"$P$",
    "flag_ruwe_outlier": r"$f_{\rm RUWE}$",
    "flag_koi_is_grazing": r"$f_{\rm grazing}$",
}
rounddict = {
    'adopted_period': 2,
    'adopted_rp': 2,
}
formatters = {}
for k,v in rounddict.items():
    formatters[mapdict[k]] = lambda x: np.round(x, v)

pdf = pdf.rename(
    mapdict, axis='columns'
)

sel = (pdf.koi_disposition == 'CONFIRMED')
latexpath1 = join(PAPERDIR, 'table_subgyr_confirmed.tex')
pdf[sel][mapdict.values()].to_latex(
    latexpath1, index=False, na_rep='--', escape=False, formatters=formatters
)

sel = (pdf.koi_disposition == 'CANDIDATE')
latexpath2 = join(PAPERDIR, 'table_subgyr_candidate.tex')
t2 = pdf[sel][mapdict.values()].to_latex(
    latexpath2, index=False, na_rep='--', escape=False, formatters=formatters
)

# trim them..
for texpath in [latexpath1, latexpath2]:
    with open(texpath, 'r') as f:
        texlines = f.readlines()
    texlines = texlines[4:-2] # trim header and footer
    with open(texpath, 'w') as f:
        f.writelines(texlines)
    print(f"Wrote {texpath}")
