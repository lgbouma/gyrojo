import os
from os.path import join
import numpy as np, pandas as pd
from gyrojo.getters import (
    get_gyro_data, get_age_results, get_koi_data,
    select_by_quality_bits
)
from gyrojo.paths import DATADIR
from gyrojo.papertools import update_latex_key_value_pair as ulkvp
from gyrojo.papertools import read_latex_key_value_pairs
from gyrojo.papertools import (
    format_lowerlimit, cast_to_int_string, replace_nan_string
)


##########################################
# cumulative KOI table
koi_df = get_koi_data('cumulative-KOI', grazing_is_ok=1)

# Number of KOIs that are not known false positives
sel = ~(koi_df['flag_koi_is_fp'])
N = len(koi_df[sel])
ulkvp('nkoisnofp', N)


##########################################
# Stars with rotation periods and quality flags calculated
df = get_gyro_data("Santos19_Santos21_dquality")

assert pd.isnull(df.Prot).sum() == 0
assert len(np.unique(df.KIC)) == len(df)

# Number of unique stars with Santos+ rotation period reported
# (...and good Gaia matches)
sel = select_by_quality_bits(df, [4, 5], [0, 0])
N = len(df[sel])
ulkvp('nuniqstarsantosrot', N)

# Number of unique stars with Santos+ rotation period reported, and
# 3800<Teff<6200K (i.e. can hypothetically compute a gyro age).
sel = select_by_quality_bits(df, [0, 4, 5], [0, 0, 0])
N = len(df[sel])
ulkvp('nuniqstarsantosrotteffcut', N)

# Number of unique stars with finite gyro ages.  (i.e. did gyrointerp produce
# gyro ages for all stars noted above?)
N2 = np.sum(~pd.isnull(df[sel].gyro_median))
ulkvp('nuniqstarfinitegyroage', N2)
assert N == N2


# Number of unique stars with Santos+ rotation period reported, and
# 3800<Teff<6200K (i.e. can compute a gyro age).

#Further requiring them to be apparently single, near the main
#sequence, with $\log g$$>$4.2, $M_{\rm G}>3.9$, and not flagged as
#eclipsing binaries 
sel = select_by_quality_bits(
    df, [0, 1, 2, 3, 4, 5, 6, 9],
        [0, 0, 0, 0, 0, 0, 0, 0]
)
N = len(df[sel])
ulkvp('nuniqstarsantosallbutruwe', N)

# Number of unique stars with Santos+ rotation, and gyro applicable
sel = select_by_quality_bits(
    df, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
)
N = len(df[sel])
M = len(df[df.flag_is_gyro_applicable])
assert N == M
ulkvp('nuniqstarsantosrotgyroappl', N)

##########################################
# get PLANET numbers!
kdf = get_koi_data('cumulative-KOI', grazing_is_ok=1)

# Number of non-false positive KOIs from cumulative koi df
N = len(kdf[~kdf.flag_koi_is_fp])
ulkvp('nnonfpcumkois', N)

# Number of unique non-false positive KOI host stars from cumulative koi df
N = len(np.unique(kdf[~kdf.flag_koi_is_fp].kepid))
ulkvp('nnonfpcumkoihosts', N)

##########################################
# ...with age results
df, _, _ = get_age_results(
    whichtype='allageinfo', grazing_is_ok=1, drop_highruwe=0
)

hasjump = ~pd.isnull(df.filename)
hasprot = ~pd.isnull(df.Prot)
n_pl_jump = len(df[hasjump].kepid)
n_st_jump = len(df[hasjump].kepid.unique())
n_pl_jump_prot = len(df[hasjump & hasprot].kepid)
n_st_jump_prot = len(df[hasjump & hasprot].kepid.unique())

ulkvp('nstwithspec', n_st_jump)
ulkvp('nplwithspec', n_pl_jump)
ulkvp('nstwithspecandprot', n_st_jump_prot)
ulkvp('nplwithspecandprot', n_pl_jump_prot)

##########################################
# specifics for special stars / planets
sdf = df[~df.kepler_name.isna()]
for c in sdf.columns:
    if c.startswith('gyro_') or c.startswith("li_"):
        sdf[c] = sdf[c].apply(cast_to_int_string)
sdf['t_gyro'] = sdf.apply(lambda row: "$"+ f"{row['gyro_median']}"+ "^{+"+
                          f"{row['gyro_+1sigma']}"+ "}_{-"+
                          f"{row['gyro_-1sigma']}" +"}$", axis=1)
sdf['t_li'] = sdf.apply(lambda row: "$"+ f"{row['li_median']}"+ "^{+"+
                          f"{row['li_+1sigma']}"+ "}_{-"+
                          f"{row['li_-1sigma']}" +"}$", axis=1)

sel = sdf.kepler_name.str.contains("Kepler-66 b")
t = sdf.loc[sel, 't_gyro'].iloc[0]
ulkvp('kepsixsixtgyro', t)

sel = sdf.kepler_name.str.contains("Kepler-67 b")
t = sdf.loc[sel, 't_gyro'].iloc[0]
ulkvp('kepsixseventgyro', t)

sel = sdf.kepler_name.str.contains("Kepler-1 b")
t = sdf.loc[sel, 't_li'].iloc[0]
ulkvp('trestwotli', t)

sel = sdf.kepler_name.str.contains("Kepler-786 b")
t = sdf.loc[sel, 't_li'].iloc[0]
ulkvp('kepseveneightsix', t)

sel = sdf.kepler_name.str.contains("Kepler-1312 b")
t = sdf.loc[sel, 't_gyro'].iloc[0]
ulkvp('kepthirteentwelve', t)

sel = sdf.kepler_name.str.contains("Kepler-1561 b")
t = sdf.loc[sel, 't_gyro'].iloc[0]
ulkvp('kepfifteensixone', t)

sel = sdf.kepler_name.str.contains("Kepler-1629 b")
t = sdf.loc[sel, 't_gyro'].iloc[0]
ulkvp('kepsixteentwonine', t)

sel = sdf.kepler_name.str.contains("Kepler-1644 b")
t = sdf.loc[sel, 't_gyro'].iloc[0]
ulkvp('kepsixteenfourfour', t)

sel = sdf.kepler_name.str.contains("Kepler-1699 b")
t = sdf.loc[sel, 't_gyro'].iloc[0]
ulkvp('kepsixteenninenine', t)

sel = sdf.kepler_name.str.contains("Kepler-1943 b")
t = sdf.loc[sel, 't_li'].iloc[0]
ulkvp('kepnineteenfourthree', t)





##########################################

df, _, _ = get_age_results(
    whichtype='gyro_li', grazing_is_ok=1, drop_highruwe=0
)
assert pd.isnull(df.gyro_median).sum() == 0

# Number of planets with a gyro age, including grazing & high ruwe cases
N = len(df)
ulkvp('nplwgyroagewithgrazingandhighruwe', N)

# Number of planet-hosting stars with a gyro age, including grazing & high ruwe cases
N = len(np.unique(df.kepid))
ulkvp('nplhoststarwgyroagewithgrazingandhighruwe', N)

N = len(np.unique(df[df.flag_dr3_ruwe_outlier].kepid))
ulkvp('nplhoststarwgyroagejusthighruwe', N)


# Number of planets with a gyro age, including grazing & high ruwe cases, below 1Gyr
sel = (df.gyro_median > 0) & (df.gyro_median <= 1000)
N = len(df[sel])
ulkvp('nplyounggyro', N)
N = len(np.unique(df[sel].kepid))
ulkvp('nplhostsyounggyro', N)

# Number of planets with a gyro age, including grazing & high ruwe cases, 1-2 Gyr
sel = (df.gyro_median > 1000) & (df.gyro_median <= 2000)
N = len(df[sel])
ulkvp('nplmidgyro', N)
N = len(np.unique(df[sel].kepid))
ulkvp('nplhostsmidgyro', N)

# Number of planets with a gyro age, including grazing & high ruwe cases, 2-3 Gyr
sel = (df.gyro_median > 2000) & (df.gyro_median <= 3000)
N = len(df[sel])
ulkvp('nploldgyro', N)
N = len(np.unique(df[sel].kepid))
ulkvp('nplhostsoldgyro', N)

# Number of planets with a gyro age, including grazing & high ruwe cases, below 1Gyr at 2sigma
sel = (df['gyro_median'] + df['gyro_+2sigma']  <= 1000)
N = len(df[sel])
ulkvp('nplyounggyrotwosigma', N)
N = len(np.unique(df[sel].kepid))
ulkvp('nplhostsyounggyrotwosigma', N)

# Number of planets with a gyro age, including grazing & high ruwe cases, below 1Gyr at 3sigma
sel = (df['gyro_median'] + df['gyro_+3sigma']  <= 1000)
N = len(df[sel])
ulkvp('nplyounggyrothreesigma', N)
N = len(np.unique(df[sel].kepid))
ulkvp('nplhostsyounggyrothreesigma', N)

# ... now drop grazing cases, keeping high ruwe
df, _, _ = get_age_results(
    whichtype='gyro', grazing_is_ok=0, drop_highruwe=0
)

# Number of planets with a gyro age, dropping grazing cases
N = len(df)
ulkvp('nplwgyroagenograzing', N)

# Number of planet-hosting stars with a gyro age, dropping grazing cases
N = len(np.unique(df.kepid))
ulkvp('nplhoststarwgyroagenograzing', N)

df, _, _ = get_age_results(
    whichtype='gyro', grazing_is_ok=0, drop_highruwe=1
)
# Planets <1 Gyr at 2-sigma with a gyro age, no grazing, no ruwe.
sel = (df['gyro_median'] + df['gyro_+2sigma']  <= 1000)
N = len(df[sel])
ulkvp('nplyounggyrotwosigmanograzingnoruwe', N)



##########################################
# assertion checks:

d = read_latex_key_value_pairs()

# lithium pieces written by koi_jump_getter; others written here.
# they should be the same, else the two refer to different samples.
#assert d['nlithiumgyrostars'] == d['nplhoststarwgyroagewithgrazingandhighruwe']
#assert d['nlithiumgyroplanets'] == d['nplwgyroagewithgrazingandhighruwe']
