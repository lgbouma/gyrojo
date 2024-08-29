#!/bin/bash

# plots that made the ms
python plot_hist_field_gyro_ages.py
python plot_star_Prot_Teff.py
python plot_koi_mean_prot_teff.py
python plot_li_vs_teff.py
python plot_st_params.py
python compare_literature_ages_to_cluster_ages.py
python plot_trilegal_comparison.py

# appendix plots
python plot_perioddiff_vs_period.py
python plot_scatter_gyroage_vs_teff.py
python plot_mcq14_vs_santos_age_histograms.py

# write latex tables and value files for ms
python _compare_lithium_scales.py
python get_values_for_manuscript.py
python make_younghighlight_table.py
python make_star_table.py

# generate rp vs period vs age
python make_gyroonly_table.py
cwd=$(pwd)
cd ~/Dropbox/proj/NEA_age_plots/drivers
python plot_rp_vs_period_scatter.py
cd "$cwd"

echo 'done! 🎉'
exit 0
