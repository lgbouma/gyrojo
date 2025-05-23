r="/Users/luke/Dropbox/proj/gyrojo/results"

# plot_gyromodeldispersion
cp $r/gyromodeldispersion/gyromodeldispersion.pdf .

# plot_st_params
# ...CAMD:
cp $r/st_params/st_params_M_G_vs_dr3_bp_rp.pdf .
#cp $r/st_params/st_params_M_G_vs_adopted_Teff.pdf .
# ...logg vs Teff
cp $r/st_params/st_params_adopted_logg_vs_adopted_Teff.pdf .
# ...G vs Teff
#cp $r/st_params/st_params_dr3_phot_g_mean_mag_vs_adopted_Teff.pdf .

cp $r/star_Prot_Teff/prot_teff_Santos19_Santos21_dquality.pdf .

# plot_koi_mean_prot_teff
cp $r/koi_mean_prot_teff/koi_mean_prot_teff_koi_X_S19S21dquality_keepgrazing.pdf .

# plot_hist_field_gyro_ages 
cp $r/hist_field_gyro_ages_20240821/comp_hist_samples_koi_gyro_ages_hist_field_gyro_ages_20240821_maxage4000.pdf .
cp $r/hist_field_gyro_ages_20240821/comp_hist_samples_koi_gyro_ages_hist_field_gyro_ages_20240821_maxage4000_preciseagesonly.pdf .
cp $r/hist_field_gyro_ages_20240821/comp_hist_samples_koi_gyro_ages_hist_field_gyro_ages_20240821_maxage4000_dropfracshortrot.pdf .

# plot_li_vs_teff
#cp $r/li_vs_teff/li_vs_teff_koi_X_S19S21dquality_eagles_logy.pdf .
cp $r/li_vs_teff/li_vs_teff_koi_X_JUMP_eagles.pdf .
cp $r/li_vs_teff/li_vs_teff_koi_X_S19S21dquality_eagles_showpoints_nodata.pdf .

# trilegal
cp $r/trilegal_comparison/threepanel_trilegal_comparison_twostepsfr.pdf .

# planets
cp $r/rp_vs_period_scatter/rp_vs_period_scatter_20240415_colorbyage_showaux-gyro_anyyoung_anyyoung_allowsgrazingandhighruwe.pdf .


########## 
# appendices (probably omit)
cp $r/gyroage_vs_teff/gyroage_vs_teff_errs_showplanets_linear.png .
cp $r/gyroage_vs_teff/gyroage_vs_teff_errs_linear.png .

cp $r/lit_ages_vs_cluster_ages/olympic_merged_lit_ages_vs_cluster_ages.pdf .

# mpl>3.6 pdf rendering failing
#cp $r/perioddiff_vs_period/perioddiff_vs_period_diffProt-m14_Prot_vs_Prot.pdf .
#cp $r/perioddiff_vs_period/perioddiff_vs_period_diffProt-r23_ProtGPS_vs_Prot.pdf .

cp $r/star_Prot_Teff/prot_teff_McQuillan2014only_dquality.pdf .


cp $r/mcq14_vs_santos_age_histograms_Santos20240821_vs_McQMcQ14_20240613/comp_hist_samples_koi_gyro_ages_hist_field_gyro_ages_20240821_maxage4000.pdf mcq_santos_comp_top.pdf
cp $r/mcq14_vs_santos_age_histograms_Santos20240821_vs_McQMcQ14_20240613/comp_hist_samples_koi_gyro_ages_hist_field_gyro_ages_20240821_maxage4000_preciseagesonly.pdf mcq_santos_comp_bottom.pdf
