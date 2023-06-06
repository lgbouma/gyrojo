--------------------
To run the field star gyro analysis:

1. `calc_field_gyro_posteriors.py`

2. `plot_field_gyro_posteriors.py`

Makes `field_gyro_posteriors_20230529_gyro_ages_X_GDR3_S19_S21_B20.csv`

3. `construct_field_star_gyro_quality_flags.py`

Given output from #2, build
`field_gyro_posteriors_20230529_gyro_ages_X_GDR3_S19_S21_B20_with_dquality.csv`

4. `plot_hist_field_gyro_ages.py`

--------------------
To run the KOI gyrochronology analysis:

1. `build_koi_table.py`:

Takes the cumulative KOI table, and left-joins it against Gaia DR3, the
Berger+20 stellar parameter catalog, and all relevant rotation period catalogs
(McQuillan+14, Mazeh+15, Santos+19, Santos+21).  The output is written to
`DATADIR/interim/koi_table_X_GDR3_B20_S19_S21_M14_M15.csv`

2. `calc_koi_gyro_posteriors.py`

Using the above table, calculate gyro the posteriors.

3. `plot_koi_gyro_posteriors.py`

Plot the resulting gyro posteriors, and write a table of their summary
statistics joined against the GDR3/B20/etc data:
`/results/koi_gyro_posteriors_{DATESTR}/step0_koi_gyro_ages_X_GDR3_B20_S19_S21_M14_M15.csv`

--------------------
To run the KOI lithium analysis:

1. Run `kepler_lithium_sample_getter.sql` on JUMP

2. `prepare_koi_jump_getter.py`

Crossmatch the gyro KOI sample (output of number #3 above) against JUMP.  Needs
access to the JUMP "explore" SQL interface to run.  Cleans the initial SQL
query results, and constucts the list of necessary spectra, and makes an scp
script that can be used to pull the deblazed spectra from shrek.
Also makes CSV files like `/data/interim/koi_jump_getter_{sampleid}.csv`, which
contain the JUMP and Gaia, B20, S19/21, etc information.

3. `scp_HIRES_lithium_data.sh`

Run this to then get the HIRES spectra to the local drive.

4.  Measure the lithium equivalent widths from the spectra, using
    `measure_Li_EWs.py`

5. `calc_koi_lithium_posteriors.py`

Use the lithium equivalent widths, and estimates of B-V colors, to calculate
the lithium posteriors and cache them to disk.

6. `plot_koi_lithium_posteriors.py`

--------------------
To multiply the gyro and lithium age posteriors, and entirely statistically
unjustified procedure:

1. `calc_koi_joint_posteriors.py`

Given the gyro and lithium posteriors, multiply them.  Saves output, including
a merged file with all the gyro, lithium, and joint summary statistics, at 
`koi_gyro_X_lithium_posteriors_20230116/{sampleid}_merged_joint_age_posteriors.csv`

--------------------
Other plots that will probably go into a manuscript:

* `plot_rp_vs_age.py`

* `plot_rp_vs_porb_binage.py`

* `plot_koi_mean_prot_vs_teff.py` 

* `plot_li_vs_teff.py`
