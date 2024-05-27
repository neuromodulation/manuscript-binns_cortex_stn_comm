Simulated data and example scripts for performing the analyses.

Overview of simulated data:
- two sets of channels (mimicking ECoG and STN signals) interacting in the high
  beta (20-30 Hz) band
- there are simulations for three example subjects (each using a different
  random seed for the simulation)
- for each subject, there are two sessions, one with strong connectivity
  (mimicking OFF therapy recordings) and one with weak connectivity (mimicking
  ON therapy recordings)
- data is stored in the BIDS format

Overview of files for analysing the simulated data:
- "0_simulate_data_demo": a script for simulating the demo data; note that this
  does not need to be run as the simulated data is already included, however
  the script is provided for transparency
- "1_data_preprocessing_preset_demo": an example preprocessing script (similar
  to those in the "processing" folder) with filepaths and settings filled in
- "2_data_processing_coh_preset_demo": an example processing script (similar to
  those in the "processing" folder) with filepath and settings filled in
- "3_analyse_results_demo": an example analysis script (similar to those in the
  "analysis" folder) with filepaths and settings filled in
- "4_plot_results_demo": an example plotting notebook (similar to those in the
  "plotting" folder) with code for plotting the simulated imaginary coherency

To check reproducibility, an image "Demo_ImCoh.png" is also present which
should be treated as an expected output and used to compare the results of the
above files.

Running all the demo files should not take more than a few minutes.