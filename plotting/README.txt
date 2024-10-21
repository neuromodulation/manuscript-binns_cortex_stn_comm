Scripts for plotting results and contact locations.

The different files are:
- "dbs_lfp_locations": plot DBS leads in the STN (Figure 2a)
- "ecog_lfp_locations": plot ECoG contacts on the cortex (Figure 2a)
- "fmri_lme_coeffs_indirect_nuclei": plot indirect pathway nuclei according to
  fMRI-LME coefficients (Figure 4b)
- "imcoh": plot ImCoh (Figure 3a & 3b)
- "mic_patterns_hyperdirect_fibres": plot hyperdirect fibres coloured according
  to MIC patterns (Figure 4a)
- "mic_patterns": create interpolated MIC patterns and plot raw points (Figures
  3a, S7)
- "power": plot power (univariate and multivariate; Figures 2, S1, & S2)
- "ssd_patterns_dbs_contacts": plot SSD patterns for DBS contacts (Figure 2c)
- "ssd_patterns": create interpolated SSD patterns and plot raw points (Figures
  2b, S3, & S4)
- "tde": plot time delay estimates (Figure 3d)
- "trgc": plot TRGC (Figures 3c, S5, & S6)

There are 2 general types of files:
- "MedOffOn": for comparing OFF therapy and ON levodopa results
- "StimOffOn": for comparing OFF therapy and ON STN-DBS results

Files have some shared constants:
- "FOLDERPATH_ANALYSIS": directory where the analysis settings are stored and
  where the results will be stored (e.g. in "Project/Analysis")