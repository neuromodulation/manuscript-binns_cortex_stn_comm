Scripts for plotting results.

The different files are:
- "fmri_lme_coeffs_indirect_nuclei": plot indirect pathway nuclei according to
  fMRI-LME coefficients (Figure 4B)
- "imcoh": plot ImCoh (Figure 3A & 3B)
- "mic_patterns_hyperdirect_fibres": plot hyperdirect fibres coloured according
  to MIC patterns (Figure 4A)
- "mic_patterns": create interpolated MIC patterns and plot raw points (Figures
  3A, S3)
- "power": plot power (univariate and multivariate; Figure 2)
- "ssd_patterns_dbs_contacts": plot SSD patterns for DBS contacts (Figure 2C)
- "ssd_patterns": create interpolated SSD patterns and plot raw points (Figures
  2B & S1)
- "tde": plot time delay estimates (Figure 3D)
- "trgc": plot TRGC (Figures 3C & S4)

There are 2 general types of files:
- "MedOffOn": for comparing OFF therapy and ON levodopa results
- "StimOffOn": for comparing OFF therapy and ON STN-DBS results

Files have some shared constants:
- "FOLDERPATH_ANALYSIS": directory where the analysis settings are stored and
  where the results will be stored (e.g. in "Project/Analysis")