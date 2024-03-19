This is where the cohort-wise settings for analysing processed data is stored.

Each type of analysis is stored as a json file:
- "con_granger_regional": for analysing TRGC from regional cortex (e.g. motor
  cortex) to entire STN
- "con_granger_whole": for analysing TRGC from entire cortex to entire STN
- "con_imcoh_regional": for analysing ImCoh between regional cortex (e.g. motor
  cortex) and entire STN
- "con_imcoh_whole": for analysing ImCoh between entire cortex and entire STN
- "con_mic_fibre_tracking_hyperdirect": for associating MIC weights with
  individual hyperdirect fibres
- "con_mic_regional": for analysing MIC between regional cortex (e.g. motor
  cortex) and entire STN
- "con_mic_topography": for analysing MIC patterns of entire cortex and entire
  STN
- "con_mic_whole": for analysing MIC between entire cortex and entire STN
- "con_tde": for analysing time delays between cortex and STN
- "pow_ssd_noavg": for analysing SSD in cortex and STN without averaging power
  over frequency bands
- "pow_ssd_topography": for analysing SSD patterns of entire cortex and entire
  STN
- "pow_ssd": for analysing SSD in entire cortex and entire STN
- "pow_standard_channels": for analysing power in cortex and STN of single
  channels
- "pow_standard_regional": for analysing power in cortex and STN of regions
  (e.g. motor cortex)
- "pow_standard_whole": for analysing power in entire cortex and STN

In all cases:
- "MedOffOn" will compare OFF therapy and ON levodopa data
- "StimOffOn" will compare OFF therapy and ON STN-DBS data
- "single_sub" handles data of each subject separately
- "multi_sub" handles data of multiple subjects together