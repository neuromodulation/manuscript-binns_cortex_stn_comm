This is where the cohort-wise settings for processing preprocessed data is
stored.

Each type of processing is stored as a json file:
- "con_granger_regional": for computing TRGC from regional cortex (e.g. motor
  cortex) to entire STN
- "con_granger_whole": for computing TRGC from entire cortex to entire STN
- "con_imcoh": for computing ImCoh between cortex and STN
- "con_mic_regional": for computing MIC between regional cortex (e.g. motor
  cortex) and entire STN
- "con_mic_whole": for computing MIC between entire cortex and entire STN
- "con_tde": for computing time delays between cortex and STN
- "pow_multitaper": for computing power in cortex and STN
- "pow_ssd": for computing SSD in cortex and STN