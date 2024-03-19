This is where the cohort-wise settings for preprocessing recordings is stored.

Each type of preprocessing is stored as a json file:
- "explore_parrm": for identifying optimal PARRM filter settings
- "for_annotations": for annotating artefacts in data
- "for_connectivity": for performing oscillatory connectivity analyses (ImCoh,
  MIC, TRGC)
- "for_power": for performing power analyses (but not SSD)
- "for_ssd": for performing SSD analysis
- "for_tde": for performing time delay analysis.