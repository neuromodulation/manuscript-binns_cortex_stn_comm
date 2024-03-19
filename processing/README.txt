Scripts for (pre)processing recordings.

Overview of (pre)processing:
- "preprocessing_explore_parrm": identify optimal PARRM settings
- "preprocessing": perform preprocessing
- "preprocessing_project_coords": project ECoG coordinates to brain surface
- "processing_annotations": annotate artefacts in data
- "processing_coh": compute ImCoh
- "processing_fmri_maps": compute whole-brain fMRI maps of connectivity
- "processing_granger_causality": compute TRGC
- "processing_mic": compute MIC
- "processing_power_ssd": compute SSD
- "processing_power_standard": compute power
- "compute_tde": compute time delays

Files generally have 3 types:
- "single": process single recording
- "preset": process multiple recordings serially
- "preset_hpc": process multiple recordings in parallel (e.g. with a
  high-performance cluster), with associated shell script files

Files have some shared constants:
- "FOLDERPATH_DATA": directory where the data is stored
- "FOLDERPATH_PREPROCESSING": directory where the preprocessing settings are
  stored and where the preprocessed data will be stored (e.g. in "Project/
  Preprocessing")
- "FOLDERPATH_PROCESSING": directory where the processing settings are stored
  and where the processed data will be stored (e.g. in "Project/Processing")
- "DATASET": name of the cohort
- "PRESET": group of recordings to analyse (see "Project/Preprocessing/
  Settings/Generic/Data File Presets/README.txt" for details)
- "ANALYSIS": if a preprocessing file, type of preprocessing to perform (see
  "Project/Preprocessing/Settings/Generic/README.txt" for details), or if a
  processing file, type of processing to perform (see "Project/Processing/
  Settings/Generic/README.txt" for details)
- "SETTINGS": type of data to compare (e.g. OFF therapy vs. ON levodopa, OFF
  therapy vs. ON STN-DBS; see "Project/Preprocessing/Settings/Specific/
  README.txt" for details)
- "PREPROCESSING": type of preprocessing to use (see "Project/Preprocessing/
  Settings/Generic/README.txt" for details)
- "SUBJECT": subject ID
- "SESSION": session name
- "TASK": task name
- "ACQUISITION": acquisition name
- "RUN": run number

Constants marked with "# var" can be changed for analysis of different data or
different methods.

