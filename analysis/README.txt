Scripts for analysing processed data.

There are different files:
- "analyse_results": for running analyses listed in "Project/Analysis/Settings"
- "mic_patterns_fmri_maps": for analysing the relationship between MIC patterns
  and fMRI connectivity (also contains plotting of results)
- "mic_patterns_hyperdirect_fibres": for analysing the relationship between MIC
  patterns and hyperdirect pathway connectivity (also contains plotting of
  results)

Files have some shared constants:
- "FOLDERPATH_PROCESSING": directory where the processing settings are stored
  and where the processed data will be stored (e.g. in "Project/Processing")
- "FOLDERPATH_ANALYSIS": directory where the analysis settings are stored and
  where the results will be stored (e.g. in "Project/Analysis")
- "ANALYSIS": type of analysis to perform (see "Project/Analysis/Settings/
  README.txt" for details)
- "TO_ANALYSE": group of recordings to analyse (see "Project/Analysis/Settings/
  Data File Presets/README.txt" for details)

Constants marked with "# var" can be changed for analysis of different data or
different methods.