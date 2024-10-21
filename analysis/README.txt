Scripts for analysing processed data.

There are different files:
- "analyse_results": for running analyses listed in "Project/Analysis/Settings"
- "compare_stim_amplitudes_recording_3mfu": for comparing the stimulation
  amplitudes for each subject during the recordings and in the 3-month
  follow-up reviews
- "compare_tde_med_stim_effects": for analysing whether the effects of
  medication and stimulation are significantly different on cortex -> STN time
  delays
- "convert_mic_patterns_to_csv": for extracting MIC patterns from analysed
  results which can be subsequently read in MATLAB
- "convert_ssd_patterns_to_csv": for extracting SSD patterns from analysed
  results which can be subsequently read in MATLAB
- "distance_to_hand_knob": for finding the distance between ECoG contacts and
  the motor cortex hand-knob area
- "mic_patterns_fmri_maps": for analysing the relationship between MIC patterns
  and fMRI connectivity (also contains plotting of results for Figure 4b)
- "mic_patterns_hyperdirect_fibres": for analysing the relationship between MIC
  patterns and hyperdirect pathway connectivity (also contains plotting of
  results for Figure 4a)

Files have some shared constants:
- "FOLDERPATH_PROCESSING": directory where the processing settings are stored
  and where the processed data will be stored (e.g. in "Project/Processing")
- "FOLDERPATH_ANALYSIS": directory where the analysis settings are stored and
  where the results will be stored (e.g. in "Project/Analysis")
- "FILEPATH_COORDS": path to the file containing ECoG contact coordinates.
- "ANALYSIS": type of analysis to perform (see "Project/Analysis/Settings/
  README.txt" for details)
- "TO_ANALYSE": group of recordings to analyse (see "Project/Analysis/Settings/
  Data File Presets/README.txt" for details)

Constants marked with "# var" can be changed for analysis of different data or
different methods.