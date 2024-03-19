This is where the processed data is stored.

Here, there is a folder for each cohort.
Within each cohort folder, there is a folder for each subject.
Within each subject folder, there is a folder for each recording session.
Within each session folder, the processed data for that cohort, subject, and
session is stored as pkl files.

E.g.:
ECOG_LFP (cohort)
    - sub-01 (subject)
        - ses-MedOff (session)
            - pow_multitaper.pkl
            - con_imcoh.pkl

The exact file names will take the form "task-X_acq-X_run-X_Z-Y-Z.pkl", where
"X" is information about the recording, "Y" is the type of data being compared
(medication or stimulation comparison), and "Z" is the type of analysis that
has been performed (e.g. power, connectivity, etc...). E.g. "task-Rest_acq-
StimOff_run-1_connectivity-med_analysis-con_imcoh.pkl"