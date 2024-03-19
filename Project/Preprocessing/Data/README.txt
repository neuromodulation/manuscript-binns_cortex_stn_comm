This is where the preprocessed data is stored.

Here, there is a folder for each cohort.
Within each cohort folder, there is a folder for each subject.
Within each subject folder, there is a folder for each recording session.
Within each session folder, the preprocessed data for that cohort, subject, and
session is stored as pkl files.

E.g.:
ECOG_LFP (cohort)
    - sub-01 (subject)
        - ses-MedOff (session)
            - for_power.pkl
            - for_connectivity.pkl

The exact file names will take the form "task-X_acq-X_run-X_preprocessed-Y-Z
.pkl", where "X" is information about the recording, "Y" is the type of data
being compared (medication or stimulation comparison), and "Z" is the specific
type of analysis that will be performed (e.g. power, connectivity, etc...).
E.g. "task-Rest_acq-StimOff_run-1_preprocessed-med_analysis-for_connectivity
.pkl"