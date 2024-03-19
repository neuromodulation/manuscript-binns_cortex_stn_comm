This is where the recording-wise settings for preprocessing recordings is
stored.

Here, there is a folder for each cohort.
Within each cohort folder, there is a folder for each subject.
Within each subject folder, there is a folder for each recording session.
Within each session folder, the settings for preprocessing each recording are
stored as json files, as well as the artefact annotations for each recording as
csv files.

E.g.:
ECOG_LFP (cohort)
    - sub-01 (subject)
        - ses-MedOff (session)
            - annotation.json
            - med_analysis.json
            - stim_analysis.json
            - annotations.csv

The exact file names should take the form "task-X_acq-X_run-X_settings-Y.json",
where "X" is information about the recording, and "Y" is the type of data being
compared (medication or stimulation comparison). E.g. "task-Rest_acq-StimOff_
run-1_settings-med_analysis.json"

See "Example_Settings.json" for an example of what each json file can contain.

