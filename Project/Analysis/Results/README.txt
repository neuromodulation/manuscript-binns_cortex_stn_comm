This is where the analysed data is stored.

Here, there is a folder for each cohort, or a folder called "multi" for results
from multiple cohorts.
Within each (multi-)cohort folder, there is a folder for each subject, or a
folder called "sub-multi" for results from multiple subjects.
Within each (multi-)subject folder, there is a folder for each recording
session, or a folder called "ses-multi" for results from multiple sessions.
Within each (multi-)session folder, the results for that cohort, subject, and
session are stored.

E.g.:
ECOG_LFP (cohort)
    - sub-multi (subject)
        - ses-multi (session)
            - pow_multitaper.pkl
            - con_imcoh.pkl

The exact file names will take the form "task-X_acq-X_run-X_Z.pkl", where "X"
is information about the recording, and "Z" is the type of analysis that has
been performed (e.g. power, connectivity, etc...). E.g. "task-Rest_acq-multi_
run-multi_con_imcoh_whole-MedOffOn_multi_sub.pkl".