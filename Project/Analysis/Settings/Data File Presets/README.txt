This is where information about the processed data to be analysed is stored.

Here, there is a json file for each group of recordings.
In each json file, the processed data to analyse are given as a list of
dictionaries with the following keys:
- "cohort": name of the cohort
- "sub": subject ID
- "ses": name of the session
- "task": name of the task
- "acq": name of the acquisition
- "run": number of the run

See "Example_Preset.json" for an example.