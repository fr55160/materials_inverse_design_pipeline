This folder is dedicated to the training of XGBoost models (pkl), one per target.

The first step is to identify the most promising descriptors for each target. It is done through 1-Descriptors_Selection.py. The outputs are data files and graphs.
'source_names' have to be adapted with real names (lines 54 to 59).
The loop has to be complete (lines 324 and 325).
The output will be saved in the new folder 'feature_selection_results'.

The second step consists in "hybrid Learning":
* A coarse tuning to reduce the parameters space
* A fine tuning
* A comparison to history to ensure that the training achieved better results than previously
If so, pkl models are saved.
Before running the script, the user must adapt:
  * the 'source_names' (lines 95 to 100)
  * the value of 'cases_to_run' (likely [1,2,3,4,5,6,7,8]; line 179)
  * Delete / comment the temporary line 'MODEL_NAME = "Hephaistos_test_on_HT_Oxidation.pkl"'