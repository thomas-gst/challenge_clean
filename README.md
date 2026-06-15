# Library Presentation

The main code (models, training loops, loss, feature engineering) for the various models presented in the report are in the `src` file.
The scripts files are there to train a given model and create a submission once trained. They should be plug-and-play (given the right dependencies are installed, and the data has been correctly enriched by running `src/data/enrich.pyù before hand). Each script begins with the full list of model hyperparameters used for the late submissions discussed in the report. 

Are notably absent : the `train_submit` script and losses used for the competition model. This is due to the fact that this code library was a a re-write of my working library so that the code would be easier to read (and also to remove all uses of scrapped views), and I didn't not have the time to adapt these scripts to the new library, but as the model was essentially inept, I don't think these scripts will be missed. Also absent is the tag extraction script as it was not used in any of the models from the report. 

We've tried to add as many docstrings as possible in the various classes and methods to make the code readable. Further details and explanations are often given in these doc strings.

The data and model weight files were removed in order to comply with the file size requirements on moodle and github. Also removed where the notebooks for data analysis as most relevant plots were present in the report. 

# Main workflow for feature engineering

The `enrich.py` script from `src/data/` does the following :
- load orgininal `train_val.csv`, `test.csv` meta data into pandas dataframes 
- enrich using the Enricher class, differently for train_val and test, accorder to the `.enrich_train_val` and `.enrich_test` methods
- save enriched data to `.parquet` files in `data/processsed`


# Main workflow for training and evaluating models

Each script from scripts/ essentially does the following :
- Load the data in various pandas dataframes (train, val, test) from `.parquet` files in `data/processed/`.
- Load the model from `src/models/`, given a hyperparameter configuration.
- Load the loss function, optimizer and pass them to a `trainer`class (all relevant scripts are in `src/utils/`) given a training hyperparameters configuration 
- Train the model using the `.train` method on the trainer, while logging on wandb (if keyword argument `silent=False` on the trainer)
- Create a submission using the `.calculate_submission` method on the trainer

# Additional scripts 
Included in `utils/` are the bash script used for running jobs on the BR's remote GPUs, as well as the job launching cross validation script to parallelize runs on those same GPUs. 
I've redacted the server names for the public online repo. 
