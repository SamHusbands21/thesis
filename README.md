# Thesis
Supervised machine learning techniques such as random forests, support vector machines and feedforward neural networks are often used in the context of clinical risk prediction. This thesis investigates whether or not there are substantial gains in performance over traditional statistical techniques like logistic regression in this context.

In terms of file structure, all the .py files should sit in the same level, with the principal data files (mortality.csv and mortality_large.csv) in a path ./data.

Should you wish to run the data reading yourself, the data is all available from the 2007-2008 NHANES wave linked: https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2007. The raw data files should be saved in the path ./data/raw_data_files

In terms of ordering the code, data_reading.py and data_reading_large.py read in the data, with data_reading_large.py reading in the data with all features. This code is optional, the two csvs (mortality.csv and mortality_large.csv) are the sole output and should be saved  be saved in the path a path ./data.
A complete list of the data files used is is provided in dataset_listing.md, as uploading all of these exceeds the github file size limit.

Following on from datareading, the bootstrapping python file creates the bootstrapped datasets and should be run second. Theses files perform the model building and optimisation on each of the bootstraps, and output predictions for each bootstrap to a csv.

After this, the model_building_X.py files can be run. This implement the different varieties of model building with different variants:
- model_building.py refers to the initial analysis with the strong number of features, and includes the training of the ensemble learner.
- model_building_sample.py refers to model building where the sample size n changes.
- model_building_features.py refers to model building where the number of randomly selected features changes
- model_building_pca.py refers to model building where the features underwent PCA and the top 30 PCs were used in model training.
- model_building_hyperparameter.py refers to model building where the number of random hyperparameter searches h of the training data is equal to X (note this was    dropped from the thesis but thought I'd leave it in here).

Following this, the calibration and discrimination files can be run. The calibration.R file implements the LOESS smoother, and is required for the model_calibration_plots.py and model_calibration_ici.py, which perform the plotting of the calibration and ici plots. The model_discrimination_roc.py and model_discrimination_AUC.py implement the ROC and AUC plots. All plots output figures to the filepath ./figures.

It should be noted that this code is really here to give a flavour of the code that was run in a more interpretable format. The real code run used more parallelisation to improve run time, at the expense of there being many more files and less interpretablity. Much of the model building utilised hex; https://hex.cs.bath.ac.uk/ in order to further improve run time. If there are any questions, please do email sam.husbands21@gmail.com.



