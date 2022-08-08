# Thesis
Supervised machine learning techniques such as random forests, support vector machines and feedforward neural networks are often used in the context of clinical risk prediction. This thesis investigates whether or not there are substantial gains in performance over traditional statistical techniques like logistic regression in this context.

In terms of file structure, all the notebooks sits in the same level, with the principal data files (mortality.csv and mortality_large.csv) in a path ./data.

Should you wish to run the data reading yourself, the data is all available from the 2007-2008 NHANES wave linked: https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2007. The raw data files should be saved in the path ./data/raw_data_files

In terms of ordering the code, data_reading.py and data_reading_large.py read in the data, with data_reading_large.py reading in the data with all features. This code is optional, the two csvs (mortality.csv and mortality_large.csv) are the sole output and should be saved  be saved in the path a path ./data.
A complete list of the data files used is is provided in dataset_listing.md, as uploading all of these exceeds the data limit.

Following on from datareading, the bootstrapping python file creates the bootstrapped datasets and should be run second. Theses files perform the model building and optimisation on each of the bootstraps, and output predictions for each bootstrap to a csv.

After this, the model_building_X.py files can be run. This implement the different varieties of model building with different variants:
- model_building.py refers to the initial analysis with the strong number of features, and includes the training of the ensemble learner.
- model_building_nX.py refers to model building where the sample size n of the training data is equal to X.
- model_building_fX.py refers to model building where the number of randomly selected features f is equal to X.
- model_building_PCA.py refers to model building where the features underwent PCA and the top 30 PCs were used in model training.
- model_building_hX.py refers to model building where the number of random hyperparameter searches h of the training data is equal to X.

Following on from this, the model_discrimination_Y.py files can be run. These implement the evaluations of discriminatory performance and output figures to the filepath ./figures:
- model_discrimination.py creates discrimination figures for the initial analysis.
- model_discrimination_n.py creates discrimination figures when the sample size varies.
- model_discrimination_f.py creates discrimination figures when the feature count varies.
- model_discrimination_PCA.py creates discrimination figures when the features are the top 30 principal components.
- mode_discrimination_h.py creates discrimination figures when the hyperparameter optimisation varies.

Lastly, the model_calibration_Y.py files can be run. These implement the evaluations of calibration performance for each algorithm, again outputting figures to the filepath ./figures:
- model_calibration.py creates discrimination figures for the initial analysis.
- model_calibration_n.py creates discrimination figures when the sample size varies.
- model_calibration_f.py creates discrimination figures when the feature count varies.
- model_calibration_PCA.py creates discrimination figures when the features are the top 30 principal components.
- mode_calibration_h.py creates discrimination figures when the hyperparameter optimisation varies.





