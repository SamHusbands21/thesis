# This code iterates through 199 bootstraps, outputting predictions on the test set for 199 different bootstrapped datasets.
# This code executes with 100 training datapoints per ML algorithm
# These predictions are saved in./predictions/datapoint_results_100
import glob
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import random
import sklearn
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.stats import loguniform, uniform, randint
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(21)
tf.random.set_seed(21)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def min_max_scaling(df):
    # copy the dataframe
    df_norm = df.copy()
    # apply min-max scaling
    for column in df_norm.columns:
        if (df_norm[column].max() - df_norm[column].min())==0:
            df_norm[column] = 0
        else:
            df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())
    return df_norm

def logistic(X_train, X_test, y_train, y_test, prints = True):
    '''Takes given arguments and returns predictions and y_test,
       along with the auc for the logistic regression'''
    lr = LogisticRegression(max_iter = 10000, solver = 'lbfgs', n_jobs = 30)
    lr.fit(X_train, y_train)
    y_pred = lr.predict_proba(X_test)[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    auc_lr = auc(fpr, tpr)
    y_test = pd.DataFrame(y_test)
    pred_df = pd.DataFrame(y_pred)
    pred_df['index'] = y_test.index
    pred_df.set_index('index', inplace = True)
    pred_df.rename(columns = {0: "prob"}, inplace = True)
    if prints == True:
        print("LR AUC:", round(auc_lr, 3))
    return pred_df.join(y_test, how = 'left'), auc_lr

def hyperparameter_randomiser_rf(max_depth_range,
                                 max_features_range,
                                 min_leaf_range,
                                 n_trees_range,
                                 prints = False):
    '''
    Returns the best hyperparameters from given ranges using a random search algorithm
    (all searches use uniform distribution):
    max_depth_range: Range of max_depth
    max_features_range: Range of maximum features per split
    min_leaf_range: Range of minimum data points per leaf node
    n_trees_range: Range of number of trees in a constituent random forest
    prints: An option to facilitate the printing of the randomly selected hyperparameters at each search.
    '''
    max_depth = randint(max_depth_range[0], max_depth_range[1] + 1).rvs(1).item()
    max_features = randint(max_features_range[0], max_features_range[1] + 1).rvs(1).item()
    min_leaf = randint(min_leaf_range[0], min_leaf_range[1] + 1).rvs(1).item()
    n_trees = randint(n_trees_range[0], n_trees_range[1] + 1).rvs(1).item()
    if prints == True:
        print("max_depth:", max_depth)
        print("max_features:", max_features)
        print("min_leaf:", min_leaf)
        print("n_trees:", n_trees)
    return max_depth, max_features, min_leaf, n_trees

def build_model_rf(max_depth, max_features, min_leaf, n_trees):
    rf = RandomForestClassifier(n_estimators = n_trees,
                                max_depth = max_depth,
                                max_features = max_features,
                                min_samples_leaf = min_leaf)
    return rf


def hyp_tuner_rf(md_range,
                 mf_range,
                 ml_range,
                 nt_range,
                 X_train,
                 y_train,
                 X_test,
                 y_test,
                 mapper,
                 cv_folds=3,
                 random_search=10):
    '''Returns the best hyperparameters from given ranges using a random search algorithm:
        md_range: Range of max depth of consituent decision trees
        mf_range: Range of max features in each split
        X_train: The training dataset (as a pandas dataframe) features
        y_train: The training dataset (as a pandas dataframe) labels
        X_test: The testing dataset (as a pandas dataframe) features
        y_test: The testing dataset (as a pandas dataframe) labels
        mapper: The mapping of index to id
        cv_folds: Number of folds of cross validation (default 3)
        random_search: how many random searches are performed (default 5)
        '''
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=21)
    cv.get_n_splits(X_train, y_train)
    # splitting the dataset for k-fold
    md_list, mf_list, ml_list, nt_list = [], [], [], []
    auc_dict = dict()
    # creating dictionaries and lists that store information on hyperparameters and aucs
    for i in range(random_search):
        # randomly selecting hyperparameters and storing them
        md, mf, ml, nt = hyperparameter_randomiser_rf(max_depth_range=md_range,
                                                      max_features_range=mf_range,
                                                      min_leaf_range=ml_range,
                                                      n_trees_range=nt_range)
        md_list.append(md)
        mf_list.append(mf)
        ml_list.append(ml)
        nt_list.append(nt)
        auc_dict[i] = []
        for train_index, test_index in cv.split(X_train, y_train):
            # cross validating
            model = build_model_rf(max_depth=md,
                                   max_features=mf,
                                   min_leaf=ml,
                                   n_trees=nt)
            X1_train, X1_valid = X_train.iloc[train_index], X_train.iloc[test_index]
            y1_train, y1_valid = y_train.iloc[train_index], y_train.iloc[test_index]
            model.fit(X1_train, y1_train)
            y1_pred = model.predict_proba(X1_valid)[:, 1]
            fpr, tpr, threshold = roc_curve(y1_valid, y1_pred)
            cauc = auc(fpr, tpr)
            auc_dict[i].append(cauc)
    # analysing results to find best hyperparameters
    hp_dict = {"max_depth": md_list,
               "max_features": mf_list,
               "min_leaf": ml_list,
               "n_trees": nt_list}
    av_auc_dict = dict()
    for k, v in auc_dict.items():
        av_auc_dict[k] = sum(v) / len(v)
    opt_hyper = max(av_auc_dict, key=av_auc_dict.get)
    opt_md = hp_dict["max_depth"][opt_hyper]
    opt_mf = hp_dict["max_features"][opt_hyper]
    opt_ml = hp_dict["min_leaf"][opt_hyper]
    opt_nt = hp_dict["n_trees"][opt_hyper]
    # building the model based on the whole training data based on optimal parameters
    opt_model = build_model_rf(max_depth = opt_md,
                               max_features = opt_mf,
                               min_leaf = opt_ml,
                               n_trees = opt_nt)
    opt_model.fit(X_train, y_train)
    y_pred_train = opt_model.predict_proba(X_train)[:, 1]
    y_pred = opt_model.predict_proba(X_test)[:, 1]
    v_auc = roc_auc_score(y_test, y_pred)
    # sorting prediction and training dataframes
    y_test_df = pd.DataFrame(y_test)
    pred_test_df = pd.DataFrame(y_pred)
    pred_test_df['index'] = y_test.index
    pred_test_df.set_index('index', inplace=True)
    pred_test_df.rename(columns={0: "prob"}, inplace=True)
    test_df = pred_test_df.join(y_test_df, how='left')
    print("RF AUC:", round(v_auc, 3))
    #training_dataframe
    y_train_df = pd.DataFrame(y_train)
    pred_train_df = pd.DataFrame(y_pred_train)
    pred_train_df['index'] = y_train.index
    pred_train_df.set_index('index', inplace=True)
    pred_train_df.rename(columns={0: "prob"}, inplace=True)
    train_df = pred_train_df.join(y_train_df, how = 'left')
    train_df = train_df.merge(mapper, left_index = True, right_index = True)
    train_df.set_index('id',drop = True, inplace = True)
    test_df = test_df.merge(mapper, left_index = True, right_index = True)
    test_df.set_index('id',drop = True, inplace = True)
    return train_df, test_df

def hyperparameter_randomiser_svm(c_range,
                                  gamma_range,
                                  c_dist = "log_uniform",
                                  gamma_dist = "log_uniform",
                                  prints = False):
    '''
    Returns the best hyperparameters from given ranges using a random search algorithm:
    c_range: Upper and lower bounds of the c distribution range as a list
    gamma_range: Upper and lower bounds of the gamma distribution range as a list
    c_dist: An argument specifying whether the distribution is drawn from a uniform or log uniform distribution
            arguments: "log_uniform", "uniform"
    gamma_dist: An argument specifying whether the distribution is drawn from a uniform or log uniform distribution
                arguments: "log_uniform", "uniform"
    prints: An option to facilitate the printing of the randomly selected hyperparameters at each search.
    '''
    if c_dist=="log_uniform":
        c = loguniform(c_range[0], c_range[1]).rvs(1).item()
    elif c_dist=="uniform":
        c = uniform(c_range[0], c_range[1]).rvs(1).item()
    else:
        raise Exception("Please provide a valid c distribution.")
    if gamma_dist=="log_uniform":
        gamma = loguniform(gamma_range[0], gamma_range[1]).rvs(1).item()
    elif gamma_dist=="uniform":
        gamma = uniform(gamma_range[0], gamma_range[1]).rvs(1).item()
    else:
        raise Exception("Please provide a valid gamma distribution.")
    if prints == True:
        print("c:", c)
        print("gamma:", gamma)
    return c, gamma

def build_model_svm(ce, gamma):
    SVM = SVC(C = ce, gamma = gamma, kernel = 'rbf', probability = True)
    return SVM

def hyp_tuner_svm(cee_range,
                  gam_range,
                  X_train,
                  y_train,
                  X_test,
                  y_test,
                  mapper,
                  cv_folds = 3,
                  random_search = 10):
    '''Returns the best hyperparameters from given ranges using a random search algorithm:
        c_range: Range of max depth of consituent decision trees
        gamma_range: Range of max features in each split
        X_train: The training dataset (as a pandas dataframe) features
        y_train: The training dataset (as a pandas dataframe) labels
        X_test: The testing dataset (as a pandas dataframe) features
        y_test: The testing dataset (as a pandas dataframe) labels
        mapper: The mapping of index to id
        cv_folds: Number of folds of cross validation (default 3)
        random_search: how many random searches are performed (default 5)
        '''
    X_train, X_test = min_max_scaling(X_train), min_max_scaling(X_test)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=21)
    cv.get_n_splits(X_train, y_train)
    # splitting the dataset for k-fold
    c_list, gamma_list = [], []
    auc_dict = dict()
    # creating dictionaries and lists that store information on hyperparameters and aucs
    for i in range(random_search):
        # randomly selecting hyperparameters and storing them
        cee, gam = hyperparameter_randomiser_svm(c_range=cee_range,
                                                 gamma_range=gam_range)
        c_list.append(cee)
        gamma_list.append(gam)
        auc_dict[i] = []
        for train_index, test_index in cv.split(X_train, y_train):
            # cross validating
            model = build_model_svm(ce=cee,
                                    gamma=gam)
            X1_train, X1_valid = X_train.iloc[train_index], X_train.iloc[test_index]
            y1_train, y1_valid = y_train.iloc[train_index], y_train.iloc[test_index]
            model.fit(X1_train, y1_train)
            y1_pred = model.predict_proba(X1_valid)[:, 1]
            fpr, tpr, threshold = roc_curve(y1_valid, y1_pred)
            cauc = auc(fpr, tpr)
            auc_dict[i].append(cauc)
    # analysing results to find best hyperparameters
    hp_dict = {"c": c_list,
               "gamma": gamma_list}
    av_auc_dict = dict()
    for k, v in auc_dict.items():
        av_auc_dict[k] = sum(v) / len(v)
    opt_hyper = max(av_auc_dict, key=av_auc_dict.get)
    opt_c = hp_dict["c"][opt_hyper]
    opt_gamma = hp_dict["gamma"][opt_hyper]
    #building best model
    opt_model = build_model_svm(ce=opt_c,
                                gamma=opt_gamma)
    opt_model.fit(X_train, y_train)
    y_pred_train = opt_model.predict_proba(X_train)[:, 1]
    y_pred = opt_model.predict_proba(X_test)[:, 1]
    v_auc = roc_auc_score(y_test, y_pred)
    # sorting prediction and training dataframes
    y_test_df = pd.DataFrame(y_test)
    pred_test_df = pd.DataFrame(y_pred)
    pred_test_df['index'] = y_test.index
    pred_test_df.set_index('index', inplace=True)
    pred_test_df.rename(columns={0: "prob"}, inplace=True)
    test_df = pred_test_df.join(y_test_df, how='left')
    print("SVM AUC:", round(v_auc, 3))
    # training_dataframe
    y_train_df = pd.DataFrame(y_train)
    pred_train_df = pd.DataFrame(y_pred_train)
    pred_train_df['index'] = y_train.index
    pred_train_df.set_index('index', inplace=True)
    pred_train_df.rename(columns={0: "prob"}, inplace=True)
    train_df = pred_train_df.join(y_train_df, how='left')
    train_df = train_df.merge(mapper, left_index = True, right_index = True)
    train_df.set_index('id',drop = True, inplace = True)
    test_df = test_df.merge(mapper, left_index = True, right_index = True)
    test_df.set_index('id',drop = True, inplace = True)
    return train_df, test_df

def hyperparameter_randomiser_nn(al_range,
                                 dropout_rate_range,
                                 internal_nodes_range,
                                 learning_rate_range,
                                 prints = False):
    '''Randomises hyperparameters from a given list:
       al_range: The upper and lower bounds of the number of hidden layers
       dropout_rate_range: A list of dropout rates randomly chosen in each hidden layer
       internal_nodes_range: The upper and lower bounds for the number of internal nodes in each hidden layer
       learning_rate_range: The upper and lower bounds for the learning rate
       '''
    no_of_layers = randint(al_range[0], al_range[1] + 1).rvs(1).item()
    int_nodes = [randint(internal_nodes_range[0], internal_nodes_range[1] + 1).rvs(1).item() for i in range(no_of_layers)]
    dropout_rate = [np.random.choice(dropout_rate_range) for i in range(no_of_layers)]
    learning_rate = loguniform(learning_rate_range[0], learning_rate_range[1]).rvs(1).item()
    if prints == True:
        print("n_layers:", no_of_layers)
        print("learning_rate:", learning_rate)
        for i in range(no_of_layers):
            print(f"int_nodes_{i}:", int_nodes[i])
            print(f"dropout_rate_{i}:", dropout_rate[i])
    return no_of_layers, dropout_rate, int_nodes, learning_rate

def build_model_nn(no_of_layers, dropout_rate, int_nodes, learning_rate, feat_count):
    '''Builds the model based on hyperparameters provided
       no_of_layers: no of hidden layers in the model (must be greater than one)
       dropout_rate: The dropout rate in each hidden layer
       int_nodes: The number of internal nodes in each hidden layer
       learning_rate: The learning rate
       feat_count: number of input features
       '''
    #building architecture
    model = Sequential()
    #first_hidden_layer
    model.add(Dense(int_nodes[0], input_shape = (feat_count,), activation = 'relu'))
    model.add(Dropout(rate = dropout_rate[0]))
    #additional layers
    for i in range(no_of_layers - 1):
        model.add(Dense(int_nodes[i + 1], activation = 'relu'))
        model.add(Dropout(rate = dropout_rate[i + 1]))
    #output layer
    model.add(Dense(1, activation = 'sigmoid'))
    #compiling model
    model.compile(optimizer = Adam(learning_rate = learning_rate),
                  loss = "binary_crossentropy",
                  metrics = ["AUC"])
    return model


def hyp_tuner_nn(al_range,
                 dr_range,
                 lr_range,
                 in_range,
                 X_train,
                 y_train,
                 X_test,
                 y_test,
                 mapper,
                 cv_folds = 3,
                 random_search = 10):
    '''Returns the best hyperparameters from given ranges using a random search algorithm:
        al_range: Range of hidden layers
        dr_range: Range of dropout layers
        lr_range: Range of learning rate
        in_range: Range of number of internal nodes in a hidden layer
        X_train: The training dataset (as a pandas dataframe) features
        y_train: The training dataset (as a pandas dataframe) labels
        X_test: The testing dataset (as a pandas dataframe) features
        y_test: The testing dataset (as a pandas dataframe) labels
        mapper: The mapping of index to id
        cv_folds: Number of folds of cross validation (default 5)
        random_search: how many random searches are performed (default 20)
        '''
    X_train, X_test = min_max_scaling(X_train), min_max_scaling(X_test)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=21)
    cv.get_n_splits(X_train, y_train)
    # splitting the dataset for k-fold
    nl_list, dr_list, lr_list, in_list, ep_list = [], [], [], [], []
    auc_dict = dict()
    # creating dictionaries and lists that store information on hyperparameters and aucs
    pred_dict = dict()
    for i in range(random_search):
        # randomly selecting hyperparameters and storing them
        n_layers, dr_rate, int_nodes, learning_rate = hyperparameter_randomiser_nn(al_range=al_range,
                                                                                   dropout_rate_range=dr_range,
                                                                                   learning_rate_range=lr_range,
                                                                                   internal_nodes_range=in_range)
        nl_list.append(n_layers)
        dr_list.append(dr_rate)
        in_list.append(int_nodes)
        lr_list.append(learning_rate)
        auc_dict[i] = []
        epochs = []
        for train_index, test_index in cv.split(X_train, y_train):
            # cross validating
            model = build_model_nn(no_of_layers=n_layers,
                                   dropout_rate=dr_rate,
                                   int_nodes=int_nodes,
                                   learning_rate=learning_rate,
                                   feat_count=X_train.shape[1])
            X1_train, X1_valid = X_train.iloc[train_index], X_train.iloc[test_index]
            y1_train, y1_valid = y_train.iloc[train_index], y_train.iloc[test_index]
            hist = model.fit(X1_train.values,
                             y1_train.values,
                             batch_size=16,
                             shuffle=True,
                             validation_data=(X1_valid, y1_valid),
                             epochs=30,
                             verbose=0,
                             callbacks=EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True))
            epochs.append(len(hist.history['loss']))
            pred = model.predict(x=X1_valid, batch_size=16, verbose = 0)
            auc_dict[i].append(roc_auc_score(y1_valid, pred))
        epochs = max(epochs)
        ep_list.append(epochs)

    # analysing results to find best hyperparameters
    hp_dict = {"hidden_layers": nl_list,
               "dropout_rates": dr_list,
               "internal_nodes": in_list,
               "learning_rate": lr_list,
               "epochs": ep_list}
    av_auc_dict = dict()
    for k, v in auc_dict.items():
        av_auc_dict[k] = sum(v) / len(v)
    opt_hyper = max(av_auc_dict, key=av_auc_dict.get)
    opt_layers = hp_dict["hidden_layers"][opt_hyper]
    opt_dropout = hp_dict["dropout_rates"][opt_hyper]
    opt_internal = hp_dict["internal_nodes"][opt_hyper]
    opt_learning = hp_dict["learning_rate"][opt_hyper]
    opt_epoch = hp_dict["epochs"][opt_hyper]
    # building the model based on the whole training data based on optimal parameters
    opt_model = build_model_nn(no_of_layers=opt_layers,
                               dropout_rate=opt_dropout,
                               int_nodes=opt_internal,
                               learning_rate=opt_learning,
                               feat_count=X_train.shape[1])
    opt_model.fit(X_train.values,
                  y_train.values,
                  batch_size=16,
                  shuffle=True,
                  epochs=opt_epoch,
                  verbose=0)
    y_pred_train = opt_model.predict(x = X_train, batch_size=16, verbose = 0)
    y_pred = opt_model.predict(x = X_test, batch_size=16, verbose = 0)
    v_auc = roc_auc_score(y_test, y_pred)
    # sorting prediction and training dataframes
    y_test_df = pd.DataFrame(y_test)
    pred_test_df = pd.DataFrame(y_pred)
    pred_test_df['index'] = y_test.index
    pred_test_df.set_index('index', inplace=True)
    pred_test_df.rename(columns={0: "prob"}, inplace=True)
    test_df = pred_test_df.join(y_test_df, how='left')
    print("NN AUC:", round(v_auc, 3))
    #training_dataframe
    y_train_df = pd.DataFrame(y_train)
    pred_train_df = pd.DataFrame(y_pred_train)
    pred_train_df['index'] = y_train.index
    pred_train_df.set_index('index', inplace=True)
    pred_train_df.rename(columns={0: "prob"}, inplace=True)
    train_df = pred_train_df.join(y_train_df, how = 'left')
    train_df = train_df.merge(mapper, left_index = True, right_index = True)
    train_df.set_index('id',drop = True, inplace = True)
    test_df = test_df.merge(mapper, left_index = True, right_index = True)
    test_df.set_index('id',drop = True, inplace = True)
    return train_df, test_df

all_filenames = glob.glob('./data/bootstraps/*.{}'.format('csv'))
all_filenames = [s.replace('\\', '/') for s in all_filenames]
datapoint_count = 100
if os.path.exists('./predictions') == False:
    os.mkdir('./predictions')
if os.path.exists('./predictions/datapoint_results_{}'.format(str(datapoint_count))) == False:
    os.mkdir('./predictions/datapoint_results_{}'.format(str(datapoint_count)))
i = 0
#fitting the models to each bootstrap
np.random.seed(21)
random.seed(21)
logistic_regression = []
random_forest = []
support_vector = []
neural_network = []
ensemble_classifier = []
for filename in all_filenames:
    i += 1
    print(str(i) + " of " + str(len(all_filenames)))
    mf, mapp  = pd.read_csv(filename).drop('id', axis = 1), pd.read_csv(filename)['id']
    #sorting out train test split
    X_train, X_test, y_train, y_test = train_test_split(mf.drop('deceased', axis = 1),
                                                        mf['deceased'], test_size = len(mf.index) - datapoint_count,
                                                        random_state = 21,
                                                        stratify = mf['deceased'])

    lr_te_df, lr_auc = logistic(X_train,
                                X_test,
                                y_train,
                                y_test)

    rf_tr_df, rf_te_df = hyp_tuner_rf(md_range=[3, 8],
                                      mf_range=[1, 9],
                                      ml_range=[1, 5],
                                      nt_range=[100, 1000],
                                      X_train = X_train,
                                      y_train = y_train,
                                      X_test = X_test,
                                      y_test = y_test,
                                      mapper = mapp)

    sv_tr_df, sv_te_df = hyp_tuner_svm(cee_range = [0.001, 1000],
                                       gam_range = [0.001, 1000],
                                       X_train = X_train,
                                       y_train = y_train,
                                       X_test = X_test,
                                       y_test = y_test,
                                       mapper = mapp)

    nn_tr_df, nn_te_df = hyp_tuner_nn(al_range = [1, 5],
                                      dr_range = [i/10 for i in range (4)],
                                      lr_range = [0.0001, 0.1],
                                      in_range = [36, 360],
                                      X_train = X_train,
                                      y_train = y_train,
                                      X_test = X_test,
                                      y_test = y_test,
                                      mapper = mapp)

    ensemble_df_train = pd.concat([rf_tr_df.iloc[:,0], sv_tr_df.iloc[:,0], nn_tr_df], axis = 1).reset_index(drop = True)
    ensemble_df_train.columns = ['rf', 'sv', 'nn', 'deceased']
    ensemble_df_test = pd.concat([rf_te_df.iloc[:,0], sv_te_df.iloc[:,0], nn_te_df], axis = 1).reset_index(drop = True)
    ensemble_df_test.columns = ['rf', 'sv', 'nn', 'deceased']
    edf_X_train = ensemble_df_train.drop('deceased', axis = 1)
    edf_y_train = ensemble_df_train['deceased']
    edf_X_test = ensemble_df_test.drop('deceased', axis = 1)
    edf_y_test = ensemble_df_test['deceased']
    en_te_df, ensemble_auc = logistic(edf_X_train,
                                      edf_X_test,
                                      edf_y_train,
                                      edf_y_test,
                                      prints = False)
    print("EC AUC:", round(ensemble_auc, 3))
    logistic_regression.append(lr_te_df.reset_index(drop = True))
    random_forest.append(rf_te_df.reset_index(drop = True))
    support_vector.append(sv_te_df.reset_index(drop = True))
    neural_network.append(nn_te_df.reset_index(drop = True))
    ensemble_classifier.append(en_te_df.reset_index(drop = True))
pd.concat(logistic_regression, axis = 1).to_csv('./predictions/datapoint_results_{}/lr_predictions.csv'.format(datapoint_count), index = False)
pd.concat(random_forest, axis = 1).to_csv('./predictions/datapoint_results_{}/rf_predictions.csv'.format(datapoint_count), index = False)
pd.concat(support_vector, axis = 1).to_csv('./predictions/datapoint_results_{}/sv_predictions.csv'.format(datapoint_count), index = False)
pd.concat(neural_network, axis = 1).to_csv('./predictions/datapoint_results_{}/nn_predictions.csv'.format(datapoint_count), index = False)
pd.concat(ensemble_classifier, axis = 1).to_csv('./predictions/datapoint_results_{}/ec_predictions.csv'.format(datapoint_count), index = False)