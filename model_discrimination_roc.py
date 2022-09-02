#this code produces the comparison ROC plots
# #packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
import os
pd.set_option('display.float_format', lambda x: '%.4f' % x)

#sorting folders
if os.path.exists('./figures') == False:
    os.mkdir('./figures')
if os.path.exists('./figures/initial_analysis') == False:
    os.mkdir('./figures/initial_analysis')
if os.path.exists('./figures/examples') == False:
    os.mkdir('./figures/examples')
if os.path.exists('./figures/features') == False:
    os.mkdir('./figures/features')
if os.path.exists('./figures/samples') == False:
    os.mkdir('./figures/samples')
if os.path.exists('./figures/pca') == False:
    os.mkdir('./figures/pca')
if os.path.exists('./figures/hyperparameters') == False:
    os.mkdir('./figures/hyperparameters')

def get_true_positive(df, fpr):
    '''Takes a false positive rate and returns a true positive rate for a dataframe
       with false and true positive rates:
       df: a dataframe with two columns; false positive rate (fpr) and true positive rate (tpr)
       fpr: a scalar between 0 and 1 (inclusive)
       '''
    temp_df = df[(df['fpr'] - fpr) <= 0].sort_values('tpr', ascending = False)
    return temp_df.iloc[0, 1]

def get_fpr_tpr(stoch_df, fpr = np.arange(0, 1.01, 0.01).tolist()):
    '''Takes a dataframe with threshold false positive and true positive rates, and standardises it for plotting
       such that the false positive rate is a list argument that increases in any increment:
       df: a dataframe with two columns; false positive rate (fpr) and true positive rate (tpr)
       fpr: a list of increments
       '''
    tpr = []
    for i in fpr:
        tpr.append(get_true_positive(stoch_df, i))
    final_df = pd.DataFrame(list(zip(fpr, tpr)),
               columns = list(stoch_df.columns))
    return final_df

def roc_auc(df, prints = False):
    '''Takes a long form dataframe of many bootstrap resamples and gives back fpr and tpr for each resample,
       as well as the area under the receiver operator curve (AUC)
       df: a dataframe with bootstap_i_pred and bootstrap_i_true as columns
       prints: an option to print progress'''
    aucs, data, cleaned_data = [], [], []
    for i in range(len(df.columns)//2):
        fpr, tpr, thr = roc_curve(df.iloc[:, 2*i + 1], df.iloc[:, 2*i])
        feeder_auc = auc(fpr, tpr)
        aucs.append(feeder_auc)
        dct = {'fpr'.format(i): fpr,
               'tpr'.format(i): tpr}
        data.append(pd.DataFrame(dct))
    np_aucs = np.array(aucs)
    lb = np.percentile(np_aucs, 2.5)
    ub = np.percentile(np_aucs, 97.5)
    j = 0
    for sto_df in data:
        cleaned_data.append(get_fpr_tpr(sto_df).set_index('fpr'))
        j += 1
        if prints==True:
            print(str(j) + " of " + str(len(df.columns)//2))
    raw_data = pd.concat(cleaned_data, axis = 1)
    summary_dic = {'median': raw_data.median(axis = 1),
                   'lb': raw_data.quantile(q = 0.025, axis = 1),
                   'ub': raw_data.quantile(q = 0.975, axis = 1)}
    return lb, ub, pd.DataFrame(summary_dic), np_aucs

#data reading
lr = pd.read_csv('./predictions/initial_results/lr_predictions.csv')
rf = pd.read_csv('./predictions/initial_results/rf_predictions.csv')
sv = pd.read_csv('./predictions/initial_results/sv_predictions.csv')
nn = pd.read_csv('./predictions/initial_results/nn_predictions.csv')
ec = pd.read_csv('./predictions/initial_results/ec_predictions.csv')

#actioning the bootstrap
lr_tuple = roc_auc(lr)
rf_tuple = roc_auc(rf)
sv_tuple = roc_auc(sv)
nn_tuple = roc_auc(nn)
ec_tuple = roc_auc(ec)

#creating the example_roc and roc_curves for the initial analysis
lr_df, rf_df, sv_df, nn_df, ec_df = lr_tuple[2], rf_tuple[2], sv_tuple[2], nn_tuple[2], ec_tuple[2]
plt.figure(figsize = (10,5), dpi = 1800)
plt.step(lr_df.index, lr_df['median'],
         color = '#e2ac04',
         label = "Logistic Regression")
plt.fill_between(x = lr_df.index,
                 y1 = lr_df['lb'],
                 y2 = lr_df['ub'],
                 color = '#e2ac04',
                 alpha = 0.2,
                 label = 'Logistic Regression 95% Confidence Interval')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Example Receiver Operator Curve: Logistic Regression')
plt.savefig('./figures/examples/example_roc.png',dpi=1800,transparent=True)
plt.clf()

plt.figure(figsize = (10,5), dpi = 1800)
plt.step(lr_df.index, lr_df['median'],
         color = '#e2ac04',
         label = "Logistic Regression")
plt.step(rf_df.index, rf_df['median'],
         color = '#1c9c74',
         label = "Random Forest")
plt.fill_between(x = lr_df.index,
                 y1 = lr_df['lb'],
                 y2 = lr_df['ub'],
                 color = '#e2ac04',
                 alpha = 0.2,
                 label = 'Logistic Regression 95% Confidence Interval')
plt.fill_between(x = rf_df.index,
                 y1 = rf_df['lb'],
                 y2 = rf_df['ub'],
                 color = '#1c9c74',
                 alpha = 0.2,
                 label = 'Random Forest 95% Confidence Interval')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 'lower right')
plt.title('Receiver Operator Curve: Logistic Regression vs Random Forest')
plt.savefig('./figures/initial_analysis/roc_lr_rf.png',dpi=1800,transparent=True)
plt.clf()

plt.figure(figsize = (10,5), dpi = 1800)
plt.step(lr_df.index, lr_df['median'],
         color = '#e2ac04',
         label = "Logistic Regression".format(lr_tuple[0], lr_tuple[1]))
plt.step(sv_df.index, sv_df['median'],
         color = '#d95f02',
         label = "Support Vector Machine".format(sv_tuple[0], sv_tuple[1]))
plt.fill_between(x = lr_df.index,
                 y1 = lr_df['lb'],
                 y2 = lr_df['ub'],
                 color = '#e2ac04',
                 alpha = 0.2,
                 label = 'Logistic Regression 95% Confidence Interval')
plt.fill_between(x = sv_df.index,
                 y1 = sv_df['lb'],
                 y2 = sv_df['ub'],
                 color = '#d95f02',
                 alpha = 0.2,
                 label = 'Support Vector Machine 95% Confidence Interval')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 'lower right')
plt.title('Receiver Operator Curve: Logistic Regression vs Support Vector Machine')
plt.savefig('./figures/initial_analysis/roc_lr_sv.png',dpi=1800,transparent=True)
plt.clf()

plt.figure(figsize = (10,5), dpi = 1800)
plt.step(lr_df.index, lr_df['median'],
         color = '#e2ac04',
         label = "Logistic Regression".format(lr_tuple[0], lr_tuple[1]))
plt.step(nn_df.index, nn_df['median'],
         color = '#7474b4',
         label = "Neural Network".format(nn_tuple[0], nn_tuple[1]))
plt.fill_between(x = lr_df.index,
                 y1 = lr_df['lb'],
                 y2 = lr_df['ub'],
                 color = '#e2ac04',
                 alpha = 0.2,
                 label = 'Logistic Regression 95% Confidence Interval')
plt.fill_between(x = nn_df.index,
                 y1 = nn_df['lb'],
                 y2 = nn_df['ub'],
                 color = '#7474b4',
                 alpha = 0.2,
                 label = 'Neural Network 95% Confidence Interval')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 'lower right')
plt.title('Receiver Operator Curve: Logistic Regression vs Neural Network')
plt.savefig('./figures/initial_analysis/roc_lr_nn.png',dpi=1800,transparent=True)
plt.clf()

plt.figure(figsize = (10,5), dpi = 1800)
plt.step(lr_df.index, lr_df['median'],
         color = '#e2ac04',
         label = "Logistic Regression")
plt.step(ec_df.index, ec_df['median'],
         color = '#e72a8a',
         label = "Ensemble")
plt.fill_between(x = lr_df.index,
                 y1 = lr_df['lb'],
                 y2 = lr_df['ub'],
                 color = '#e2ac04',
                 alpha = 0.2,
                 label = 'Logistic Regression 95% Confidence Interval')
plt.fill_between(x = ec_df.index,
                 y1 = ec_df['lb'],
                 y2 = ec_df['ub'],
                 color = '#e72a8a',
                 alpha = 0.2,
                 label = 'Ensemble 95% Confidence Interval')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 'lower right')
plt.title('Receiver Operator Curve: Logistic Regression vs Ensemble')
plt.savefig('./figures/initial_analysis/roc_lr_ec.png',dpi=1800,transparent=True)
plt.clf()








































