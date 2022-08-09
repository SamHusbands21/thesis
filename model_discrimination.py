#this code produces the AUC comparison and ROC plots
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

#making AUC comparison figures - initial analysis
rf_diff = rf_tuple[3] - lr_tuple[3]
sv_diff = sv_tuple[3] - lr_tuple[3]
nn_diff = nn_tuple[3] -lr_tuple[3]
ec_diff = ec_tuple[3] - lr_tuple[3]
rf_df = pd.DataFrame(rf_diff)
sv_df = pd.DataFrame(sv_diff)
nn_df = pd.DataFrame(nn_diff)
ec_df = pd.DataFrame(ec_diff)
rf_df['classifier'] = "Random Forest"
sv_df['classifier'] = "SVM"
nn_df["classifier"] = "Neural Network"
ec_df['classifier'] = 'Ensemble'
rf_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
sv_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
nn_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
ec_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
col_palette = {"Random Forest" : "firebrick",
               "SVM" : "gold",
               "Neural Network" : "darkorange"}
plt.figure(figsize = (10,5), dpi = 1800)
boxplot_df = pd.concat([rf_df, sv_df, nn_df], axis = 0)
sns.boxplot(x = "classifier", y = "AUC_differential", data = boxplot_df, whis = [2.5, 97.5], palette = col_palette)
plt.ylim(-0.15, 0.15)
plt.axhline(0, linewidth=2, linestyle = '-', color='r')
plt.xlabel('Machine Learning Algorithm')
plt.ylabel('Difference in AUC (ML - LR)')
plt.title('AUC comparison')
plt.savefig('./figures/initial_analysis/AUC_comparison_1.png',dpi=1800,transparent=True)
plt.clf()

rf_df = pd.DataFrame(rf_diff)
sv_df = pd.DataFrame(sv_diff)
nn_df = pd.DataFrame(nn_diff)
ec_df = pd.DataFrame(ec_diff)
rf_df['classifier'] = "Random Forest"
sv_df['classifier'] = "SVM"
nn_df["classifier"] = "Neural Network"
ec_df['classifier'] = 'Ensemble'
rf_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
sv_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
nn_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
ec_df.rename(columns = {0: 'AUC_differential'}, inplace = True)

col_palette = {"Random Forest" : "firebrick",
               "SVM" : "gold",
               "Neural Network" : "darkorange",
               "Ensemble" : "indigo"}
plt.figure(figsize = (10,5), dpi = 1800)
boxplot_df = pd.concat([rf_df, sv_df, nn_df, ec_df], axis = 0)
sns.boxplot(x = "classifier", y = "AUC_differential", data = boxplot_df, whis = [2.5, 97.5], palette = col_palette)
plt.ylim(-0.15, 0.15)
plt.axhline(0, linewidth=2, linestyle = '-', color='r')
plt.xlabel('Machine Learning Algorithm')
plt.ylabel('Difference in AUC (ML - LR)')
plt.title('AUC comparison')
plt.savefig('./figures/initial_analysis/AUC_comparison_2.png',dpi=1800,transparent=True)
plt.clf()

#creating the example_roc and roc_curves for the initial analysis
lr_df, rf_df, sv_df, nn_df, ec_df = lr_tuple[2], rf_tuple[2], sv_tuple[2], nn_tuple[2], ec_tuple[2]
plt.figure(figsize = (10,5), dpi = 1800)
plt.step(lr_df.index, lr_df['median'],
         color = 'deepskyblue',
         label = "Logistic Regression (AUC: [{:1.3f}, {:1.3f}])".format(lr_tuple[0], lr_tuple[1]))
plt.fill_between(x = lr_df.index,
                 y1 = lr_df['lb'],
                 y2 = lr_df['ub'],
                 color = 'deepskyblue',
                 alpha = 0.2,
                 label = 'Logistic Regression 95% Confidence Interval')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Example Receiver Operator Curve: Logistic Regression')
plt.savefig('./figures/examples/example_roc.png',dpi=1800,transparent=True)
plt.clf()

plt.figure(figsize = (10,5), dpi = 1800)
plt.step(lr_df.index, lr_df['median'],
         color = 'deepskyblue',
         label = "Logistic Regression")
plt.step(rf_df.index, rf_df['median'],
         color = 'firebrick',
         label = "Random Forest")
plt.fill_between(x = lr_df.index,
                 y1 = lr_df['lb'],
                 y2 = lr_df['ub'],
                 color = 'deepskyblue',
                 alpha = 0.2,
                 label = 'Logistic Regression 95% Confidence Interval')
plt.fill_between(x = rf_df.index,
                 y1 = rf_df['lb'],
                 y2 = rf_df['ub'],
                 color = 'firebrick',
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
         color = 'deepskyblue',
         label = "Logistic Regression".format(lr_tuple[0], lr_tuple[1]))
plt.step(sv_df.index, sv_df['median'],
         color = 'gold',
         label = "Support Vector Machine".format(sv_tuple[0], sv_tuple[1]))
plt.fill_between(x = lr_df.index,
                 y1 = lr_df['lb'],
                 y2 = lr_df['ub'],
                 color = 'deepskyblue',
                 alpha = 0.2,
                 label = 'Logistic Regression 95% Confidence Interval')
plt.fill_between(x = sv_df.index,
                 y1 = sv_df['lb'],
                 y2 = sv_df['ub'],
                 color = 'gold',
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
         color = 'deepskyblue',
         label = "Logistic Regression".format(lr_tuple[0], lr_tuple[1]))
plt.step(nn_df.index, nn_df['median'],
         color = 'darkorange',
         label = "Neural Network".format(nn_tuple[0], nn_tuple[1]))
plt.fill_between(x = lr_df.index,
                 y1 = lr_df['lb'],
                 y2 = lr_df['ub'],
                 color = 'deepskyblue',
                 alpha = 0.2,
                 label = 'Logistic Regression 95% Confidence Interval')
plt.fill_between(x = nn_df.index,
                 y1 = nn_df['lb'],
                 y2 = nn_df['ub'],
                 color = 'darkorange',
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
         color = 'deepskyblue',
         label = "Logistic Regression")
plt.step(ec_df.index, ec_df['median'],
         color = 'indigo',
         label = "Ensemble")
plt.fill_between(x = lr_df.index,
                 y1 = lr_df['lb'],
                 y2 = lr_df['ub'],
                 color = 'deepskyblue',
                 alpha = 0.2,
                 label = 'Logistic Regression 95% Confidence Interval')
plt.fill_between(x = ec_df.index,
                 y1 = ec_df['lb'],
                 y2 = ec_df['ub'],
                 color = 'indigo',
                 alpha = 0.2,
                 label = 'Ensemble 95% Confidence Interval')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 'lower right')
plt.title('Receiver Operator Curve: Logistic Regression vs Ensemble')
plt.savefig('./figures/initial_analysis/roc_lr_ec.png',dpi=1800,transparent=True)
plt.clf()

#sample code
#data reading
lr_100 = pd.read_csv('./predictions/datapoint_results_100/lr_predictions.csv')
lr_200 = pd.read_csv('./predictions/datapoint_results_200/lr_predictions.csv')
lr_400 = pd.read_csv('./predictions/datapoint_results_400/lr_predictions.csv')
lr_800 = pd.read_csv('./predictions/datapoint_results_800/lr_predictions.csv')
lr_1600 = pd.read_csv('./predictions/datapoint_results_1600/lr_predictions.csv')
lr_3200 = pd.read_csv('./predictions/datapoint_results_3200/lr_predictions.csv')
rf_100 = pd.read_csv('./predictions/datapoint_results_100/rf_predictions.csv')
rf_200 = pd.read_csv('./predictions/datapoint_results_200/rf_predictions.csv')
rf_400 = pd.read_csv('./predictions/datapoint_results_400/rf_predictions.csv')
rf_800 = pd.read_csv('./predictions/datapoint_results_800/rf_predictions.csv')
rf_1600 = pd.read_csv('./predictions/datapoint_results_1600/rf_predictions.csv')
rf_3200 = pd.read_csv('./predictions/datapoint_results_3200/rf_predictions.csv')
sv_100 = pd.read_csv('./predictions/datapoint_results_100/sv_predictions.csv')
sv_200 = pd.read_csv('./predictions/datapoint_results_200/sv_predictions.csv')
sv_400 = pd.read_csv('./predictions/datapoint_results_400/sv_predictions.csv')
sv_800 = pd.read_csv('./predictions/datapoint_results_800/sv_predictions.csv')
sv_1600 = pd.read_csv('./predictions/datapoint_results_1600/sv_predictions.csv')
sv_3200 = pd.read_csv('./predictions/datapoint_results_3200/sv_predictions.csv')
nn_100 = pd.read_csv('./predictions/datapoint_results_100/nn_predictions.csv')
nn_200 = pd.read_csv('./predictions/datapoint_results_200/nn_predictions.csv')
nn_400 = pd.read_csv('./predictions/datapoint_results_400/nn_predictions.csv')
nn_800 = pd.read_csv('./predictions/datapoint_results_800/nn_predictions.csv')
nn_1600 = pd.read_csv('./predictions/datapoint_results_1600/nn_predictions.csv')
nn_3200 = pd.read_csv('./predictions/datapoint_results_3200/nn_predictions.csv')
ec_100 = pd.read_csv('./predictions/datapoint_results_100/ec_predictions.csv')
ec_200 = pd.read_csv('./predictions/datapoint_results_200/ec_predictions.csv')
ec_400 = pd.read_csv('./predictions/datapoint_results_400/ec_predictions.csv')
ec_800 = pd.read_csv('./predictions/datapoint_results_800/ec_predictions.csv')
ec_1600 = pd.read_csv('./predictions/datapoint_results_1600/ec_predictions.csv')
ec_3200 = pd.read_csv('./predictions/datapoint_results_3200/ec_predictions.csv')

#actioning the bootstrap
lr_100_tuple = roc_auc(lr_100)
lr_200_tuple = roc_auc(lr_200)
lr_400_tuple = roc_auc(lr_400)
lr_800_tuple = roc_auc(lr_800)
lr_1600_tuple = roc_auc(lr_1600)
lr_3200_tuple = roc_auc(lr_3200)
rf_100_tuple = roc_auc(rf_100)
rf_200_tuple = roc_auc(rf_200)
rf_400_tuple = roc_auc(rf_400)
rf_800_tuple = roc_auc(rf_800)
rf_1600_tuple = roc_auc(rf_1600)
rf_3200_tuple = roc_auc(rf_3200)
sv_100_tuple = roc_auc(sv_100)
sv_200_tuple = roc_auc(sv_200)
sv_400_tuple = roc_auc(sv_400)
sv_800_tuple = roc_auc(sv_800)
sv_1600_tuple = roc_auc(sv_1600)
sv_3200_tuple = roc_auc(sv_3200)
nn_100_tuple = roc_auc(nn_100)
nn_200_tuple = roc_auc(nn_200)
nn_400_tuple = roc_auc(nn_400)
nn_800_tuple = roc_auc(nn_800)
nn_1600_tuple = roc_auc(nn_1600)
nn_3200_tuple = roc_auc(nn_3200)
ec_100_tuple = roc_auc(ec_100)
ec_200_tuple = roc_auc(ec_200)
ec_400_tuple = roc_auc(ec_400)
ec_800_tuple = roc_auc(ec_800)
ec_1600_tuple = roc_auc(ec_1600)
ec_3200_tuple = roc_auc(ec_3200)

#diff in performance
rf_100_diff = rf_100_tuple[3] - lr_100_tuple[3]
sv_100_diff = sv_100_tuple[3] - lr_100_tuple[3]
nn_100_diff = nn_100_tuple[3] -lr_100_tuple[3]
ec_100_diff = ec_100_tuple[3] - lr_100_tuple[3]
rf_200_diff = rf_200_tuple[3] - lr_200_tuple[3]
sv_200_diff = sv_200_tuple[3] - lr_200_tuple[3]
nn_200_diff = nn_200_tuple[3] -lr_200_tuple[3]
ec_200_diff = ec_200_tuple[3] - lr_200_tuple[3]
rf_400_diff = rf_400_tuple[3] - lr_400_tuple[3]
sv_400_diff = sv_400_tuple[3] - lr_400_tuple[3]
nn_400_diff = nn_400_tuple[3] -lr_400_tuple[3]
ec_400_diff = ec_400_tuple[3] - lr_400_tuple[3]
rf_800_diff = rf_800_tuple[3] - lr_800_tuple[3]
sv_800_diff = sv_800_tuple[3] - lr_800_tuple[3]
nn_800_diff = nn_800_tuple[3] -lr_800_tuple[3]
ec_800_diff = ec_800_tuple[3] - lr_800_tuple[3]
rf_1600_diff = rf_1600_tuple[3] - lr_1600_tuple[3]
sv_1600_diff = sv_1600_tuple[3] - lr_1600_tuple[3]
nn_1600_diff = nn_1600_tuple[3] -lr_1600_tuple[3]
ec_1600_diff = ec_1600_tuple[3] - lr_1600_tuple[3]
rf_3200_diff = rf_3200_tuple[3] - lr_3200_tuple[3]
sv_3200_diff = sv_3200_tuple[3] - lr_3200_tuple[3]
nn_3200_diff = nn_3200_tuple[3] -lr_3200_tuple[3]
ec_3200_diff = ec_3200_tuple[3] - lr_3200_tuple[3]

#AUC differentials as sample increases - plots
rf_df = pd.DataFrame(rf_100_diff)
sv_df = pd.DataFrame(sv_100_diff)
nn_df = pd.DataFrame(nn_100_diff)
ec_df = pd.DataFrame(ec_100_diff)
rf_df['classifier'] = "Random Forest"
sv_df['classifier'] = "SVM"
nn_df["classifier"] = "Neural Network"
ec_df['classifier'] = 'Ensemble'
rf_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
sv_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
nn_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
ec_df.rename(columns = {0: 'AUC_differential'}, inplace = True)

col_palette = {"Random Forest" : "firebrick",
               "SVM" : "gold",
               "Neural Network" : "darkorange",
               "Ensemble" : "indigo"}
plt.figure(figsize = (10,5), dpi = 1800)
boxplot_df = pd.concat([rf_df, sv_df, nn_df, ec_df], axis = 0)
sns.boxplot(x = "classifier", y = "AUC_differential", data = boxplot_df, whis = [2.5, 97.5], palette = col_palette)
plt.axhline(0, linewidth=2, linestyle = '-', color='r')
plt.xlabel('Machine Learning Algorithm')
plt.ylabel('Difference in AUC (ML - LR)')
plt.title('AUC Comparison: Sample = 100')
plt.savefig('./figures/samples/AUC_comparison_sample100.png',dpi=1800,transparent=True)
plt.clf()

rf_df = pd.DataFrame(rf_200_diff)
sv_df = pd.DataFrame(sv_200_diff)
nn_df = pd.DataFrame(nn_200_diff)
ec_df = pd.DataFrame(ec_200_diff)
rf_df['classifier'] = "Random Forest"
sv_df['classifier'] = "SVM"
nn_df["classifier"] = "Neural Network"
ec_df['classifier'] = 'Ensemble'
rf_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
sv_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
nn_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
ec_df.rename(columns = {0: 'AUC_differential'}, inplace = True)

col_palette = {"Random Forest" : "firebrick",
               "SVM" : "gold",
               "Neural Network" : "darkorange",
               "Ensemble" : "indigo"}
plt.figure(figsize = (10,5), dpi = 1800)
boxplot_df = pd.concat([rf_df, sv_df, nn_df, ec_df], axis = 0)
sns.boxplot(x = "classifier", y = "AUC_differential", data = boxplot_df, whis = [2.5, 97.5], palette = col_palette)
plt.axhline(0, linewidth=2, linestyle = '-', color='r')
plt.xlabel('Machine Learning Algorithm')
plt.ylabel('Difference in AUC (ML - LR)')
plt.title('AUC Comparison: Sample = 200')
plt.savefig('./figures/samples/AUC_comparison_sample200.png',dpi=1800,transparent=True)
plt.clf()

rf_df = pd.DataFrame(rf_400_diff)
sv_df = pd.DataFrame(sv_400_diff)
nn_df = pd.DataFrame(nn_400_diff)
ec_df = pd.DataFrame(ec_400_diff)
rf_df['classifier'] = "Random Forest"
sv_df['classifier'] = "SVM"
nn_df["classifier"] = "Neural Network"
ec_df['classifier'] = 'Ensemble'
rf_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
sv_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
nn_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
ec_df.rename(columns = {0: 'AUC_differential'}, inplace = True)

col_palette = {"Random Forest" : "firebrick",
               "SVM" : "gold",
               "Neural Network" : "darkorange",
               "Ensemble" : "indigo"}
plt.figure(figsize = (10,5), dpi = 1800)
boxplot_df = pd.concat([rf_df, sv_df, nn_df, ec_df], axis = 0)
sns.boxplot(x = "classifier", y = "AUC_differential", data = boxplot_df, whis = [2.5, 97.5], palette = col_palette)
plt.axhline(0, linewidth=2, linestyle = '-', color='r')
plt.xlabel('Machine Learning Algorithm')
plt.ylabel('Difference in AUC (ML - LR)')
plt.title('AUC Comparison: Sample = 400')
plt.savefig('./figures/samples/AUC_comparison_sample400.png',dpi=1800,transparent=True)
plt.clf()

rf_df = pd.DataFrame(rf_800_diff)
sv_df = pd.DataFrame(sv_800_diff)
nn_df = pd.DataFrame(nn_800_diff)
ec_df = pd.DataFrame(ec_800_diff)
rf_df['classifier'] = "Random Forest"
sv_df['classifier'] = "SVM"
nn_df["classifier"] = "Neural Network"
ec_df['classifier'] = 'Ensemble'
rf_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
sv_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
nn_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
ec_df.rename(columns = {0: 'AUC_differential'}, inplace = True)

col_palette = {"Random Forest" : "firebrick",
               "SVM" : "gold",
               "Neural Network" : "darkorange",
               "Ensemble" : "indigo"}
plt.figure(figsize = (10,5), dpi = 1800)
boxplot_df = pd.concat([rf_df, sv_df, nn_df, ec_df], axis = 0)
sns.boxplot(x = "classifier", y = "AUC_differential", data = boxplot_df, whis = [2.5, 97.5], palette = col_palette)
plt.axhline(0, linewidth=2, linestyle = '-', color='r')
plt.xlabel('Machine Learning Algorithm')
plt.ylabel('Difference in AUC (ML - LR)')
plt.title('AUC Comparison: Sample = 800')
plt.savefig('./figures/samples/AUC_comparison_sample800.png',dpi=1800,transparent=True)
plt.clf()

rf_df = pd.DataFrame(rf_1600_diff)
sv_df = pd.DataFrame(sv_1600_diff)
nn_df = pd.DataFrame(nn_1600_diff)
ec_df = pd.DataFrame(ec_1600_diff)
rf_df['classifier'] = "Random Forest"
sv_df['classifier'] = "SVM"
nn_df["classifier"] = "Neural Network"
ec_df['classifier'] = 'Ensemble'
rf_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
sv_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
nn_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
ec_df.rename(columns = {0: 'AUC_differential'}, inplace = True)

col_palette = {"Random Forest" : "firebrick",
               "SVM" : "gold",
               "Neural Network" : "darkorange",
               "Ensemble" : "indigo"}
plt.figure(figsize = (10,5), dpi = 1800)
boxplot_df = pd.concat([rf_df, sv_df, nn_df, ec_df], axis = 0)
sns.boxplot(x = "classifier", y = "AUC_differential", data = boxplot_df, whis = [2.5, 97.5], palette = col_palette)
plt.axhline(0, linewidth=2, linestyle = '-', color='r')
plt.xlabel('Machine Learning Algorithm')
plt.ylabel('Difference in AUC (ML - LR)')
plt.title('AUC Comparison: Sample = 1600')
plt.savefig('./figures/samples/AUC_comparison_sample1600.png',dpi=1800,transparent=True)
plt.clf()

rf_df = pd.DataFrame(rf_3200_diff)
sv_df = pd.DataFrame(sv_3200_diff)
nn_df = pd.DataFrame(nn_3200_diff)
ec_df = pd.DataFrame(ec_3200_diff)
rf_df['classifier'] = "Random Forest"
sv_df['classifier'] = "SVM"
nn_df["classifier"] = "Neural Network"
ec_df['classifier'] = 'Ensemble'
rf_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
sv_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
nn_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
ec_df.rename(columns = {0: 'AUC_differential'}, inplace = True)

col_palette = {"Random Forest" : "firebrick",
               "SVM" : "gold",
               "Neural Network" : "darkorange",
               "Ensemble" : "indigo"}
plt.figure(figsize = (10,5), dpi = 1800)
boxplot_df = pd.concat([rf_df, sv_df, nn_df, ec_df], axis = 0)
sns.boxplot(x = "classifier", y = "AUC_differential", data = boxplot_df, whis = [2.5, 97.5], palette = col_palette)
plt.axhline(0, linewidth=2, linestyle = '-', color='r')
plt.xlabel('Machine Learning Algorithm')
plt.ylabel('Difference in AUC (ML - LR)')
plt.title('AUC Comparison: Sample = 3200')
plt.savefig('./figures/samples/AUC_comparison_sample3200.png',dpi=1800,transparent=True)
plt.clf()

#data reading
lr_10 = pd.read_csv('./predictions/feature_results_10/lr_predictions.csv')
lr_25 = pd.read_csv('./predictions/feature_results_25/lr_predictions.csv')
lr_50 = pd.read_csv('./predictions/feature_results_50/lr_predictions.csv')
lr_100 = pd.read_csv('./predictions/feature_results_100/lr_predictions.csv')
lr_250 = pd.read_csv('./predictions/feature_results_250/lr_predictions.csv')
lr_500 = pd.read_csv('./predictions/feature_results_500/lr_predictions.csv')
rf_10 = pd.read_csv('./predictions/feature_results_10/rf_predictions.csv')
rf_25 = pd.read_csv('./predictions/feature_results_25/rf_predictions.csv')
rf_50 = pd.read_csv('./predictions/feature_results_50/rf_predictions.csv')
rf_100 = pd.read_csv('./predictions/feature_results_100/rf_predictions.csv')
rf_250 = pd.read_csv('./predictions/feature_results_250/rf_predictions.csv')
rf_500 = pd.read_csv('./predictions/feature_results_500/rf_predictions.csv')
sv_10 = pd.read_csv('./predictions/feature_results_10/sv_predictions.csv')
sv_25 = pd.read_csv('./predictions/feature_results_25/sv_predictions.csv')
sv_50 = pd.read_csv('./predictions/feature_results_50/sv_predictions.csv')
sv_100 = pd.read_csv('./predictions/feature_results_100/sv_predictions.csv')
sv_250 = pd.read_csv('./predictions/feature_results_250/sv_predictions.csv')
sv_500 = pd.read_csv('./predictions/feature_results_500/sv_predictions.csv')
nn_10 = pd.read_csv('./predictions/feature_results_10/nn_predictions.csv')
nn_25 = pd.read_csv('./predictions/feature_results_25/nn_predictions.csv')
nn_50 = pd.read_csv('./predictions/feature_results_50/nn_predictions.csv')
nn_100 = pd.read_csv('./predictions/feature_results_100/nn_predictions.csv')
nn_250 = pd.read_csv('./predictions/feature_results_250/nn_predictions.csv')
nn_500 = pd.read_csv('./predictions/feature_results_500/nn_predictions.csv')
ec_10 = pd.read_csv('./predictions/feature_results_10/ec_predictions.csv')
ec_25 = pd.read_csv('./predictions/feature_results_25/ec_predictions.csv')
ec_50 = pd.read_csv('./predictions/feature_results_50/ec_predictions.csv')
ec_100 = pd.read_csv('./predictions/feature_results_100/ec_predictions.csv')
ec_250 = pd.read_csv('./predictions/feature_results_250/ec_predictions.csv')
ec_500 = pd.read_csv('./predictions/feature_results_500/ec_predictions.csv')

#actioning the bootstrap
lr_10_tuple = roc_auc(lr_10)
lr_25_tuple = roc_auc(lr_25)
lr_50_tuple = roc_auc(lr_50)
lr_100_tuple = roc_auc(lr_100)
lr_250_tuple = roc_auc(lr_250)
lr_500_tuple = roc_auc(lr_500)
rf_10_tuple = roc_auc(rf_10)
rf_25_tuple = roc_auc(rf_25)
rf_50_tuple = roc_auc(rf_50)
rf_100_tuple = roc_auc(rf_100)
rf_250_tuple = roc_auc(rf_250)
rf_500_tuple = roc_auc(rf_500)
sv_10_tuple = roc_auc(sv_10)
sv_25_tuple = roc_auc(sv_25)
sv_50_tuple = roc_auc(sv_50)
sv_100_tuple = roc_auc(sv_100)
sv_250_tuple = roc_auc(sv_250)
sv_500_tuple = roc_auc(sv_500)
nn_10_tuple = roc_auc(nn_10)
nn_25_tuple = roc_auc(nn_25)
nn_50_tuple = roc_auc(nn_50)
nn_100_tuple = roc_auc(nn_100)
nn_250_tuple = roc_auc(nn_250)
nn_500_tuple = roc_auc(nn_500)
ec_10_tuple = roc_auc(ec_10)
ec_25_tuple = roc_auc(ec_25)
ec_50_tuple = roc_auc(ec_50)
ec_100_tuple = roc_auc(ec_100)
ec_250_tuple = roc_auc(ec_250)
ec_500_tuple = roc_auc(ec_500)
rf_10_diff = rf_10_tuple[3] - lr_10_tuple[3]
sv_10_diff = sv_10_tuple[3] - lr_10_tuple[3]
nn_10_diff = nn_10_tuple[3] -lr_10_tuple[3]
ec_10_diff = ec_10_tuple[3] - lr_10_tuple[3]
rf_25_diff = rf_25_tuple[3] - lr_25_tuple[3]
sv_25_diff = sv_25_tuple[3] - lr_25_tuple[3]
nn_25_diff = nn_25_tuple[3] -lr_25_tuple[3]
ec_25_diff = ec_25_tuple[3] - lr_25_tuple[3]
rf_50_diff = rf_50_tuple[3] - lr_50_tuple[3]
sv_50_diff = sv_50_tuple[3] - lr_50_tuple[3]
nn_50_diff = nn_50_tuple[3] -lr_50_tuple[3]
ec_50_diff = ec_50_tuple[3] - lr_50_tuple[3]
rf_100_diff = rf_100_tuple[3] - lr_100_tuple[3]
sv_100_diff = sv_100_tuple[3] - lr_100_tuple[3]
nn_100_diff = nn_100_tuple[3] -lr_100_tuple[3]
ec_100_diff = ec_100_tuple[3] - lr_100_tuple[3]
rf_250_diff = rf_250_tuple[3] - lr_250_tuple[3]
sv_250_diff = sv_250_tuple[3] - lr_250_tuple[3]
nn_250_diff = nn_250_tuple[3] -lr_250_tuple[3]
ec_250_diff = ec_250_tuple[3] - lr_250_tuple[3]
rf_500_diff = rf_500_tuple[3] - lr_500_tuple[3]
sv_500_diff = sv_500_tuple[3] - lr_500_tuple[3]
nn_500_diff = nn_500_tuple[3] -lr_500_tuple[3]
ec_500_diff = ec_500_tuple[3] - lr_500_tuple[3]

#plotting
rf_df = pd.DataFrame(rf_10_diff)
sv_df = pd.DataFrame(sv_10_diff)
nn_df = pd.DataFrame(nn_10_diff)
ec_df = pd.DataFrame(ec_10_diff)
rf_df['classifier'] = "Random Forest"
sv_df['classifier'] = "SVM"
nn_df["classifier"] = "Neural Network"
ec_df['classifier'] = 'Ensemble'
rf_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
sv_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
nn_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
ec_df.rename(columns = {0: 'AUC_differential'}, inplace = True)

col_palette = {"Random Forest" : "firebrick",
               "SVM" : "gold",
               "Neural Network" : "darkorange",
               "Ensemble" : "indigo"}
plt.figure(figsize = (10,5), dpi = 1800)
boxplot_df = pd.concat([rf_df, sv_df, nn_df, ec_df], axis = 0)
sns.boxplot(x = "classifier", y = "AUC_differential", data = boxplot_df, whis = [2.5, 97.5], palette = col_palette)
plt.axhline(0, linewidth=2, linestyle = '-', color='r')
plt.xlabel('Machine Learning Algorithm')
plt.ylabel('Difference in AUC (ML - LR)')
plt.title('AUC Comparison: Features = 10')
plt.savefig('./figures/features/AUC_comparison_features10.png',dpi=1800,transparent=True)
plt.clf()

rf_df = pd.DataFrame(rf_25_diff)
sv_df = pd.DataFrame(sv_25_diff)
nn_df = pd.DataFrame(nn_25_diff)
ec_df = pd.DataFrame(ec_25_diff)
rf_df['classifier'] = "Random Forest"
sv_df['classifier'] = "SVM"
nn_df["classifier"] = "Neural Network"
ec_df['classifier'] = 'Ensemble'
rf_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
sv_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
nn_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
ec_df.rename(columns = {0: 'AUC_differential'}, inplace = True)

col_palette = {"Random Forest" : "firebrick",
               "SVM" : "gold",
               "Neural Network" : "darkorange",
               "Ensemble" : "indigo"}
plt.figure(figsize = (10,5), dpi = 1800)
boxplot_df = pd.concat([rf_df, sv_df, nn_df, ec_df], axis = 0)
sns.boxplot(x = "classifier", y = "AUC_differential", data = boxplot_df, whis = [2.5, 97.5], palette = col_palette)
plt.axhline(0, linewidth=2, linestyle = '-', color='r')
plt.xlabel('Machine Learning Algorithm')
plt.ylabel('Difference in AUC (ML - LR)')
plt.title('AUC Comparison: Features = 25')
plt.savefig('./figures/features/AUC_comparison_features25.png',dpi=1800,transparent=True)
plt.clf()

rf_df = pd.DataFrame(rf_50_diff)
sv_df = pd.DataFrame(sv_50_diff)
nn_df = pd.DataFrame(nn_50_diff)
ec_df = pd.DataFrame(ec_50_diff)
rf_df['classifier'] = "Random Forest"
sv_df['classifier'] = "SVM"
nn_df["classifier"] = "Neural Network"
ec_df['classifier'] = 'Ensemble'
rf_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
sv_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
nn_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
ec_df.rename(columns = {0: 'AUC_differential'}, inplace = True)

col_palette = {"Random Forest" : "firebrick",
               "SVM" : "gold",
               "Neural Network" : "darkorange",
               "Ensemble" : "indigo"}
plt.figure(figsize = (10,5), dpi = 1800)
boxplot_df = pd.concat([rf_df, sv_df, nn_df, ec_df], axis = 0)
sns.boxplot(x = "classifier", y = "AUC_differential", data = boxplot_df, whis = [2.5, 97.5], palette = col_palette)
plt.axhline(0, linewidth=2, linestyle = '-', color='r')
plt.xlabel('Machine Learning Algorithm')
plt.ylabel('Difference in AUC (ML - LR)')
plt.title('AUC Comparison: Features = 50')
plt.savefig('./figures/features/AUC_comparison_features50.png',dpi=1800,transparent=True)
plt.clf()

rf_df = pd.DataFrame(rf_100_diff)
sv_df = pd.DataFrame(sv_100_diff)
nn_df = pd.DataFrame(nn_100_diff)
ec_df = pd.DataFrame(ec_100_diff)
rf_df['classifier'] = "Random Forest"
sv_df['classifier'] = "SVM"
nn_df["classifier"] = "Neural Network"
ec_df['classifier'] = 'Ensemble'
rf_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
sv_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
nn_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
ec_df.rename(columns = {0: 'AUC_differential'}, inplace = True)

col_palette = {"Random Forest" : "firebrick",
               "SVM" : "gold",
               "Neural Network" : "darkorange",
               "Ensemble" : "indigo"}
plt.figure(figsize = (10,5), dpi = 1800)
boxplot_df = pd.concat([rf_df, sv_df, nn_df, ec_df], axis = 0)
sns.boxplot(x = "classifier", y = "AUC_differential", data = boxplot_df, whis = [2.5, 97.5], palette = col_palette)
plt.axhline(0, linewidth=2, linestyle = '-', color='r')
plt.xlabel('Machine Learning Algorithm')
plt.ylabel('Difference in AUC (ML - LR)')
plt.title('AUC Comparison: Features = 100')
plt.savefig('./figures/features/AUC_comparison_features100.png',dpi=1800,transparent=True)
plt.clf()

rf_df = pd.DataFrame(rf_250_diff)
sv_df = pd.DataFrame(sv_250_diff)
nn_df = pd.DataFrame(nn_250_diff)
ec_df = pd.DataFrame(ec_250_diff)
rf_df['classifier'] = "Random Forest"
sv_df['classifier'] = "SVM"
nn_df["classifier"] = "Neural Network"
ec_df['classifier'] = 'Ensemble'
rf_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
sv_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
nn_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
ec_df.rename(columns = {0: 'AUC_differential'}, inplace = True)

col_palette = {"Random Forest" : "firebrick",
               "SVM" : "gold",
               "Neural Network" : "darkorange",
               "Ensemble" : "indigo"}
plt.figure(figsize = (10,5), dpi = 1800)
boxplot_df = pd.concat([rf_df, sv_df, nn_df, ec_df], axis = 0)
sns.boxplot(x = "classifier", y = "AUC_differential", data = boxplot_df, whis = [2.5, 97.5], palette = col_palette)
plt.axhline(0, linewidth=2, linestyle = '-', color='r')
plt.xlabel('Machine Learning Algorithm')
plt.ylabel('Difference in AUC (ML - LR)')
plt.title('AUC Comparison: Features = 250')
plt.savefig('./figures/features/AUC_comparison_features250.png',dpi=1800,transparent=True)
plt.clf()

rf_df = pd.DataFrame(rf_500_diff)
sv_df = pd.DataFrame(sv_500_diff)
nn_df = pd.DataFrame(nn_500_diff)
ec_df = pd.DataFrame(ec_500_diff)
rf_df['classifier'] = "Random Forest"
sv_df['classifier'] = "SVM"
nn_df["classifier"] = "Neural Network"
ec_df['classifier'] = 'Ensemble'
rf_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
sv_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
nn_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
ec_df.rename(columns = {0: 'AUC_differential'}, inplace = True)

col_palette = {"Random Forest" : "firebrick",
               "SVM" : "gold",
               "Neural Network" : "darkorange",
               "Ensemble" : "indigo"}
plt.figure(figsize = (10,5), dpi = 1800)
boxplot_df = pd.concat([rf_df, sv_df, nn_df, ec_df], axis = 0)
sns.boxplot(x = "classifier", y = "AUC_differential", data = boxplot_df, whis = [2.5, 97.5], palette = col_palette)
plt.axhline(0, linewidth=2, linestyle = '-', color='r')
plt.xlabel('Machine Learning Algorithm')
plt.ylabel('Difference in AUC (ML - LR)')
plt.title('AUC Comparison: Features = 500')
plt.savefig('./figures/features/AUC_comparison_features500.png',dpi=1800,transparent=True)
plt.clf()

#data reading
lr = pd.read_csv('./predictions/pca_results/lr_predictions.csv')
rf = pd.read_csv('./predictions/pca_results/rf_predictions.csv')
sv = pd.read_csv('./predictions/pca_results/sv_predictions.csv')
nn = pd.read_csv('./predictions/pca_results/nn_predictions.csv')
ec = pd.read_csv('./predictions/pca_results/ec_predictions.csv')

#actioning the bootstrap
lr_tuple = roc_auc(lr)
rf_tuple = roc_auc(rf)
sv_tuple = roc_auc(sv)
nn_tuple = roc_auc(nn)
ec_tuple = roc_auc(ec)

rf_diff = rf_tuple[3] - lr_tuple[3]
sv_diff = sv_tuple[3] - lr_tuple[3]
nn_diff = nn_tuple[3] -lr_tuple[3]
ec_diff = ec_tuple[3] - lr_tuple[3]

rf_df = pd.DataFrame(rf_diff)
sv_df = pd.DataFrame(sv_diff)
nn_df = pd.DataFrame(nn_diff)
ec_df = pd.DataFrame(ec_diff)
rf_df['classifier'] = "Random Forest"
sv_df['classifier'] = "SVM"
nn_df["classifier"] = "Neural Network"
ec_df['classifier'] = 'Ensemble'
rf_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
sv_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
nn_df.rename(columns = {0: 'AUC_differential'}, inplace = True)
ec_df.rename(columns = {0: 'AUC_differential'}, inplace = True)

col_palette = {"Random Forest" : "firebrick",
               "SVM" : "gold",
               "Neural Network" : "darkorange",
               "Ensemble" : "indigo"}
plt.figure(figsize = (10,5), dpi = 1800)
boxplot_df = pd.concat([rf_df, sv_df, nn_df, ec_df], axis = 0)
sns.boxplot(x = "classifier", y = "AUC_differential", data = boxplot_df, whis = [2.5, 97.5], palette = col_palette)
plt.axhline(0, linewidth=2, linestyle = '-', color='r')
plt.xlabel('Machine Learning Algorithm')
plt.ylabel('Difference in AUC (ML - LR)')
plt.title('AUC comparison: PCA')
plt.savefig('./figures/pca/AUC_comparison.png',dpi=1800,transparent=True)
plt.show()
plt.clf()