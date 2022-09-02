#this code produces the AUC comparison plots
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

def roc_auc(lr_df, rf_df, sv_df, nn_df, ec_df = None):
    '''Takes many long form dataframes of many bootstrap resamples and gives back the AUC for each resample.
       lr_df: a dataframe with prob and deceased as columns that comes from logistic reg model
       rf_df: a dataframe with prob and deceased as columns that comes from random forest model
       sv_df: a dataframe with prob and deceased as columns that comes from SVM
       nn_df: a dataframe with prob and deceased as columns that comes from NN model
       ec_df: a dataframe with prob and deceased as columns that comes from ensemble classifier (optional argument)'''
    lr_aucs, rf_aucs, sv_aucs, nn_aucs, ec_aucs = [], [], [], [], []
    for i in range(len(lr_df.columns)//2):
        fpr, tpr, thr = roc_curve(lr_df.iloc[:, 2*i + 1], lr_df.iloc[:, 2*i])
        feeder_auc = auc(fpr, tpr)
        lr_aucs.append(feeder_auc)
    for i in range(len(rf_df.columns)//2):
        fpr, tpr, thr = roc_curve(rf_df.iloc[:, 2*i + 1], rf_df.iloc[:, 2*i])
        feeder_auc = auc(fpr, tpr)
        rf_aucs.append(feeder_auc)
    for i in range(len(sv_df.columns)//2):
        fpr, tpr, thr = roc_curve(sv_df.iloc[:, 2*i + 1], sv_df.iloc[:, 2*i])
        feeder_auc = auc(fpr, tpr)
        sv_aucs.append(feeder_auc)
    for i in range(len(nn_df.columns)//2):
        fpr, tpr, thr = roc_curve(nn_df.iloc[:, 2*i + 1], nn_df.iloc[:, 2*i])
        feeder_auc = auc(fpr, tpr)
        nn_aucs.append(feeder_auc)
    dct = {'lr_auc': lr_aucs,
           'rf_auc': rf_aucs,
           'sv_auc': sv_aucs,
           'nn_auc': nn_aucs}
    if ec_df is not None:
        for i in range(len(ec_df.columns) // 2):
            fpr, tpr, thr = roc_curve(ec_df.iloc[:, 2 * i + 1], ec_df.iloc[:, 2 * i])
            feeder_auc = auc(fpr, tpr)
            ec_aucs.append(feeder_auc)
        dct['ec_auc'] = ec_aucs
    return pd.DataFrame(dct)

def differencer(df):
    '''Takes the dataframe of AUCs from the roc_auc function and returns the difference between lr and each ML model, for plotting
       df: a dataframe from the roc_auc functions'''
    intermediary_df = pd.DataFrame()
    output_df = pd.DataFrame()
    lr_col, ml_cols = list(df.columns)[0], (df.columns)[1:]
    for ml_col in ml_cols:
        intermediary_df[ml_col] = df[ml_col] - df[lr_col]
    output_df['Random Forest'] = intermediary_df.iloc[:, 0]
    output_df['SVM'] = intermediary_df.iloc[:, 1]
    output_df['Neural Network'] = intermediary_df.iloc[:, 2]
    if len(intermediary_df.columns)==4:
        output_df['Ensemble'] = intermediary_df.iloc[:, 3]
    output_df = pd.melt(output_df)
    output_df.rename(columns ={'variable': 'classifier'}, inplace = True)
    output_df.rename(columns = {'value': 'auc_diff'}, inplace = True)
    return output_df

def get_plot_df(lr_df, rf_df, sv_df, nn_df, ec_df = None):
    '''Takes the dataframes of predictions from bootstraps and returns an output dataframe for plotting
        lr_df: a dataframe with prob and deceased as columns that comes from logistic reg model
        rf_df: a dataframe with prob and deceased as columns that comes from random forest model
        sv_df: a dataframe with prob and deceased as columns that comes from SVM
        nn_df: a dataframe with prob and deceased as columns that comes from NN model
        ec_df: a dataframe with prob and deceased as columns that comes from ensemble classifier (optional argument)'''
    plot_df = differencer(roc_auc(lr_df = lr_df,
                                  rf_df = rf_df,
                                  sv_df = sv_df,
                                  nn_df = nn_df,
                                  ec_df = ec_df))
    return plot_df

def plotter(df, path, title, upper = None, lower = None):
    plt.figure(figsize=(10, 5), dpi=1800)
    sns.boxplot(x="classifier", y="auc_diff", whis=[2.5, 97.5],
                data=df, palette="Dark2")
    if upper is not None and lower is not None:
        plt.ylim(lower, upper)
    plt.axhline(0, linewidth=1, linestyle='-', color='r')
    plt.xlabel('Machine Learning Algorithm')
    plt.ylabel('Difference in AUC (ML - LR)')
    plt.title(title)
    if upper is not None and lower is not None:
        plt.text(x=-0.45, y= (0.95*(upper-lower) + lower), s='Favours ML', fontsize=10, fontstyle='italic', color='r')
        plt.text(x=-0.45, y= (0.03*(upper-lower) + lower), s='Favours LR', fontsize=10, fontstyle='italic', color='r')
    else:
        upper, lower = plt.gca().get_ylim()[1], plt.gca().get_ylim()[0]
        plt.text(x=-0.45, y=(0.95 * (upper - lower) + lower), s='Favours ML', fontsize=10, fontstyle='italic',
                 color='r')
        plt.text(x=-0.45, y=(0.03 * (upper - lower) + lower), s='Favours LR', fontsize=10, fontstyle='italic',
                 color='r')
    plt.savefig(path, dpi=1800, transparent=True)
    plt.clf()

#data reading
lr = pd.read_csv('./predictions/initial_results/lr_predictions.csv')
rf = pd.read_csv('./predictions/initial_results/rf_predictions.csv')
sv = pd.read_csv('./predictions/initial_results/sv_predictions.csv')
nn = pd.read_csv('./predictions/initial_results/nn_predictions.csv')
ec = pd.read_csv('./predictions/initial_results/ec_predictions.csv')

inital_df = get_plot_df(lr_df = lr,
                        rf_df = rf,
                        sv_df = sv,
                        nn_df = nn)

plotter(inital_df, upper = 0.15, lower = -0.15, title = 'AUC Comparison', path = './figures/initial_analysis/AUC_comparison_1.png')

ensemble_df = get_plot_df(lr_df = lr,
                          rf_df = rf,
                          sv_df = sv,
                          nn_df = nn,
                          ec_df = ec)

plotter(ensemble_df, upper = 0.15, lower = -0.15, title = 'AUC Comparison: with Ensemble Classifier', path = './figures/initial_analysis/AUC_comparison_2.png')

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

df_100 = get_plot_df(lr_df = lr_100,
                     rf_df = rf_100,
                     sv_df = sv_100,
                     nn_df = nn_100,
                     ec_df = ec_100)
df_200 = get_plot_df(lr_df = lr_200,
                     rf_df = rf_200,
                     sv_df = sv_200,
                     nn_df = nn_200,
                     ec_df = ec_200)
df_400 = get_plot_df(lr_df = lr_400,
                     rf_df = rf_400,
                     sv_df = sv_400,
                     nn_df = nn_400,
                     ec_df = ec_400)
df_800 = get_plot_df(lr_df = lr_800,
                     rf_df = rf_800,
                     sv_df = sv_800,
                     nn_df = nn_800,
                     ec_df = ec_800)
df_1600 = get_plot_df(lr_df = lr_1600,
                      rf_df = rf_1600,
                      sv_df = sv_1600,
                      nn_df = nn_1600,
                      ec_df = ec_1600)
df_3200 = get_plot_df(lr_df = lr_3200,
                      rf_df = rf_3200,
                      sv_df = sv_3200,
                      nn_df = nn_3200,
                      ec_df = ec_3200)

plotter(df_100, title = 'AUC Comparison: Sample = 100', path = './figures/samples/AUC_comparison_sample100.png')
plotter(df_200, title = 'AUC Comparison: Sample = 200', path = './figures/samples/AUC_comparison_sample200.png')
plotter(df_400, title = 'AUC Comparison: Sample = 400', path = './figures/samples/AUC_comparison_sample400.png')
plotter(df_800, title = 'AUC Comparison: Sample = 800', path = './figures/samples/AUC_comparison_sample800.png')
plotter(df_1600, title = 'AUC Comparison: Sample = 1600', path = './figures/samples/AUC_comparison_sample1600.png')
plotter(df_3200, title = 'AUC Comparison: Sample = 3200', path = './figures/samples/AUC_comparison_sample3200.png')

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

df_10 = get_plot_df(lr_df = lr_10,
                    rf_df = rf_10,
                    sv_df = sv_10,
                    nn_df = nn_10,
                    ec_df = ec_10)
df_25 = get_plot_df(lr_df = lr_25,
                    rf_df = rf_25,
                    sv_df = sv_25,
                    nn_df = nn_25,
                    ec_df = ec_25)
df_50 = get_plot_df(lr_df = lr_50,
                    rf_df = rf_50,
                    sv_df = sv_50,
                    nn_df = nn_50,
                    ec_df = ec_50)
df_100 = get_plot_df(lr_df = lr_100,
                     rf_df = rf_100,
                     sv_df = sv_100,
                     nn_df = nn_100,
                     ec_df = ec_100)
df_250 = get_plot_df(lr_df = lr_250,
                     rf_df = rf_250,
                     sv_df = sv_250,
                     nn_df = nn_250,
                     ec_df = ec_250)
df_500 = get_plot_df(lr_df = lr_500,
                     rf_df = rf_500,
                     sv_df = sv_500,
                     nn_df = nn_500,
                     ec_df = ec_500)

plotter(df_10, title = 'AUC Comparison: Features = 10', path = './figures/features/AUC_comparison_features10.png')
plotter(df_25, title = 'AUC Comparison: Features = 25', path = './figures/features/AUC_comparison_features25.png')
plotter(df_50, title = 'AUC Comparison: Features = 50', path = './figures/features/AUC_comparison_features50.png')
plotter(df_100, title = 'AUC Comparison: Features = 100', path = './figures/features/AUC_comparison_features100.png')
plotter(df_250, title = 'AUC Comparison: Features = 250', path = './figures/features/AUC_comparison_features250.png')
plotter(df_500, title = 'AUC Comparison: Features = 500', path = './figures/features/AUC_comparison_features500.png')

lr = pd.read_csv('./predictions/pca_results/lr_predictions.csv')
rf = pd.read_csv('./predictions/pca_results/rf_predictions.csv')
sv = pd.read_csv('./predictions/pca_results/sv_predictions.csv')
nn = pd.read_csv('./predictions/pca_results/nn_predictions.csv')
ec = pd.read_csv('./predictions/pca_results/ec_predictions.csv')

pca_df = get_plot_df(lr_df = lr,
                     rf_df = rf,
                     sv_df = sv,
                     nn_df = nn,
                     ec_df = ec)

plotter(pca_df, title = 'AUC Comparison: PCA', path = './figures/pca/AUC_comparison.png')