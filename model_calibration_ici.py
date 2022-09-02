#this code produces the ICI comparison plots
# #packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.float_format', lambda x: '%.4f' % x)

def merger(lr, rf, sv, nn, ec = None):
    '''Simple function that merges Integrated Calibration Index dataframes
       and puts them in the required form for plotting'''
    if ec is not None:
        df = pd.concat([lr, rf, sv, nn, ec], axis = 1)
        df.columns = ['lr', 'rf', 'sv', 'nn', 'ec']
    else:
        df = pd.concat([lr, rf, sv, nn], axis = 1)
        df.columns = ['lr', 'rf', 'sv', 'nn']
    intermediary_df = pd.DataFrame()
    output_df = pd.DataFrame()
    lr_col, ml_cols = list(df.columns)[0], (df.columns)[1:]
    for ml_col in ml_cols:
        intermediary_df[ml_col] = df[lr_col] - df[ml_col]
    output_df['Random Forest'] = intermediary_df.iloc[:, 0]
    output_df['SVM'] = intermediary_df.iloc[:, 1]
    output_df['Neural Network'] = intermediary_df.iloc[:, 2]
    if len(intermediary_df.columns) == 4:
        output_df['Ensemble'] = intermediary_df.iloc[:, 3]
    output_df = pd.melt(output_df)
    output_df.rename(columns={'variable': 'classifier'}, inplace=True)
    output_df.rename(columns={'value': 'ici_diff'}, inplace=True)
    return output_df

def plotter(df, path, title, upper = None, lower = None):
    plt.figure(figsize=(10, 5), dpi=1800)
    sns.boxplot(x="classifier", y="ici_diff", whis=[2.5, 97.5],
                data=df, palette="Dark2")
    if upper is not None and lower is not None:
        plt.ylim(lower, upper)
    plt.axhline(0, linewidth=1, linestyle='-', color='r')
    plt.xlabel('Machine Learning Algorithm')
    plt.ylabel('Difference in ICI (LR - ML)')
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
init_lr_ici = pd.read_csv('./predictions/initial_results/lr_ice.csv')
init_rf_ici = pd.read_csv('./predictions/initial_results/rf_ice.csv')
init_sv_ici = pd.read_csv('./predictions/initial_results/sv_ice.csv')
init_nn_ici = pd.read_csv('./predictions/initial_results/nn_ice.csv')
init_ec_ici = pd.read_csv('./predictions/initial_results/ec_ice.csv')

initial_df = merger(init_lr_ici,
                    init_rf_ici,
                    init_sv_ici,
                    init_nn_ici)

plotter(initial_df, title = 'ICI Comparison', path = './figures/initial_analysis/ICI_comparison_1.png')

ensemble_df = merger(init_lr_ici,
                     init_rf_ici,
                     init_sv_ici,
                     init_nn_ici,
                     init_ec_ici)

plotter(ensemble_df, title = 'ICI Comparison: with Ensemble Classifier', path = './figures/initial_analysis/ICI_comparison_2.png')

lr_100 = pd.read_csv('./predictions/datapoint_results_100/lr_ice.csv')
lr_200 = pd.read_csv('./predictions/datapoint_results_200/lr_ice.csv')
lr_400 = pd.read_csv('./predictions/datapoint_results_400/lr_ice.csv')
lr_800 = pd.read_csv('./predictions/datapoint_results_800/lr_ice.csv')
lr_1600 = pd.read_csv('./predictions/datapoint_results_1600/lr_ice.csv')
lr_3200 = pd.read_csv('./predictions/datapoint_results_3200/lr_ice.csv')
rf_100 = pd.read_csv('./predictions/datapoint_results_100/rf_ice.csv')
rf_200 = pd.read_csv('./predictions/datapoint_results_200/rf_ice.csv')
rf_400 = pd.read_csv('./predictions/datapoint_results_400/rf_ice.csv')
rf_800 = pd.read_csv('./predictions/datapoint_results_800/rf_ice.csv')
rf_1600 = pd.read_csv('./predictions/datapoint_results_1600/rf_ice.csv')
rf_3200 = pd.read_csv('./predictions/datapoint_results_3200/rf_ice.csv')
sv_100 = pd.read_csv('./predictions/datapoint_results_100/sv_ice.csv')
sv_200 = pd.read_csv('./predictions/datapoint_results_200/sv_ice.csv')
sv_400 = pd.read_csv('./predictions/datapoint_results_400/sv_ice.csv')
sv_800 = pd.read_csv('./predictions/datapoint_results_800/sv_ice.csv')
sv_1600 = pd.read_csv('./predictions/datapoint_results_1600/sv_ice.csv')
sv_3200 = pd.read_csv('./predictions/datapoint_results_3200/sv_ice.csv')
nn_100 = pd.read_csv('./predictions/datapoint_results_100/nn_ice.csv')
nn_200 = pd.read_csv('./predictions/datapoint_results_200/nn_ice.csv')
nn_400 = pd.read_csv('./predictions/datapoint_results_400/nn_ice.csv')
nn_800 = pd.read_csv('./predictions/datapoint_results_800/nn_ice.csv')
nn_1600 = pd.read_csv('./predictions/datapoint_results_1600/nn_ice.csv')
nn_3200 = pd.read_csv('./predictions/datapoint_results_3200/nn_ice.csv')
ec_100 = pd.read_csv('./predictions/datapoint_results_100/ec_ice.csv')
ec_200 = pd.read_csv('./predictions/datapoint_results_200/ec_ice.csv')
ec_400 = pd.read_csv('./predictions/datapoint_results_400/ec_ice.csv')
ec_800 = pd.read_csv('./predictions/datapoint_results_800/ec_ice.csv')
ec_1600 = pd.read_csv('./predictions/datapoint_results_1600/ec_ice.csv')
ec_3200 = pd.read_csv('./predictions/datapoint_results_3200/ec_ice.csv')

df_100 = merger(lr = lr_100,
                rf = rf_100,
                sv = sv_100,
                nn = nn_100,
                ec = ec_100)
df_200 = merger(lr = lr_200,
                rf = rf_200,
                sv = sv_200,
                nn = nn_200,
                ec = ec_200)
df_400 = merger(lr = lr_400,
                rf = rf_400,
                sv = sv_400,
                nn = nn_400,
                ec = ec_400)
df_800 = merger(lr = lr_800,
                rf = rf_800,
                sv = sv_800,
                nn = nn_800,
                ec = ec_800)
df_1600 = merger(lr = lr_1600,
                 rf = rf_1600,
                 sv = sv_1600,
                 nn = nn_1600,
                 ec = ec_1600)
df_3200 = merger(lr = lr_3200,
                 rf = rf_3200,
                 sv = sv_3200,
                 nn = nn_3200,
                 ec = ec_3200)

plotter(df_100, upper= 0.1, lower = -0.25, title = 'ICI Comparison: Sample = 100', path = './figures/samples/ICI_comparison_sample100.png')
plotter(df_200, title = 'ICI Comparison: Sample = 200', path = './figures/samples/ICI_comparison_sample200.png')
plotter(df_400, title = 'ICI Comparison: Sample = 400', path = './figures/samples/ICI_comparison_sample400.png')
plotter(df_800, title = 'ICI Comparison: Sample = 800', path = './figures/samples/ICI_comparison_sample800.png')
plotter(df_1600, title = 'ICI Comparison: Sample = 1600', path = './figures/samples/ICI_comparison_sample1600.png')
plotter(df_3200, title = 'ICI Comparison: Sample = 3200', path = './figures/samples/ICI_comparison_sample3200.png')

lr_10 = pd.read_csv('./predictions/feature_results_10/lr_ice.csv')
lr_25 = pd.read_csv('./predictions/feature_results_25/lr_ice.csv')
lr_50 = pd.read_csv('./predictions/feature_results_50/lr_ice.csv')
lr_100 = pd.read_csv('./predictions/feature_results_100/lr_ice.csv')
lr_250 = pd.read_csv('./predictions/feature_results_250/lr_ice.csv')
lr_500 = pd.read_csv('./predictions/feature_results_500/lr_ice.csv')
rf_10 = pd.read_csv('./predictions/feature_results_10/rf_ice.csv')
rf_25 = pd.read_csv('./predictions/feature_results_25/rf_ice.csv')
rf_50 = pd.read_csv('./predictions/feature_results_50/rf_ice.csv')
rf_100 = pd.read_csv('./predictions/feature_results_100/rf_ice.csv')
rf_250 = pd.read_csv('./predictions/feature_results_250/rf_ice.csv')
rf_500 = pd.read_csv('./predictions/feature_results_500/rf_ice.csv')
sv_10 = pd.read_csv('./predictions/feature_results_10/sv_ice.csv')
sv_25 = pd.read_csv('./predictions/feature_results_25/sv_ice.csv')
sv_50 = pd.read_csv('./predictions/feature_results_50/sv_ice.csv')
sv_100 = pd.read_csv('./predictions/feature_results_100/sv_ice.csv')
sv_250 = pd.read_csv('./predictions/feature_results_250/sv_ice.csv')
sv_500 = pd.read_csv('./predictions/feature_results_500/sv_ice.csv')
nn_10 = pd.read_csv('./predictions/feature_results_10/nn_ice.csv')
nn_25 = pd.read_csv('./predictions/feature_results_25/nn_ice.csv')
nn_50 = pd.read_csv('./predictions/feature_results_50/nn_ice.csv')
nn_100 = pd.read_csv('./predictions/feature_results_100/nn_ice.csv')
nn_250 = pd.read_csv('./predictions/feature_results_250/nn_ice.csv')
nn_500 = pd.read_csv('./predictions/feature_results_500/nn_ice.csv')
ec_10 = pd.read_csv('./predictions/feature_results_10/ec_ice.csv')
ec_25 = pd.read_csv('./predictions/feature_results_25/ec_ice.csv')
ec_50 = pd.read_csv('./predictions/feature_results_50/ec_ice.csv')
ec_100 = pd.read_csv('./predictions/feature_results_100/ec_ice.csv')
ec_250 = pd.read_csv('./predictions/feature_results_250/ec_ice.csv')
ec_500 = pd.read_csv('./predictions/feature_results_500/ec_ice.csv')

df_10 = merger(lr = lr_10,
               rf = rf_10,
               sv = sv_10,
               nn = nn_10,
               ec = ec_10)
df_25 = merger(lr = lr_25,
               rf = rf_25,
               sv = sv_25,
               nn = nn_25,
               ec = ec_25)
df_50 = merger(lr = lr_50,
               rf = rf_50,
               sv = sv_50,
               nn = nn_50,
               ec = ec_50)
df_100 = merger(lr = lr_100,
                rf = rf_100,
                sv = sv_100,
                nn = nn_100,
                ec = ec_100)
df_250 = merger(lr = lr_250,
                rf = rf_250,
                sv = sv_250,
                nn = nn_250,
                ec = ec_250)
df_500 = merger(lr = lr_500,
                rf = rf_500,
                sv = sv_500,
                nn = nn_500,
                ec = ec_500)

plotter(df_10, upper = 0.1, lower = -0.2, title = 'ICI Comparison: Features = 10', path = './figures/features/ICI_comparison_features10.png')
plotter(df_25, title = 'ICI Comparison: Features = 25', path = './figures/features/ICI_comparison_features25.png')
plotter(df_50, title = 'ICI Comparison: Features = 50', path = './figures/features/ICI_comparison_features50.png')
plotter(df_100, title = 'ICI Comparison: Features = 100', path = './figures/features/ICI_comparison_features100.png')
plotter(df_250, title = 'ICI Comparison: Features = 250', path = './figures/features/ICI_comparison_features250.png')
plotter(df_500, title = 'ICI Comparison: Features = 500', path = './figures/features/ICI_comparison_features500.png')

lr = pd.read_csv('./predictions/pca_results/lr_ice.csv')
rf = pd.read_csv('./predictions/pca_results/rf_ice.csv')
sv = pd.read_csv('./predictions/pca_results/sv_ice.csv')
nn = pd.read_csv('./predictions/pca_results/nn_ice.csv')
ec = pd.read_csv('./predictions/pca_results/ec_ice.csv')

pca_df = merger(lr = lr,
                rf = rf,
                sv = sv,
                nn = nn,
                ec = ec)

plotter(pca_df, title = 'ICI Comparison: PCA', path = './figures/pca/ICI_comparison.png')