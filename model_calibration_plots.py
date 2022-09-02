#this code produces the calibration plots
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.float_format', lambda x: '%.4f' % x)

def cali_conf_intervals(dataset):
    '''Takes dataframes produced by the cali curve function in R and generates the median calibration curve, accompanied with
       it's 95% bootstrapped confidence interval'''
    observed_proba = dataset.iloc[:, 1:]
    summary_dic = {'median': observed_proba.median(axis = 1),
                   'lb': observed_proba.quantile(q = 0.025, axis = 1),
                   'ub': observed_proba.quantile(q = 0.975, axis = 1)}
    output_df = pd.DataFrame(summary_dic)
    output_df.index = dataset.iloc[:, 0]
    output_df = output_df.iloc[::5, :]
    return output_df

lr = pd.read_csv('./predictions/initial_results/lr_cali.csv')
rf = pd.read_csv('./predictions/initial_results/rf_cali.csv')
sv = pd.read_csv('./predictions/initial_results/sv_cali.csv')
nn = pd.read_csv('./predictions/initial_results/nn_cali.csv')
ec = pd.read_csv('./predictions/initial_results/ec_cali.csv')

lr_df = cali_conf_intervals(lr)
rf_df = cali_conf_intervals(rf)
sv_df = cali_conf_intervals(sv)
nn_df = cali_conf_intervals(nn)
ec_df = cali_conf_intervals(ec)

plt.figure(figsize = (10,5), dpi = 1800)
plt.plot(lr_df.index, lr_df['median'],
         color = '#e2ac04',
         label = "Logistic Regression")
plt.ylim(0,1)
plt.xlim(0,1)
plt.plot([0, 1], [0, 1], linestyle = '--', color = 'black', label = 'Perfect Calibration')
plt.fill_between(x = lr_df.index,
                 y1 = lr_df['lb'],
                 y2 = lr_df['ub'],
                 color = '#e2ac04',
                 alpha = 0.2,
                 label = 'Logistic Regression 95% Confidence Interval')
plt.xlabel('Predicted Probability of Death')
plt.ylabel('Observed Mortality')
handles, labels = plt.gca().get_legend_handles_labels()
order = [0, 2, 1]
plt.legend([handles[i] for i in order], [labels[i] for i in order], loc='lower right')
plt.title('Example Calibration Plot: Logistic Regression')
plt.savefig('./figures/examples/example_cali.png',dpi=1800,transparent=True)
plt.clf()

def plotter(lr_df, other_df, other_df_colour, other_df_name, plot_name):
    plt.figure(figsize = (10,5), dpi = 1800)
    plt.plot(lr_df.index, lr_df['median'],
             color = '#e2ac04',
             label = "Logistic Regression")
    plt.fill_between(x = lr_df.index,
                     y1 = lr_df['lb'],
                     y2 = lr_df['ub'],
                     color = '#e2ac04',
                     alpha = 0.2,
                     label = 'Logistic Regression 95% Confidence Interval')
    plt.plot(rf_df.index, other_df['median'],
             color = other_df_colour,
             label = other_df_name)
    plt.fill_between(x = other_df.index,
                     y1 = other_df['lb'],
                     y2 = other_df['ub'],
                     color = other_df_colour,
                     alpha = 0.2,
                     label = other_df_name + ' 95% Confidence Interval')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.plot([0, 1], [0, 1], linestyle = '--', color = 'black', label = 'Perfect Calibration')

    plt.xlabel('Predicted Probability of Death')
    plt.ylabel('Observed Mortality')
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0, 2, 1, 3, 4]
    plt.legend([handles[i] for i in order], [labels[i] for i in order], loc='lower right')
    plt.title('Calibration Plot: Logistic Regression vs ' + other_df_name)
    plt.savefig('./figures/initial_analysis/' + plot_name + '.png',dpi=1800,transparent=True)
    plt.clf()

plotter(lr_df = lr_df,
        other_df = rf_df,
        other_df_colour = '#1c9c74',
        other_df_name = 'Random Forest',
        plot_name = 'lr_rf_cali')

plotter(lr_df = lr_df,
        other_df = sv_df,
        other_df_colour = '#d95f02',
        other_df_name = 'Support Vector Machine',
        plot_name = 'lr_sv_cali')

plotter(lr_df = lr_df,
        other_df = nn_df,
        other_df_colour = '#7474b4',
        other_df_name = 'Neural Network',
        plot_name = 'lr_nn_cali')

plotter(lr_df = lr_df,
        other_df = ec_df,
        other_df_colour = '#e72a8a',
        other_df_name = 'Ensemble',
        plot_name = 'lr_ec_cali')