"""
Script for creating and saving plots for the master thesis report.
"""
import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

SMALL_SIZE = 9
MEDIUM_SIZE = 11
BIGGER_SIZE = 13

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def get_experiment_label(experiment: str) -> str:
    """
    Returns a label for the plots from a experiment folder name.
    :param experiment: Experiment folder name
    :return: A label
    """
    label = ''
    if 'equireg' in experiment:
        label += 'EquiReg '
    sup_percentage = int(float(experiment.split('_')[1]) * 100)
    label += str(sup_percentage) + '%'
    return label


def get_results(path: str, pred_type: str, result_type: str) -> pd.DataFrame:
    """
    Get experiment results as a dataframe.
    :param path: Path to results
    :param pred_type: Test or validation results
    :param result_type: Standard or equivariance results
    :return: The results in a pandas dataframe
    """
    results = pd.DataFrame()

    for experiment_path in glob.glob(os.path.join(path, '**', 'best_' + pred_type + '_prediction'), recursive=True):
        experiment = experiment_path.split('/')[-2]
        if result_type == 'standard':
            if pred_type == 'val':
                experiment_result = pd.read_csv(os.path.join(experiment_path, 'results_summary.csv')).loc[
                    [1], ['Dice Whole', 'Dice Core', 'Dice Enhancing']]
                experiment_result['Dice Collective'] = experiment_result.mean(axis=1).values
            elif pred_type == 'test':
                experiment_result = pd.read_csv(os.path.join(experiment_path, 'Stats_Validation_final.csv'))
                assert experiment_result.loc[125]['Label'] == 'Mean'
                experiment_result = experiment_result.loc[[125], ['Dice_WT', 'Dice_TC', 'Dice_ET']]
                experiment_result = experiment_result.rename(
                    columns={'Dice_WT': 'Dice Whole', 'Dice_TC': 'Dice Core', 'Dice_ET': 'Dice Enhancing'})
                experiment_result['Dice Collective'] = experiment_result.mean(axis=1).values
        elif result_type == 'equivariance':
            experiment_result = pd.read_csv(os.path.join(experiment_path, 'equivariance_results_summary.csv')).loc[
                [1], ['Equivariance Dice Whole', 'Equivariance Dice Core', 'Equivariance Dice Enhancing']]
            experiment_result['Equivariance Dice Collective'] = experiment_result.mean(axis=1).values

        experiment_result['Experiment'] = [experiment]
        experiment_result['Experiment Label'] = [get_experiment_label(experiment)]
        experiment_result['Supervised Percentage'] = [float(experiment.split('_')[1]) * 100]

        if 'equireg' in experiment:
            experiment_result['Method'] = 'EquiReg'
        else:
            experiment_result['Method'] = 'Baseline'

        if '1.0' in experiment_path:
            experiment_result['Seed'] = ['None']
        else:
            experiment_result['Seed'] = [str(int(experiment.split('_')[-1]) - 41)]

        results = results.append(
            experiment_result, ignore_index=True)

    results = results.sort_values(by=['Supervised Percentage', 'Experiment Label', 'Experiment'])
    return results


if not os.path.isdir('report_plots'):
    os.makedirs('report_plots')

results_path = 'data/predictions/results'
pred_type = 'test'
baseline_experiments = ['baseline_1.0', 'baseline_0.1', 'baseline_0.05', 'baseline_0.01']
equireg_experiments = ['equireg_0.1', 'equireg_0.05', 'equireg_0.01', 'equireg_1.0']

results = get_results(results_path, pred_type, 'standard')
equivariance_results = get_results(results_path, pred_type, 'equivariance')

mean_resluts = results.groupby(['Experiment Label'], as_index=False).mean()
mean_resluts = mean_resluts.sort_values(by=['Supervised Percentage', 'Experiment Label'])
mean_resluts = mean_resluts.reset_index(drop=True)
mean_resluts = mean_resluts.drop([7])

# Calculate bridging coefficients, see report for definition
bridging_coefs = pd.DataFrame()
for i in range(0, len(mean_resluts) - 1, 2):
    part_equireg_results = mean_resluts.loc[mean_resluts.index[i + 1], 'Dice Whole':'Dice Collective']
    part_sup_results = mean_resluts.loc[mean_resluts.index[i], 'Dice Whole':'Dice Collective']
    all_sup_results = mean_resluts.loc[mean_resluts.index[-1], 'Dice Whole':'Dice Collective']

    bridging_coef = (part_equireg_results - part_sup_results) / (all_sup_results - part_sup_results)
    bridging_coef['Experiment Label'] = mean_resluts.loc[mean_resluts.index[i + 1], 'Experiment Label']

    bridging_coefs = bridging_coefs.append(bridging_coef, ignore_index=True)

# Bridging coefficient plots
plt.figure()
plt.grid()
ax = sns.barplot(x='Experiment Label', y='Dice Whole', data=bridging_coefs)
ax.set_ylim([0, 1.0])
plt.ylabel('')
plt.title('Bridging Coefficients for Whole Tumor')
plt.savefig(os.path.join('report_plots', 'bridge_whole.jpg'), bbox_inches='tight')

plt.figure()
plt.grid()
ax = sns.barplot(x='Experiment Label', y='Dice Core', data=bridging_coefs)
ax.set_ylim([0, 1.0])
plt.ylabel('')
plt.title('Bridging Coefficients for Tumor Core')
plt.savefig(os.path.join('report_plots', 'bridge_core.jpg'), bbox_inches='tight')

plt.figure()
plt.grid()
ax = sns.barplot(x='Experiment Label', y='Dice Enhancing', data=bridging_coefs)
ax.set_ylim([0, 1.0])
plt.ylabel('')
plt.title('Bridging Coefficients for Enhancing Tumor')
plt.savefig(os.path.join('report_plots', 'bridge_enhancing.jpg'), bbox_inches='tight')

plt.figure()
plt.grid()
ax = sns.barplot(x='Experiment Label', y='Dice Collective', data=bridging_coefs)
ax.set_ylim([0, 1.0])
plt.ylabel('')
plt.title('Collective Bridging Coefficients')
plt.savefig(os.path.join('report_plots', 'bridge_collective.jpg'), bbox_inches='tight')

# Swarm plots for the raw experiment results
plt.figure(figsize=[9, 4.8])
plt.grid()
ax = sns.swarmplot(x='Experiment Label', y='Dice Whole', data=results, hue='Seed')
ax.set_xticks([1.5, 3.5, 5.5], minor=True)
ax.xaxis.grid(True, which='minor', color='k')
plt.ylabel('Dice Score')
plt.title('Dice Scores for Whole Tumor')
plt.savefig(os.path.join('report_plots', 'raw_whole.jpg'), bbox_inches='tight')

plt.figure(figsize=[9, 4.8])
plt.grid()
ax = sns.swarmplot(x='Experiment Label', y='Dice Core', data=results, hue='Seed')
ax.set_xticks([1.5, 3.5, 5.5], minor=True)
ax.xaxis.grid(True, which='minor', color='k')
plt.ylabel('Dice Score')
plt.title('Dice Scores for Tumor Core')
plt.savefig(os.path.join('report_plots', 'raw_core.jpg'), bbox_inches='tight')

plt.figure(figsize=[9, 4.8])
plt.grid()
ax = sns.swarmplot(x='Experiment Label', y='Dice Enhancing', data=results, hue='Seed')
ax.set_xticks([1.5, 3.5, 5.5], minor=True)
ax.xaxis.grid(True, which='minor', color='k')
plt.ylabel('Dice Score')
plt.title('Dice Scores for Enhancing Tumor')
plt.savefig(os.path.join('report_plots', 'raw_enhancing.jpg'), bbox_inches='tight')

plt.figure(figsize=[9, 4.8])
plt.grid()
ax = sns.swarmplot(x='Experiment Label', y='Dice Collective', data=results, hue='Seed')
ax.set_xticks([1.5, 3.5, 5.5], minor=True)
ax.xaxis.grid(True, which='minor', color='k')
plt.ylabel('Dice Score')
plt.title('Collective Dice Scores')
plt.savefig(os.path.join('report_plots', 'raw_collective.jpg'), bbox_inches='tight')

# Line plots comparing baseline and equireg
plt.figure()
plt.grid(which="both")
plt.ylabel('Dice Score')
plt.title('Mean Dice Scores for Whole Tumor')
ax = sns.lineplot(x='Supervised Percentage', y='Dice Whole', data=results,
                  hue='Method',
                  marker='o')
ax.legend(loc='right')
plt.xscale('log')
ax.xaxis.set_major_formatter(PercentFormatter())
plt.savefig(os.path.join('report_plots', 'line_whole.jpg'), bbox_inches='tight')

plt.figure()
plt.grid(which="both")
plt.ylabel('Dice Score')
plt.title('Mean Dice Scores for Tumor Core')
ax = sns.lineplot(x='Supervised Percentage', y='Dice Core', data=results,
                  hue='Method',
                  marker='o')
ax.legend(loc='right')
plt.xscale('log')
ax.xaxis.set_major_formatter(PercentFormatter())
plt.savefig(os.path.join('report_plots', 'line_core.jpg'), bbox_inches='tight')

plt.figure()
plt.grid(which="both")
plt.ylabel('Dice Score')
plt.title('Mean Dice Scores for Enhancing Tumor')
ax = sns.lineplot(x='Supervised Percentage', y='Dice Enhancing', data=results,
                  hue='Method', marker='o')
ax.legend(loc='right')
plt.xscale('log')
ax.xaxis.set_major_formatter(PercentFormatter())
plt.savefig(os.path.join('report_plots', 'line_enhancing.jpg'), bbox_inches='tight')

plt.figure()
plt.grid(which="both")
plt.ylabel('Dice Score')
plt.title('Mean Collective Dice Scores')
ax = sns.lineplot(x='Supervised Percentage', y='Dice Collective', data=results,
                  hue='Method',
                  marker='o')
ax.legend(loc='right')
plt.xscale('log')
ax.xaxis.set_major_formatter(PercentFormatter())
plt.savefig(os.path.join('report_plots', 'line_collective.jpg'), bbox_inches='tight')

# Bar plot with the equivariance results
plt.figure(figsize=[9, 4.8])
plt.grid(which="both")
ax = sns.barplot(x='Experiment Label', y='Equivariance Dice Collective', data=equivariance_results)
ax.set_ylim([0.4, 1.0])
plt.ylabel('Equivariance Dice Score')
plt.title('Mean Collective Equivariance Dice Scores')
plt.savefig(os.path.join('report_plots', 'equivariance_collective.jpg'), bbox_inches='tight')

plt.show(block=True)
