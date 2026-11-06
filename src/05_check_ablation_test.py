#!/usr/bin/env python3
# encoding: utf-8

import os 
import re
import csv

from tqdm import tqdm

import itertools
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from statsmodels.formula.api import ols, mixedlm
from scipy.stats import kendalltau, rankdata

from utils.indexer import Indexer
from utils.log_utils import log, logTable, c_blue, c_green, c_yellow, c_red, c_purple, c_grey, c_darkgrey, color_palette_list
from utils.log_utils import log, logTable, color_palette_list, bcolors
from utils.yaml_utils import getMetricsLogFile
from utils.plot_distribution import plotDataDistribution, PAPER_FIG_SIZE_IN, PAPER_RC
from models.BatchSizeStudy import CNN_14L_B10, CNN_14L_B25, CNN_14L_B50, CNN_14L_B80
from models import CNN_14L
from utils import output_path, ablation_data_file, deposit_path, read_deposit, analysis_seed
from utils.deposit_data import getDepositAblationLog, getDepositModelData
from utils.compute_switched_probability import computeSwitchedProbability


montecarlo_samples = 20000 # Slow version :) -> 1000000
bootstrap_samples = 20000 # Slow version :) -> 100000


anova_table_analysis = False
mixedlm_analysis = False
percentile_analysis = False
kendall_w_analysis = False
plot_interaction = True
add_extra_data = False # Includes data from the batch size study
log_complete_table = False  # CSV version is stored anyway
# Decision error simulated here (about 30 min). Not the table of the article: this simulation redraws the runs of the
# second step, while the article table comes from scripts/article/ablation_trees.py, where the viewer reuses each draw
compute_decision_error_redrawn = False

analysis_path = './analysis_results/ablation_anova'
number_of_conditions = 9 # Combinations of param changes (3 batch rates with 3 learning rates   )

date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d+")

default_learning_rate = 0.001


def get_date_keys(d):
    """Returns the keys that look like a date."""
    return [k for k in d.keys() if date_pattern.match(k)]

    
def interpret_pvalue(p):
    if p < 0.01:
        return 'Highly Significant'
    elif p < 0.05:
        return 'Significant'
    elif p < 0.1:
        return 'Marginally Significant'
    elif p < 0.2:
        return 'Weakly Significant'
    else:
        return 'Not Significant'

def generate_interaction_plot(df, x_var, y_var, hue_var, title, **kwargs):
    """Generates an interaction plot for the given variables.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x_var (str): Name of the x-axis variable.
        y_var (str): Name of the y-axis variable.
        hue_var (str): Name of the variable to use for color coding.
        title (str): Title of the plot.
    """

    # Drawn at its printed size in the article
    with plt.rc_context(PAPER_RC):
        plt.figure(figsize=PAPER_FIG_SIZE_IN)
        num_hue = df[hue_var].nunique()

        # Add individual points in a single wide column with jitter
        sns.stripplot(data=df, x=x_var, y=y_var, hue=hue_var,
                      alpha=0.35, size=2, palette=color_palette_list[:num_hue],
                      jitter=0.3, legend=False,
                      order=kwargs.get('order'))  # the first plot fixes the categories of the axis

        # Add mean line and summary points separated by hue (on top)
        sns.pointplot(data=df, x=x_var, y=y_var, hue=hue_var, palette=color_palette_list[:num_hue],
            markers='o',        # Ensures circles are used as markers
            linestyles='-',     # Solid line
            linewidth=1.5,      # Line width
            markersize=4,
            zorder=10,          # Ensure it appears on top
            seed=analysis_seed, # Bootstrap of the confidence intervals
            **kwargs
        )
        plt.title(title)
        plt.xlabel(x_var.replace('_', ' ').capitalize())
        plt.ylabel(y_var.replace('_', ' ').capitalize())
        plt.legend(title=hue_var.replace('_', ' ').capitalize())
        plt.tight_layout()
        # plt.show()

        plt.savefig(os.path.join(analysis_path, f'interaction_plot_{x_var}_{y_var}_{hue_var}.pdf'), format="pdf")
    log(f"Saved interaction plot: {os.path.join(analysis_path, f'interaction_plot_{x_var}_{y_var}_{hue_var}.pdf')}", color=bcolors.OKGREEN)


def conditionLabel(condition):
    """Condition '<batch size>_<learning rate>' or '<batch size>-<learning rate>' in the vocabulary of the article."""
    batch_size, learning_rate = re.split(r'[_-]', str(condition), maxsplit=1)
    return f'BS {batch_size}, LR {learning_rate}'


def plotScaterplot(pivot_df, output_path='.', filename='scatterplot_conditions'):
    conditions = list(pivot_df.columns)
    pairs = list(itertools.combinations(conditions, 2))
    os.makedirs(output_path, exist_ok=True)

    for cond1, cond2 in pairs:
        data = pivot_df[[cond1, cond2]].dropna()
        label1, label2 = conditionLabel(cond1), conditionLabel(cond2)
        # Drawn at its printed size in the article
        with plt.rc_context(PAPER_RC):
            plt.figure(figsize=(PAPER_FIG_SIZE_IN[0], PAPER_FIG_SIZE_IN[0]))
            sns.scatterplot(x=cond1, y=cond2, data=data, color=c_blue, s=8)   # palette blue (no seaborn default)
            plt.title(f'Accuracy: {label1} $\\mathit{{vs}}.$ {label2}')
            plt.xlabel(f'{label1} (accuracy, %)')
            plt.ylabel(f'{label2} (accuracy, %)')
            plot_filename = f'{filename}_{cond1}_vs_{cond2}.pdf'
            filepath = os.path.join(output_path, plot_filename)
            plt.tight_layout()
            plt.savefig(filepath, format="pdf", bbox_inches='tight')
            plt.close()

    print(f"Plots saved to: {output_path}")


"""
    Simulates decission tree based on an ablation test with provided data (with three learning rates and three batch sizes levels).
    After the number of iterations provided it computes probability of reaching the best combination of hyperparameters.
"""
def computeDecisionErrorTemplate(df, start_lr = None, start_batch = None, sampling_method = None, iterations=montecarlo_samples):

    # Levels with the highest sample of the step. The comparison is strict and a tie is no win: the tied levels stay
    # together and share the iteration below, so that the result does not depend on the order of the levels
    def bestLevels(levels, df_step, param):
        best_levels = []
        best_acc = -float('inf')
        for val in levels:
            sample = sampling_method(df_step[df_step[param] == val]['accuracy'].values, 1)
            if sample > best_acc:
                best_acc = sample
                best_levels = [val]
            elif sample == best_acc:
                best_levels.append(val)
        return best_levels

    batch_list = df['batch_size'].unique()
    lr_list = df['learning_rate'].unique()
    
    best_overall_trial = df.loc[df['accuracy'].idxmax()]
    best_trial_batch = best_overall_trial['batch_size']
    best_trial_lr = best_overall_trial['learning_rate']
    
    ok_samples = 0
    no_ok_samples = 0

    for it in tqdm(range(iterations), desc="Running iterations", leave=False):
        if start_lr is not None or start_batch is not None:
            
            first_param, second_param = (
                ('batch_size', 'learning_rate') if start_batch is not None else ('learning_rate', 'batch_size')
            )
            first_value = start_batch if start_batch is not None else start_lr

            # print(f"Iteration {it+1}/{iterations}: Starting with {first_param}={first_value}")
            df_first_step = df[df[first_param] == first_value][['index', second_param, 'accuracy']]
            best_first_levels = bestLevels(batch_list if second_param == 'batch_size' else lr_list, df_first_step, second_param)

            for best_first_level in best_first_levels:
                df_second_step = df[df[second_param] == best_first_level][['index', first_param, 'accuracy']]
                best_second_levels = bestLevels(lr_list if first_param == 'learning_rate' else batch_list, df_second_step, first_param)

                for best_second_level in best_second_levels:
                    credit = 1 / (len(best_first_levels) * len(best_second_levels))
                    if first_param == "learning_rate" and best_second_level == best_trial_lr and \
                       second_param == "batch_size" and best_first_level == best_trial_batch or \
                       first_param == "batch_size" and best_second_level == best_trial_batch and \
                       second_param == "learning_rate" and best_first_level == best_trial_lr:
                        ok_samples += credit
                        # tqdm.write("OK sample found!")
                    else:
                        no_ok_samples += credit

                    # tqdm.write(f"Sampled best: {first_param}={best_first_level}, {second_param}={best_second_level} | True best: batch_size={best_trial_batch}, learning_rate={best_trial_lr}")
                    # print(f"Sampled best: {first_param}={best_first_level}, {second_param}={best_second_level} | True best: batch_size={best_trial_batch}, learning_rate={best_trial_lr}")
                    # exit()

    p_ok = ok_samples / (ok_samples + no_ok_samples)
    p_no_ok = no_ok_samples / (ok_samples + no_ok_samples)

    return p_ok, p_no_ok

"""
    Simulates the ablation decission tree: fixing one of the hyperparams chechs all levels of the other one;
    once the best has been selected, checks all levels of the first hyperparam to select the best overall.
    This simulation is made with both monte carlo and bootstrap sampling methods starting with all three levels
    of learning rate and batch size previously trained.

    @see computeDecisionErrorTemplate
"""
def computeAblationDecisionErrorProbability(df):
    log(f"Analysis of ablation decision error probability:", color=bcolors.OKGREEN)

    lr_list = df['learning_rate'].unique()
    batch_list = df['batch_size'].unique()
    
    table_data = [['Fixed Param', 'Start Value', 'MonteCarlo', 'Bootstrap']]
    for start_batch in batch_list:
        log(f"Estimation Ablation Decision error starting with fixed batch {start_batch}:", color=bcolors.OKCYAN)
        
        montecarlo_sampling = lambda sampling_data, n_samples: np.random.normal(loc=np.mean(sampling_data), scale=np.std(sampling_data), size=n_samples)
        montecarlo_p_ok, montecarlo_p_no_ok = computeDecisionErrorTemplate(df=df, start_batch=start_batch, sampling_method=montecarlo_sampling, iterations=montecarlo_samples)
        log(f"  · MonteCarlo ({montecarlo_samples} iterations): p_ok={montecarlo_p_ok:.4f}, p_no_ok={montecarlo_p_no_ok:.4f}")
        
        bootstrap_sampling = lambda sampling_data, n_samples: np.random.choice(sampling_data, size=n_samples, replace=True)
        bootstrap_p_ok, bootstrap_p_no_ok = computeDecisionErrorTemplate(df=df, start_batch=start_batch, sampling_method=bootstrap_sampling, iterations=bootstrap_samples)
        log(f"  · Bootstrap ({bootstrap_samples} iterations): p_ok={bootstrap_p_ok:.4f}, p_no_ok={bootstrap_p_no_ok:.4f}")
        table_data.append(['Batch Size', start_batch, f"{montecarlo_p_no_ok:.4f}", f"{bootstrap_p_no_ok:.4f}"])

    for start_lr in lr_list:
        log(f"Estimation Ablation Decision error  starting with fixed lr {start_lr}:", color=bcolors.OKCYAN)
        
        montecarlo_sampling = lambda sampling_data, n_samples: np.random.normal(loc=np.mean(sampling_data), scale=np.std(sampling_data), size=n_samples)
        montecarlo_p_ok, montecarlo_p_no_ok = computeDecisionErrorTemplate(df=df, start_lr=start_lr, sampling_method=montecarlo_sampling, iterations=montecarlo_samples)
        log(f"  · MonteCarlo ({montecarlo_samples} iterations): p_ok={montecarlo_p_ok:.4f}, p_no_ok={montecarlo_p_no_ok:.4f}")
        
        bootstrap_sampling = lambda sampling_data, n_samples: np.random.choice(sampling_data, size=n_samples, replace=True)
        bootstrap_p_ok, bootstrap_p_no_ok = computeDecisionErrorTemplate(df=df, start_lr=start_lr, sampling_method=bootstrap_sampling, iterations=bootstrap_samples)
        log(f"  · Bootstrap ({bootstrap_samples} iterations): p_ok={bootstrap_p_ok:.4f}, p_no_ok={bootstrap_p_no_ok:.4f}")
        table_data.append(['Learning Rate', start_lr, f"{montecarlo_p_no_ok:.4f}", f"{bootstrap_p_no_ok:.4f}"])


    # Other name than the table of the article (scripts/article/ablation_trees.py), which has different values: see
    # compute_decision_error_redrawn at the top of this file
    logTable(table_data, os.path.join(analysis_path, 'tables'), f'ablation_decision_error_probability_redrawn', screen_log = True)

if __name__ == "__main__":
    os.makedirs(os.path.join(analysis_path, 'tables'), exist_ok=True)

    train_duration = []
    indexer = Indexer()
    ablation_metrics = getDepositAblationLog(deposit_path) if read_deposit else getMetricsLogFile(ablation_data_file)
    ablation_matrix = []
    table_keys = [['index', 'seed','batch_size', 'learning_rate', 'accuracy']]
    
    log_ablation_matrix = []
    log_complete_table_keys = [['index', 'trial', 'seed', 'Epoch', 'time (min)','batch_size', 'learning_rate', 'accuracy']]
    
    def extractMetrics(trial, learning_rate=None, batch_size=None):
        try:
            seed = None if not 'seed' in trial else trial['seed']
            accuracy = trial['accuracy']*100
            batchs = batch_size if batch_size is not None else trial['batch_size']
            learningr = learning_rate if learning_rate is not None else trial['learning_rate']
            index = indexer.get_index(seed)
            
            time = trial['train_duration']/60
            best_epoch = trial['best_epoch']
        except KeyError as e:
            log(f"KeyError: {e} in trial: {trial}", color=bcolors)
            return None, None, 0
        
        log_ablation_row = [index, trial_idx, seed, best_epoch, time, batchs, learningr, accuracy]
        ablation_row = [index, seed, batchs, learningr, accuracy]
        
        return ablation_row, log_ablation_row, time


    for key, iteration in ablation_metrics.items():
        for trial_idx, (condition, trial) in enumerate(iteration.items()):
            ablation_row, log_ablation_row, time = extractMetrics(trial)
            
            ablation_matrix.append(ablation_row)
            log_ablation_matrix.append(log_ablation_row)
            
            train_duration.append(time)
    
    if add_extra_data:
        log(f"Adding extra data from batch size study", color=bcolors.OKCYAN)
        for model in [CNN_14L, CNN_14L_B10, CNN_14L_B25, CNN_14L_B50, CNN_14L_B80]:
            if read_deposit:
                model_metrics = getDepositModelData(deposit_path)[model.__name__]
            else:
                model_metrics = getMetricsLogFile(os.path.join(output_path, model.__name__, f"randomseed_training_metrics.yaml"))
            for key, trial in model_metrics.items():
                ablation_row, log_ablation_row, time = extractMetrics(trial, default_learning_rate, model.batch_size)
                
                ablation_matrix.append(ablation_row)
                log_ablation_matrix.append(log_ablation_row)
                
                train_duration.append(time)


    log_ablation_matrix.sort(key=lambda x: x[0])
    logTable(log_complete_table_keys+log_ablation_matrix, os.path.join(analysis_path, 'tables'), f'log_ablation_matrix', screen_log = log_complete_table)


    average_time_min = sum(train_duration)/len(train_duration)

    decimal_sec = average_time_min - int(average_time_min)
    seconds = decimal_sec * 60
    log(f"Average duration of each individual trial (out of {len(ablation_matrix)}) is of {int(average_time_min)} min {seconds:.2f} s")

    # Create dataframe and set variables as categoric (there are two categories for learning rate
    # and three for batch)
    df = pd.DataFrame(ablation_matrix, columns=table_keys[0])
    # df['index'] = df['index'].astype('category')
    # df['batch_size'] = df['batch_size'].astype('category')
    # df['learning_rate'] = df['learning_rate'].astype('category')
    if not add_extra_data: # Extra data do not contain these 9 conditions... sorryn't
        df_removed = df[df.groupby('index', observed=False)['index'].transform('count') < number_of_conditions]
        df = df[df.groupby('index', observed=False)['index'].transform('count') == number_of_conditions]
        log(f"Removed some indexes ({len(df_removed['index'].unique())}) as they do not have {number_of_conditions} trials: {df_removed['index'].unique()}")
        log(f"Model estimation based on {len(df['index'].unique())} random-condition iterations ({number_of_conditions} trials each): {len(df)} trials used of the {len(ablation_matrix)} logged.")
    else:
        log(f"Model estimation based on {len(df['index'].unique())} random-condition iterations; with a total of {len(ablation_matrix)} trials.")
    if len(df['index'].unique()) < 1:
        log(f"[WARN] Still not enough complete iterations to be processed", color=bcolors.WARNING)
        exit()

    csv_path = os.path.join(analysis_path,'tables',f'{"extended_" if add_extra_data else ""}ablation_matrix.csv')
    df.to_csv(csv_path, encoding='utf-8')

    # First model with just principal effects_
    variable_index = 'C(index)' # C(index) treats index as a categoric variable
    variable_batch = 'C(batch_size)'
    variable_learnr = 'C(learning_rate)'

    ##################################
    ##  Regression with ANOVA table ##
    ##################################
    if anova_table_analysis:
        # pd.set_option('display.float_format', '{:.25e}'.format)
        def annova_analysis(df, variable_index, variable_batch, variable_learnr):
            formula_main = f'accuracy ~ {variable_index} + {variable_batch} + {variable_learnr}'

            # Principal effects with double interactions_
            formula_double = (f'accuracy ~ {variable_index} + {variable_batch} + {variable_learnr} '
                            f'+ {variable_index}:{variable_batch} + {variable_index}:{variable_learnr} + {variable_batch}:{variable_learnr}'
                            )
            
            # Principal effects with double interactions with triple interactions for just batch and lr:
            formula_triple = f'accuracy ~ C(seed) + {variable_batch} * {variable_learnr}' # * {variable_index}'

            formula_noseed = f'accuracy ~ {variable_batch} * {variable_learnr}'

            for formula in [formula_main, formula_double, formula_triple, formula_noseed]:
                model = ols(formula, data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                
                # log(f"\nResulting model with {formula}:", color=bcolors.OKCYAN)
                # log(model.summary())
                
                log(f"\nANOVA table from model {formula}:", color=bcolors.OKCYAN)
            
                anova_table['Interpretation'] = anova_table['PR(>F)'].apply(interpret_pvalue)
                log(anova_table)
        
        log(f"\nPerform ANOVA analysis with complete data:", color=bcolors.OKGREEN)
        annova_analysis(df, variable_index, variable_batch, variable_learnr)
        # log(f"\nPerform ANOVA analysis with filtered data (no 70 batch):", color=bcolors.OKGREEN)
        # annova_analysis(df[df['batch_size'] != 70].copy(), variable_index, variable_batch, variable_learnr)
        
        def annova_repeated_measures(df, subject, factor_list):
            log(f"\nPerforming ANOVA Repeated Measures with complete model (subject: {subject}; factor list: {factor_list}):", color=bcolors.OKCYAN)
            aovrm = AnovaRM(
                data=df,
                depvar='accuracy',
                subject=subject,
                within=factor_list,
            )
            anova_results = aovrm.fit()
            log(anova_results)

            import pingouin as pg
            # Has too many levels on each factor, not allowed by sphericity test implemented in this library
            # spher_results = pg.sphericity(
            #     data=df,
            #     dv='accuracy',
            #     subject=subject,
            #     within=factor_list
            # )
            # log(f"Spher test results:\n{spher_results}")

            aov = pg.rm_anova(
                data=df,
                dv='accuracy',
                within=factor_list,
                subject=subject,
                correction=True,  # Applies the correction if needed
                detailed=True     # Shows the full table
            )
            log(f"Greenhouse-Geisser annova correcte: \n{aov}")
        
        annova_repeated_measures(df, subject='seed', factor_list=['batch_size', 'learning_rate'])
        # annova_repeated_measures(df, subject='batch_size', factor_list=['seed', 'learning_rate'])
        # annova_repeated_measures(df, subject='learning_rate', factor_list=['seed', 'batch_size'])

    ###############################
    ###   Mixed effects models   ##
    ###############################
    if mixedlm_analysis:
        # groups=df["index"] gives each index its own base intercept (random effect). The formula holds the fixed effects
        # only: patsy would read '(1 | index)' as one more fixed effect. Fitted with ML (reml=False) so that AIC and BIC exist.
        fixed_effects_formula = f'accuracy ~ {variable_batch} * {variable_learnr}'
        log(f"\nFitting Model with Random Intercepts from {fixed_effects_formula}:", color=bcolors.OKCYAN)
        try:
            model_intercept = mixedlm(fixed_effects_formula, data=df, groups=df["index"]).fit(reml=False)
            log(model_intercept.summary())
            log(f"AIC: {model_intercept.aic:.2f}, BIC: {model_intercept.bic:.2f}")
        except Exception as e:
            log(f"Failed to fit model: {e}", color=bcolors.ERROR)

        # Random Slopes Model:
        # Model with interactions and random slopes for both variables:
        random_effects_formula = f'{variable_batch} + {variable_learnr}' # No '1 +' for random slopes, implies random intercept too if not specified
        
        formula_full_random = f'accuracy ~ {variable_batch} * {variable_learnr} + ({variable_batch} + {variable_learnr} | index)'
        log(f"\nFitting Model with Random Slopes with fixed effects ({fixed_effects_formula}) and random effects ({random_effects_formula}):", color=bcolors.OKCYAN)
        try:
            model_rs = mixedlm(fixed_effects_formula, data=df, groups=df["index"], re_formula=random_effects_formula).fit(reml=False)
            log(model_rs.summary())

            log(f"AIC: {model_rs.aic:.2f}, BIC: {model_rs.bic:.2f}")
            log("\nINTERPRETATION:", color=bcolors.WARNING)
            log("Compare AIC/BIC values. If Model 2's are lower, it's a better fit.")
            log("Look at the 'Random-effects Parameters' table for 'C(batch_size) Var' and 'C(learning_rate) Var'.")
            log("If these variances are > 0, you have statistically shown that the effect of these hyperparameters is UNSTABLE across different seeds.")
        except Exception as e:
            log(f"Failed to fit random slopes model: {e}", color=bcolors.ERROR)
            log("This can happen if the data doesn't support such a complex model. The simpler random intercept model might be the most appropriate conclusion.", color=bcolors.WARNING)


    #############################
    ##   Percentile analysis   ##
    #############################
    """
        Hypotheis:
            If a seed (index) produces a low (or high) performance with a
            set of hyperparameters, will that same seed also produce a low 
            (or high) performance when other hyperparameters are used?
    """
    if percentile_analysis:
        df_percentile = copy.deepcopy(df) # Avoid modifying the original df
        log(f"\nPerform Percentile Analysis of Seed Consistency Across Hyperparameters", color=bcolors.OKCYAN)
        
        df_percentile['condition'] = df_percentile['batch_size'].astype(str) + '_' + df_percentile['learning_rate'].astype(str)
        df_percentile['percentile_within_condition'] = df_percentile.groupby('condition')['accuracy'].rank(pct=True, ascending=False)
        percentile_table = df_percentile.pivot_table(index='seed', columns='condition', values='percentile_within_condition')
        # Percentile table (ascending=False: from 1/n, the best seed performance in that condition, to 1, the worst)
        log(f"Percentile talbe indexed by seed:\n{percentile_table}")

        # Compute the correlation matrix between conditions
        correlations = percentile_table.corr(method='spearman')

        total_repeated_table = (df_percentile.groupby('condition')['accuracy'].apply(lambda x: x.value_counts().loc[lambda v: v > 1].sum())
                                .reset_index(name='total_repetitions'))
        values_repeated_ties = (df_percentile.groupby('condition')['percentile_within_condition'].apply(lambda x: x.value_counts().loc[lambda v: v > 1].count())
                                .reset_index(name='repeated_percentiles'))
        unique_percentiles = (df_percentile.groupby('condition')['percentile_within_condition'].nunique()
                                .reset_index(name='unique_percentiles'))
        accuracy_amplitude = (df_percentile.groupby('condition')['accuracy'].agg(lambda x: x.max() - x.min())
                                .reset_index(name='accuracy_amplitude'))
        
        summary_table = total_repeated_table \
            .merge(values_repeated_ties, on='condition') \
            .merge(unique_percentiles, on='condition') \
            .merge(accuracy_amplitude, on='condition')
        print(summary_table)
        with open(f'{analysis_path}/tables/percentile_repetitions.tex', 'w', encoding='utf-8') as f:
            f.write(summary_table.to_latex())

        pivot_df = df_percentile.pivot_table(index='seed', columns='condition', values='accuracy')
        condition_list = list(pivot_df.columns)
        plot_df = pivot_df[condition_list].dropna()

        plotScaterplot(pivot_df, output_path=analysis_path, filename='scatterplot_conditions')
        # g = sns.PairGrid(plot_df)
        # g.map_lower(sns.scatterplot)  # Only lower part of the matrix, remove repeated and diagonal
        # plt.suptitle('Accuracy scatterplots between conditions', y=1.02)
        # plt.show()

        from tabulate import tabulate 
        log(f"\nSpearman Correlation (Seed Ranks) between Hyperparameter Combinations:")
        log(tabulate(correlations , headers='keys', tablefmt='fancy_grid'))

        # Just one part of the matrix (upper triangle) to avoid redundancy
        mask = np.triu(np.ones(correlations.shape), k=1).astype(bool)
        correlations_no_diag = correlations.where(mask)
        with open(f'{analysis_path}/tables/percentile_correlations.tex', 'w', encoding='utf-8') as f:
            f.write(correlations_no_diag.to_latex(float_format="%.4f".__mod__))


        corr_matrix_values = correlations.to_numpy(copy=True)
        np.fill_diagonal(corr_matrix_values, np.nan)
        corr_matrix = pd.DataFrame(
            corr_matrix_values,
            index=correlations.index,
            columns=correlations.columns,
        )
        corr_matrix['batch_size'] = [c.split('_')[0] for c in corr_matrix.index]
        corr_matrix['learning_rate'] = [c.split('_')[1] for c in corr_matrix.index]

        # batch_corr_means = corr_matrix.groupby('batch_size').mean(numeric_only=True)
        batch_corr_means = corr_matrix.groupby('batch_size').median(numeric_only=True)
        print("\nMedian of correlations by batch size:")
        print(batch_corr_means)
        with open(f'{analysis_path}/tables/percentile_correlations_by_batch.tex', 'w', encoding='utf-8') as f:
            f.write(batch_corr_means.to_latex(float_format="%.4f".__mod__))


        # lr_corr_means = corr_matrix.groupby('learning_rate').mean(numeric_only=True)
        lr_corr_means = corr_matrix.groupby('learning_rate').median(numeric_only=True)
        print("\nMedian of correlations by learning rate:")
        print(lr_corr_means)
        with open(f'{analysis_path}/tables/percentile_correlations_by_lr.tex', 'w', encoding='utf-8') as f:
            f.write(lr_corr_means.to_latex(float_format="%.4f".__mod__))


        log("\nOverall Interpretation of Seed Consistency", color=bcolors.WARNING)
        log("If 'Spearman Correlation (Seed Ranks)' values are high (e.g., > 0.7-0.8) across most configurations,")
        log("it suggests that a seed's performance rank is largely consistent regardless of hyperparameter choice.")
        log("This would mean that 'good' seeds tend to be good, and 'bad' seeds tend to be bad, across different setups.")
        log("If correlations are low, it implies that the effect of hyperparameters on accuracy varies significantly by seed (i.e., good seeds for one config might be bad for another).")


        # A tie is no win for any condition: the conditions tied for the best of a seed share it (fractional count), so
        # that the percentages sum to 100 and do not depend on the order of the conditions
        is_seed_best = df['accuracy'] == df.groupby('index')['accuracy'].transform('max')
        best_config_credit = is_seed_best / is_seed_best.groupby(df['index']).transform('sum')
        best_config_counts = best_config_credit[is_seed_best].groupby([df['batch_size'], df['learning_rate']]).sum().reset_index(name='count')
        total_seeds = df['index'].nunique()
        best_config_counts['percentage'] = (best_config_counts['count'] / total_seeds) * 100

        log("\nPercentage of Seeds for which each HP Combination yielded the HIGHEST Accuracy:")
        log_df = best_config_counts.sort_values(by='percentage', ascending=False).round(3)
        log(log_df)
        log("Interpretation: If one combination consistently appears as 'best' (e.g., >50%), then the best HP choice is robust to seed changes. If percentages are spread out, it implies the 'best' combination is highly dependent on the seed.", color=bcolors.WARNING)

        with open(f'{analysis_path}/tables/best_seed_percentage.tex', 'w', encoding='utf-8') as f:
            f.write(log_df.to_latex())

        # Create labels for the x-axis that combine batch_size and learning_rate
        best_config_counts['combo_label'] = best_config_counts.apply(
            lambda row: f"BS {int(row['batch_size'])}\nLR {row['learning_rate']}", axis=1
        )
        # Drawn at its printed size in the article
        with plt.rc_context(PAPER_RC):
            plt.figure(figsize=(PAPER_FIG_SIZE_IN[0], 2.4))
            sns.barplot(
                data=best_config_counts.sort_values(by='percentage', ascending=False),
                x='combo_label',
                y='percentage',
                hue='combo_label',
                palette=color_palette_list[:len(best_config_counts)], # Choose a color palette
                saturation=1, # keep the palette hex as is (seaborn desaturates bars to 0.75 by default)
                legend=False,
            )
            plt.title('Percentage of Seeds for which\neach HP Combination is "Best"')
            plt.xlabel('Hyperparameter Combination')
            plt.ylabel('Percentage of Seeds (%)')
            plt.xticks(fontsize=6.5)
            # plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_path, 'best_hp_combination_per_seed_bar_chart.pdf'), format='pdf')

    ########################################################################
    ##   Analysis: Kendall's W (Consistency of HP Combination Rankings)   ##
    ########################################################################
    if kendall_w_analysis:
        def kendall_w_analysis(df):
            log(f"\nPerform Kendall's W (Consistency of HP Combination Rankings)", color=bcolors.OKCYAN)
            log("Assessing the agreement among seeds on the overall ranking of hyperparameter combinations.")

            # Combine batch_size and learning_rate into a single column for pivoting
            df['hp_combo'] = df.apply(lambda row: f"BS_{int(row['batch_size'])}_{row['learning_rate']}", axis=1)


            # Pivot the data: index (seed) as rows, hp_combo as columns, accuracy as values
            # This creates a matrix where each row is a seed's accuracies for all 9 combinations
            pivot_df = df.pivot_table(index='index', columns='hp_combo', values='accuracy')

            # For each row (seed), rank the columns (HP combinations).
            # We want higher accuracy to get a better rank (e.g., rank 1). So, ascending=False.
            # 'average' method handles ties by assigning the average rank.
            ranked_df = pivot_df.rank(axis=1, ascending=False, method='average')

            # Convert to numpy array for calculation
            ranks_matrix = ranked_df.values

            # Check for NaN values, which shouldn't happen if df_removed logic works correctly
            if np.isnan(ranks_matrix).any():
                log("[WARN] NaN values found in rank matrix. This indicates incomplete data for some seeds. Kendall's W might be inaccurate.", color=bcolors.WARNING)
                # You might want to filter out rows with NaNs if this happens unexpectedly
                # ranks_matrix = ranks_matrix[~np.isnan(ranks_matrix).any(axis=1)]

            m = ranked_df.shape[0] # Number of raters (seeds)
            n = ranked_df.shape[1] # Number of items (HP combinations)

            if m < 2 or n < 2:
                log(f"[ERROR] Cannot calculate Kendall's W: Not enough seeds ({m}) or HP combinations ({n}).", color=bcolors.ERROR)
                kendall_w = np.nan
            else:
                # Calculate Kendall's W using the formula: W = (12 * S) / (m^2 * (n^3 - n))
                # Where S = sum of squared deviations of column rank sums from their mean
                
                # Sum of ranks for each HP combination across all seeds
                sum_of_ranks_per_hp_combo = ranked_df.sum(axis=0)

                # Mean of all ranks across all combinations and seeds (should be (n+1)/2)
                mean_rank_overall = (n + 1) / 2.0 # Theoretical mean rank

                # Calculate S: Sum of squared deviations of column sums of ranks from the mean sum of ranks
                S = np.sum((sum_of_ranks_per_hp_combo - m * mean_rank_overall)**2)

                # Denominator for Kendall's W
                kendall_w_denominator = m**2 * (n**3 - n)

                if kendall_w_denominator == 0: # Should not happen if m>=2, n>=2
                    kendall_w = np.nan
                else:
                    kendall_w = (12 * S) / kendall_w_denominator

            log(f"Number of Seeds (Raters): {m}")
            log(f"Number of HP Combinations (Items): {n}")
            log(f"Kendall's W: {kendall_w:.4f}")
            
        log(f"\nKendall's W Analysis with complete data:", color=bcolors.OKGREEN)
        kendall_w_analysis(df)
        log(f"\nKendall's W Analysis with  with filtered data (no 70 batch):", color=bcolors.OKGREEN)
        kendall_w_analysis(df[df['batch_size'] != 70].copy())
        log("\nInterpretation:", color=bcolors.WARNING)
        log("Kendall's W ranges from 0 to 1, where 1 indicates perfect agreement among seeds on the ranking of HP combinations,")
        log("and 0 indicates no agreement.")
        log("A low Kendall's W value (e.g., < 0.3) suggests that the overall ranking of hyperparameter combinations")
        log("is highly inconsistent across different seeds. This strongly supports the idea that the 'best' (and worst, and in-between)")
        log("combination(s) might change depending on the seed used, making conclusions less generalizable.")

    # Find the row with the maximum accuracy in the entire DataFrame
    best_overall_trial = df.loc[df['accuracy'].idxmax()]

    log("\nOverall Best Hyperparameter Combination", color=bcolors.OKCYAN)
    log("The hyperparameter combination that achieved the single highest accuracy across all seeds and trials is:")
    log(f"  Accuracy: {best_overall_trial['accuracy']:.2f}%")
    log(f"  Batch Size: {int(best_overall_trial['batch_size'])}")
    log(f"  Learning Rate: {best_overall_trial['learning_rate']:.4f}")
    log(f"  Corresponding Seed (Index): {int(best_overall_trial['index'])}")

    # If you also want to see the average performance of this specific combination:
    avg_accuracy_of_best_combo = df[(df['batch_size'] == best_overall_trial['batch_size']) & 
                                    (df['learning_rate'] == best_overall_trial['learning_rate'])]['accuracy'].mean()

    log(f"\nAverage accuracy for this combination (across all seeds): {avg_accuracy_of_best_combo:.2f}%")

    log("\nInterpretation:", color=bcolors.WARNING)
    log("This result tells you the single best performance recorded. However, remember that due to seed variability (as shown ")
    log("by the percentile analysis), this specific result might not be consistently achieved if you re-run the same combination")
    log("with a different seed.")
    log("The average accuracy for this combination gives a more robust estimate of its typical performance.")
    #########################
    ##  Interaction plots  ##
    #########################
    if plot_interaction:
        np.random.seed(analysis_seed)  # jitter of the points
        generate_interaction_plot(df, 'batch_size', 'accuracy', 'learning_rate', 'Interaction Plot:\n' r'Batch Size $\mathit{vs}.$ Accuracy by Learning Rate')
        generate_interaction_plot(df, 'learning_rate', 'accuracy', 'batch_size', 'Interaction Plot:\n' r'Learning Rate $\mathit{vs}.$ Accuracy by Batch Size')
        
        # Order indexes based on accuracy for a given learning rate level
        df['index'] = df['index'].astype('category')
        subset = df[df['batch_size'] == 10]
        ordered_categories = subset.groupby('index', observed=False)['accuracy'].mean().sort_values().index.tolist()
        generate_interaction_plot(df, 'index', 'accuracy', 'batch_size', r'Interaction Plot: Index $\mathit{vs}.$ Accuracy by Batch Size', order=ordered_categories)
        
        # Order indexes based on accuracy for a given learning rate level
        df['index'] = df['index'].astype('category')
        subset = df[df['learning_rate'] == 0.01]
        ordered_categories = subset.groupby('index', observed=False)['accuracy'].mean().sort_values().index.tolist()
        generate_interaction_plot(df, 'index', 'accuracy', 'learning_rate', r'Interaction Plot: Index $\mathit{vs}.$ Accuracy by Learning Rate', order=ordered_categories)


    # Plot distributions to check error probability
    metrics_data = {}
    all_models = set()
    for _, row in df.iterrows():
        condition = f"{int(row['batch_size'])}-{row['learning_rate']}"
        key = str(row['seed'])
        if condition not in metrics_data:
            metrics_data[condition] = {}
        metrics_data[condition][key] = {'accuracy': row['accuracy']/100.0}
        all_models.add(condition)


    log(f"\nPlotting Data Distributions for Ablation Study", color=bcolors.OKCYAN)
    plotDataDistribution(metrics_data=metrics_data,
                        models_plot_list=[
                        ['10-0.001', '10-0.005', '10-0.01'],
                        ['40-0.001', '40-0.005', '40-0.01'],
                        ['70-0.001', '70-0.005', '70-0.01'],
                        ['10-0.001', '40-0.001', '70-0.001'],
                        ['10-0.005', '40-0.005', '70-0.005'],
                        ['10-0.01', '40-0.01', '70-0.01'],
                        sorted(all_models)],
                            color_list=[
                            [c_purple, c_yellow, c_grey],
                            [c_purple, c_yellow, c_grey],
                            [c_purple, c_yellow, c_grey],
                            [c_purple, c_yellow, c_grey],
                            [c_purple, c_yellow, c_grey],
                            [c_purple, c_yellow, c_grey],
                            color_palette_list],
                            analysis_path = analysis_path,
                            single_plots = False,
                            paper_size = True)
    

    accuracy_data = {}
    for model, data in metrics_data.items():
        accuracy_data[model] = [entry['accuracy']*100 for entry in metrics_data[model].values()]
    np.random.seed(analysis_seed)
    computeSwitchedProbability(accuracy_data, ['10-0.001', '10-0.005', '10-0.01',
                                                '40-0.001', '40-0.005', '40-0.01',
                                                '70-0.001', '70-0.005', '70-0.01'], analysis_path)


    if compute_decision_error_redrawn:
        np.random.seed(analysis_seed)
        computeAblationDecisionErrorProbability(df)