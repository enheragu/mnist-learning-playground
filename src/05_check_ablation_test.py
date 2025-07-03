#!/usr/bin/env python3
# encoding: utf-8

import os 
import re
import csv

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm
from scipy.stats import kendalltau, rankdata


from utils.log_utils import log, logTable, color_palette_list, bcolors
from utils.yaml_utils import getMetricsLogFile
from utils.plot_distribution import plotDataDistribution
from models.BatchSizeStudy import CNN_14L_B10, CNN_14L_B25, CNN_14L_B50, CNN_14L_B80
from models import CNN_14L
from utils import output_path

anova_table_analysis = True
mixedlm_analysis = False
percentile_analysis = True
kendall_w_analysis = True
plot_interaction = False
add_extra_data = False # Inculdes data from batch size study
log_complete_table = False  # CSV version is stored anyway

analysis_path = './analysis_results/ablation_anova'
data_file = f'{output_path}/CNN_14L_Ablation/training_metrics.yaml'
number_of_conditions = 9 # Combinations of param changes (3 batch rates with 3 learning rates   )

date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d+")

default_learning_rate = 0.001


def get_date_keys(d):
    """Devuelve las keys que tienen pinta de fecha."""
    return [k for k in d.keys() if date_pattern.match(k)]

class Indexer:
    def __init__(self):
        self.data2index = {}
        self.next_index = 0

    def get_index(self, dato = None):
        if dato not in self.data2index:
            self.data2index[dato] = self.next_index
            self.next_index += 1
        elif dato is None:
            self.data2index[dato] = self.next_index
            self.next_index += 1
        return self.data2index[dato]
    
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

    plt.figure(figsize=(20, 12))
    num_hue = df[hue_var].nunique()
    sns.pointplot(data=df, x=x_var, y=y_var, hue=hue_var, palette=color_palette_list[:num_hue],    
        markers='o',        # Asegura que se usen círculos como puntos
        linestyles='-',     # Línea sólida
        **kwargs
    )
    plt.title(title)
    plt.xlabel(x_var.replace('_','').title())
    plt.ylabel(y_var.replace('_','').title())
    plt.legend(title=hue_var)
    plt.tight_layout()
    # plt.show()
    
    plt.savefig(os.path.join(analysis_path, f'interaction_plot_{x_var}_{y_var}_{hue_var}.png'))



if __name__ == "__main__":
    os.makedirs(analysis_path, exist_ok=True)

    train_duration = []
    indexer = Indexer()
    ablation_metrics = getMetricsLogFile(data_file)
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
            model_metrics = getMetricsLogFile(os.path.join(output_path, model.__name__, f"randomseed_training_metrics.yaml"))
            for key, trial in model_metrics.items():
                ablation_row, log_ablation_row, time = extractMetrics(trial, default_learning_rate, model.batch_size)
                
                ablation_matrix.append(ablation_row)
                log_ablation_matrix.append(log_ablation_row)
                
                train_duration.append(time)


    log_ablation_matrix.sort(key=lambda x: x[0])
    logTable(log_complete_table_keys+log_ablation_matrix, analysis_path, f'log_ablation_matrix', screen_log = log_complete_table)

    with open(os.path.join(analysis_path,f'{"extended_" if add_extra_data else ""}ablation_matrix.csv'), mode="w", newline="") as file:
        csv.writer(file).writerows(table_keys+ablation_matrix)

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
        log(f"Model estimation based on {len(df['index'].unique())} random-condition iterations ({number_of_conditions} per eac; total of {len(ablation_matrix)} trials).")
    else:
        log(f"Model estimation based on {len(df['index'].unique())} random-condition iterations; with a total of {len(ablation_matrix)} trials.")
    if len(df['index'].unique()) < 1:
        log(f"[WARN] Still not enough complete iterations to be processed", color=bcolors.WARNING)
        exit()


    # First model with just principal effects_
    variable_index = 'C(index)' # C(index) treats index as a categoric variable
    variable_batch = 'C(batch_size)'
    variable_learnr = 'C(learning_rate)'

    ##################################
    ##  Regresion with ANOVA table  ##
    ##################################
    if anova_table_analysis:
        def annova_analysis(df, variable_index, variable_batch, variable_learnr):
            formula_main = f'accuracy ~ {variable_index} + {variable_batch} + {variable_learnr}'

            # Principal effects with double interactions_
            formula_double = (f'accuracy ~ {variable_index} + {variable_batch} + {variable_learnr} '
                            f'+ {variable_index}:{variable_batch} + {variable_index}:{variable_learnr} + {variable_batch}:{variable_learnr}'
                            )
            
            # Principal effects with double interactions with triple interactions:
            formula_triple = f'accuracy ~ {variable_batch} * {variable_learnr}' # * {variable_index}'

            for formula in [formula_main, formula_double, formula_triple]:
                model = ols(formula, data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                
                # log(f"\nResulting model with {formula}:", color=bcolors.OKCYAN)
                # log(model.summary())
                
                log(f"\nANOVA table from model {formula}:", color=bcolors.OKCYAN)
            
                anova_table['Interpretation'] = anova_table['PR(>F)'].apply(interpret_pvalue)
                log(anova_table)

        log(f"\nPerform ANOVA analysis with complete data:", color=bcolors.OKGREEN)
        annova_analysis(df, variable_index, variable_batch, variable_learnr)
        log(f"\nPerform ANOVA analysis with filtered data (no 70 batch):", color=bcolors.OKGREEN)
        annova_analysis(df[df['batch_size'] != 70].copy(), variable_index, variable_batch, variable_learnr)

    ###############################
    ###   Mixed effects models   ##
    ###############################
    if mixedlm_analysis:
        # El término '1|index' le dice al modelo que cada index tiene su propia intercepción base (efecto aleatorio)
        fixed_effects_formula = f'accuracy ~ {variable_batch} * {variable_learnr} + (1 | index)' 
        log(f"\nFitting Model with Random Intercepts from {fixed_effects_formula}:", color=bcolors.OKCYAN)
        try:
            model_intercept = mixedlm(fixed_effects_formula, data=df, groups=df["index"]).fit()
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
            model_rs = mixedlm(fixed_effects_formula, data=df, groups=df["index"], re_formula=random_effects_formula).fit()
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
        log(f"\nPerform Percentile Analysis of Seed Consistency Across Hyperparameters", color=bcolors.OKCYAN)
        overall_avg_accuracy_per_index = df.groupby('index')['accuracy'].mean().sort_values(ascending=False)
        # Rank the indexes based on their average accuracy, the better the percent rank, the better the index in
        # terms of accuracy.
        index_overall_percent_rank = overall_avg_accuracy_per_index.rank(pct=True, ascending=False)

        # Prepare to iterate each batch size and learning rate combination
        unique_hyperparam_configs = df[['batch_size', 'learning_rate']].drop_duplicates()
        consistency_results = []

        for _, config_row in unique_hyperparam_configs.iterrows():
            bs = config_row['batch_size']
            lr = config_row['learning_rate']
            subset_df = df[(df['batch_size'] == bs) & (df['learning_rate'] == lr)].copy() # Get this specific hiperparam configuration

            if not subset_df.empty:
                subset_df_sorted = subset_df.sort_values(by='accuracy', ascending=False)
                # Get percentil
                subset_df_sorted['percent_rank_config'] = subset_df_sorted['accuracy'].rank(pct=True, ascending=False)

                # Unir con el rank general de la semilla
                subset_df_sorted['overall_percent_rank'] = subset_df_sorted['index'].map(index_overall_percent_rank)

                # Calcular la diferencia (o correlación) entre el rank general y el rank específico
                # Una pequeña diferencia (o alta correlación) sugiere consistencia
                rank_difference = (subset_df_sorted['percent_rank_config'] - subset_df_sorted['overall_percent_rank']).abs().mean()
                
                # Una métrica más formal: Coeficiente de correlación de Spearman (rango)
                # Alta correlación de Spearman (>0.7-0.8) indica que las semillas mantienen su orden.
                # Baja correlación (cercana a 0) indica que el orden de las semillas varía mucho.
                spearman_corr = subset_df_sorted['percent_rank_config'].corr(subset_df_sorted['overall_percent_rank'], method='spearman')

                consistency_results.append({
                    'batch_size': bs,
                    'learning_rate': lr,
                    'Mean Absolute Rank Difference': rank_difference,
                    'Spearman Correlation (Seed Ranks)': spearman_corr
                })
                # log(f"Config BS={bs}, LR={lr}: Avg Rank Diff={rank_difference:.4f}, Spearman Corr={spearman_corr:.4f}")
            else:
                log(f"No data for config BS={bs}, LR={lr}. Skipping.", color=bcolors.WARNING)


        consistency_df = pd.DataFrame(consistency_results)
        log(consistency_df)

        log("\nOverall Interpretation of Seed Consistency", color=bcolors.WARNING)
        log("If 'Spearman Correlation (Seed Ranks)' values are high (e.g., > 0.7-0.8) across most configurations,")
        log("it suggests that a seed's performance rank is largely consistent regardless of hyperparameter choice.")
        log("This would mean that 'good' seeds tend to be good, and 'bad' seeds tend to be bad, across different setups.")
        log("If correlations are low, it implies that the effect of hyperparameters on accuracy varies significantly by seed (i.e., good seeds for one config might be bad for another).")


        best_config_per_seed = df.loc[df.groupby('index')['accuracy'].idxmax()]
        best_config_counts = best_config_per_seed.groupby(['batch_size', 'learning_rate']).size().reset_index(name='count')
        total_seeds = df['index'].nunique()
        best_config_counts['percentage'] = (best_config_counts['count'] / total_seeds) * 100

        log("\nPercentage of Seeds for which each HP Combination yielded the HIGHEST Accuracy:")
        log(best_config_counts.sort_values(by='percentage', ascending=False).round(3))
        log("Interpretation: If one combination consistently appears as 'best' (e.g., >50%), then the best HP choice is robust to seed changes. If percentages are spread out, it implies the 'best' combination is highly dependent on the seed.", color=bcolors.WARNING)


        plt.figure(figsize=(12, 7))
        # Create labels for the x-axis that combine batch_size and learning_rate
        best_config_counts['combo_label'] = best_config_counts.apply(
            lambda row: f"BS:{int(row['batch_size'])}, LR:{row['learning_rate']}", axis=1
        )
        sns.barplot(
            data=best_config_counts.sort_values(by='percentage', ascending=False),
            x='combo_label',
            y='percentage',
            palette='viridis' # Choose a color palette
        )
        plt.title('Percentage of Seeds for which each HP Combination is "Best"')
        plt.xlabel('Hyperparameter Combination')
        plt.ylabel('Percentage of Seeds (%)')
        plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_path, 'best_hp_combination_per_seed_bar_chart.png'))

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
        generate_interaction_plot(df, 'batch_size', 'accuracy', 'learning_rate', 'Interation Plot: Batch Size vs Accuracy by Learning Rate')
        
        # Order indexes based on accuracy for a given learning rate level
        df['index'] = df['index'].astype('category')
        subset = df[df['batch_size'] == 40]
        ordered_categories = subset.groupby('index', observed=False)['accuracy'].mean().sort_values().index.tolist()
        generate_interaction_plot(df, 'index', 'accuracy', 'batch_size', 'Interation Plot: Index vs Accuracy by Batch Size', order=ordered_categories)
        
        # Order indexes based on accuracy for a given learning rate level
        df['index'] = df['index'].astype('category')
        subset = df[df['learning_rate'] == 0.001]
        ordered_categories = subset.groupby('index', observed=False)['accuracy'].mean().sort_values().index.tolist()
        generate_interaction_plot(df, 'index', 'accuracy', 'learning_rate', 'Interation Plot: Index vs Accuracy by Learning Rate', order=ordered_categories)
