#!/usr/bin/env python3
# encoding: utf-8

import math
import numpy as np
import scipy.stats as stats

from plot_distribution import getAllModelData

montecarlo_samples = 10000000
bootstrap_samples = 100000
"""
    Given 2 sets of data it computes the probability of, when getting one random sample
    from each of them, the order is switched:
        So having data from g1 and g2 and mean(g1) > mean(g2); 
        computes the p(sample2) > p(sample1)
    The probability is based on a MonteCarlo simulation (with n_samples) based on an 
    approximation based on a Normal distribution of the provided data.
"""
def computeMonteCarloSwitchedProbability(g1_data, g2_data, g1_name = "group1", g2_name = "group2", n_samples = montecarlo_samples):
    g1_mean = np.mean(g1_data)
    g2_mean = np.mean(g2_data)

    g1_std = np.std(g1_data)
    g2_std = np.std(g2_data)

    samples1 = np.random.normal(g1_mean, g1_std, n_samples)
    samples2 = np.random.normal(g2_mean, g2_std, n_samples)
    
    count1 = np.sum(samples1 > samples2)
    probability1 = count1 / n_samples

    # print(f"\t[MonteCarlo] {g1_name} samples: {len(g1_data)}")
    # print(f"\t[MonteCarlo] {g2_name} samples: {len(g2_data)}")
    print(f"\t[MonteCarlo] {n_samples = }")
    print(f"\t[MonteCarlo] p({g1_name} > {g2_name}) = {probability1}")
    print(f"\t[MonteCarlo] p({g1_name} < {g2_name}) = {1.0-probability1}")

"""
    Does the same as computeMonteCarloSwitchedProbability function, but in this case
    instead of MonteCarlo, the data is resampled and then values are extracted,
    as in a Bootstrap approach, based on the original data instead of the approximated 
    Normal distribution.
"""
def computeBootstrapSwitchedProbability(g1_data, g2_data, g1_name = "group1", g2_name = "group2", n_samples = bootstrap_samples):
    
    count1 = 0
    # Bootstrap resampling
    for _ in range(n_samples):
        sample1 = np.random.choice(g1_data, size=len(g1_data), replace=True)
        sample2 = np.random.choice(g2_data, size=len(g2_data), replace=True)
        
        valor1 = np.random.choice(sample1)
        valor2 = np.random.choice(sample2) 
        if valor1 > valor2: count1 += 1

    probability1 = count1 / n_samples

    # print(f"\t[Bootstrap] {g1_name} samples: {len(g1_data)}")
    # print(f"\t[Bootstrap] {g2_name} samples: {len(g2_data)}")
    print(f"\t[Bootstrap] {n_samples = }")
    print(f"\t[Bootstrap] p({g1_name} > {g2_name}) = {probability1}")
    print(f"\t[Bootstrap] p({g1_name} < {g2_name}) = {1.0-probability1}")


"""
    Given a set of data, computes the amount of repetitions a model should be trained to get
    a better result than the one a 5% top (of real data). Based on a MonteCarlo simulation
    based on the approximated distribution
"""
def computeMonteCarloBetterResultSampleSize(g1_data, g1_name, n_samples = montecarlo_samples):
    percentile = 98
    g1_mean = np.mean(g1_data)
    g1_std = np.std(g1_data)
    
    percentile_95 = np.percentile(g1_data, percentile)
    sample_size_list = []
    for _ in range(int(n_samples/100)):
        n = 0
        while True:
            observation = np.random.normal(g1_mean, g1_std, 1)
            if observation > percentile_95:
                sample_size_list.append(n)
                break
            n+=1

    sample_size = np.mean(sample_size_list)

    accumulated_prob = stats.norm.cdf(percentile_95, loc=g1_mean, scale=g1_std) 
    probability_in_normal_dist = 1 - accumulated_prob

    print(f"\t[] To get a value as {percentile_95} (percentile {percentile} of {g1_name} data), or higher, you would need ({probability_in_normal_dist*100} probability in aproximated normal distribution):")
    print(f"\t[MonteCarlo] Samples computed: {sample_size}")
    print(f"\t[ReglaDeTres] Samples computed: {accumulated_prob/probability_in_normal_dist}")


if __name__ == "__main__":
    metrics_data = getAllModelData()

    accuracy_data = {}
    print("Data available is:")
    for model, data in metrics_data.items():
        accuracy_data[model] = [entry['accuracy']*100 for entry in metrics_data[model].values()]
        print(f"\t· [{model}] samples: {len(accuracy_data[model])}")
    # print(f"Accuracy data filtered: {accuracy_data}")

    ### Probability
    print("Analysis for HiddenLayerPerceptron and DNN_6L models:")
    g1_data = accuracy_data['HiddenLayerPerceptron']
    g2_data = accuracy_data['DNN_6L']

    computeMonteCarloSwitchedProbability(g1_data, g2_data,'HiddenLayerPerceptron','DNN_6L')
    computeBootstrapSwitchedProbability(g1_data, g2_data,'HiddenLayerPerceptron','DNN_6L')

    print("Analysis of sample size for SimplePerceptron:")
    computeMonteCarloBetterResultSampleSize(accuracy_data['SimplePerceptron'],'SimplePerceptron')