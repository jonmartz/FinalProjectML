import os
import pandas as pd
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman


metric = 'mean_squared_error'

# get average over cv folds
df_results = pd.read_csv('results.csv', usecols=['Dataset Name', 'Algorithm Name', metric])
algorithm_names = pd.unique(df_results['Algorithm Name'])
groups_by_algorithm = df_results.groupby('Algorithm Name')
average_results = {}
for algorithm_name in algorithm_names:
    df_algorithm = groups_by_algorithm.get_group(algorithm_name)
    average_results[algorithm_name] = df_algorithm.groupby('Dataset Name').mean()[metric]
df_results = pd.DataFrame(average_results)

# friedman and post hoc tests
t_stat, p_val = friedmanchisquare(*[df_results[i] for i in algorithm_names])
print('\nfriedman test p-val = %s' % p_val)
post_hoc_p_vals = posthoc_nemenyi_friedman(df_results.to_numpy())
post_hoc_p_vals.columns = algorithm_names
print('\npost hoc p-vals:\n%s' % post_hoc_p_vals)

