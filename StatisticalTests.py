import os
import pandas as pd
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman


metric = 'mean_squared_error'

# get average over cv folds
df_results = pd.read_csv('results/results.csv', usecols=['Dataset Name', 'Algorithm Name', metric])
dataset_names = pd.unique(df_results['Dataset Name'])
model_names = pd.unique(df_results['Algorithm Name'])
average_results = {'dataset': dataset_names}
groups_by_model = df_results.groupby('Algorithm Name')
for model_name in model_names:
    df_model = groups_by_model.get_group(model_name)
    groups_by_dataset = df_model.groupby('Dataset Name')
    model_mean = []
    for dataset_name in dataset_names:
        model_mean.append(groups_by_dataset.get_group(dataset_name)[metric].mean())
    average_results[model_name] = model_mean
df_results = pd.DataFrame(average_results)
df_results.to_csv('results/average_results.csv', index=False)

# friedman and post hoc tests
t_stat, p_val = friedmanchisquare(*[df_results[i] for i in model_names])
print('\nfriedman test p-val = %s' % p_val)
post_hoc_p_vals = posthoc_nemenyi_friedman(df_results.drop(columns='dataset').to_numpy())
post_hoc_p_vals.columns = model_names
print('\npost hoc p-vals:\n%s' % post_hoc_p_vals)

