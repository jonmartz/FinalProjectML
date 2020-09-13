import os
import csv
import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from pycobra.cobra import Cobra
from pycobra.ewa import Ewa
from pycobra.diagnostics import Diagnostics


# from algorithms.BagBoo import BagBoo


def get_time_string(time_in_seconds):
    eta_string = '%.1f(secs)' % (time_in_seconds % 60)
    if time_in_seconds >= 60:
        time_in_seconds /= 60
        eta_string = '%d(mins) %s' % (time_in_seconds % 60, eta_string)
        if time_in_seconds >= 60:
            time_in_seconds /= 60
            eta_string = '%d(hours) %s' % (time_in_seconds % 24, eta_string)
            if time_in_seconds >= 24:
                time_in_seconds /= 24
                eta_string = '%d(days) %s' % (time_in_seconds, eta_string)
    return eta_string


num_folds = 10
random_search_iters = 10
random_search_cv = 3
models = [
    # {'name': 'AdaBoost', 'model': AdaBoostRegressor(DecisionTreeRegressor(random_state=1), random_state=1),
    #  'rs_params': {'n_estimators': randint(5, 100), 'base_estimator__ccp_alpha': uniform(0, 0.1)}},

    # {'name': 'RandomForest', 'model': RandomForestRegressor(random_state=1),
    #  'rs_params': {'n_estimators': randint(5, 50), 'ccp_alpha': uniform(0, 0.01)}},

    # {'name': 'RegressionTree', 'model': DecisionTreeRegressor(random_state=1),
    #  'rs_params': {'ccp_alpha': uniform(0, 0.01)}},

    {'name': 'Cobra', 'model': Cobra},

    {'name': 'Ewa', 'model': Ewa},

    # {'name': 'BagBoo', 'model': BagBoo(random_state=1),
    #  'rs_params': {'n_bag': randint(5, 100), 'ccp_alpha': uniform(0, 0.1)}},
]

datasets_in_log = {}
if not os.path.exists('results/results.csv'):
    with open('results/results.csv', 'w', newline='') as log_file:
        writer = csv.writer(log_file)
        header = ['Dataset Name', 'Algorithm Name', 'Cross Validation [1-10]', 'Hyper-Parameters Values',
                  'mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'r2_score',
                  'explained_variance_score', 'Training Time', 'Inference Time']
        writer.writerow(header)
    with open('results/progress_log.txt', 'w') as file:
        file.write('Progress:\n')
else:  # load already logged results to avoid redundancy and shorten runtime
    for dataset_name, log_dataset in pd.read_csv('results/results.csv').groupby('Dataset Name'):
        folds_in_dataset = {}
        datasets_in_log[dataset_name] = folds_in_dataset
        for fold, log_fold in log_dataset.groupby('Cross Validation [1-10]'):
            folds_in_dataset[fold] = set(log_fold['Algorithm Name'])

files = os.listdir(os.fsencode('datasets'))
avg_runtime, iteration, iterations = 0, 0, len(files) * len(models) * num_folds
for dataset_idx, file in enumerate(files):

    # load and pre-process dataset
    dataset_name = os.fsdecode(file)[:-4]
    dataset = pd.read_csv('datasets/%s.csv' % dataset_name)
    # dataset = dataset.fillna(method='ffill').fillna(method='bfill')  # fill nan values
    dataset = dataset.fillna(dataset.mean())  # fill nan values
    dataset = pd.get_dummies(dataset)  # one-hot encode categorical features
    array = dataset.to_numpy()
    # array = StandardScaler().fit_transform(array)
    X, y = array[:, :-1], array[:, -1]

    # start k-fold cross validation
    folds = list(KFold(n_splits=num_folds, shuffle=True, random_state=1).split(dataset))
    for fold_idx, fold in enumerate(folds):
        # organize samples for this fold
        indexes_train, indexes_test = fold
        X_test, y_test = X[indexes_test], y[indexes_test]
        X_train, y_train = X[indexes_train], y[indexes_train]

        for model_idx, model in enumerate(models):
            iteration += 1
            try:  # check if log already contains this iteration
                if model['name'] not in datasets_in_log[dataset_name][fold_idx + 1]:
                    raise KeyError()
            except KeyError:  # if not, run iteration
                start_time = int(time() * 1000)

                if model['name'] in ['Cobra', 'Ewa']:
                    # fit model
                    best_model = model['model'](random_state=1)
                    start_time_train = time()
                    np.random.seed(1)
                    # val_idx = np.random.choice(X_train.shape[0], int(X_train.shape[0] * 0.2), replace=False)
                    # X_val, y_val = X_train[val_idx, :], y_train[val_idx]
                    X_val, y_val = X_train, y_train
                    if model['name'] == 'Cobra':
                        best_model.set_epsilon(X_epsilon=X_val, y_epsilon=y_val, grid_points=random_search_iters)
                        best_parameters = {'epsilon': best_model.epsilon}
                    else:  # Ewa
                        best_model.set_beta(X_beta=X_val, y_beta=y_val)
                        best_parameters = {'beta': best_model.beta}
                    best_model.fit(X_train, y_train)
                    runtime_train = time() - start_time_train
                else:
                    best_model = RandomizedSearchCV(model['model'], model['rs_params'], random_state=1,
                                                    n_iter=random_search_iters, cv=random_search_cv)
                    start_time_train = time()
                    best_model.fit(X_train, y_train)
                    runtime_train = time() - start_time_train
                    best_parameters = best_model.best_params_

                # test model
                start_time_test = time()
                y_pred = best_model.predict(X_test)
                runtime_test = time() - start_time_test

                # compute metrics and prepare log entry
                row = [dataset_name, model['name'], fold_idx + 1, best_parameters,
                       metrics.mean_squared_error(y_test, y_pred),
                       metrics.mean_absolute_error(y_test, y_pred),
                       metrics.median_absolute_error(y_test, y_pred),
                       metrics.r2_score(y_test, y_pred),
                       metrics.explained_variance_score(y_test, y_pred),
                       runtime_train, runtime_test]

                # save entry to log
                with open('results/results.csv', 'a', newline='') as log_file:
                    writer = csv.writer(log_file)
                    writer.writerow(row)

                # print runtime and ETA
                runtime = (round(time() * 1000) - start_time) / 1000
                avg_runtime = (avg_runtime * (iteration - 1) + runtime) / iteration
                eta = get_time_string((iterations - iteration) * avg_runtime)
                progress_row = '%d/%d dataset %d/%d (%s) fold %d/%d model %d/%d (%s) time: %s ETA: %s' % (
                    iteration, iterations, dataset_idx + 1, len(files), dataset_name, fold_idx + 1, len(folds),
                    model_idx + 1, len(models), model['name'], get_time_string(runtime), eta)
                print(progress_row)
                with open('results/progress_log.txt', 'a') as file:
                    file.write('%s\n' % progress_row)
