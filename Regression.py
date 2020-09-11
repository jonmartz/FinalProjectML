import os
import csv
import pandas as pd
from time import time
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn import metrics


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


num_folds = 2
random_search_iters = 2
random_search_cv = 2
models = [
    # {'name': 'AdaBoost', 'model': AdaBoostRegressor(DecisionTreeRegressor(random_state=1), random_state=1),
    #  'rs_params': {'n_estimators': randint(5, 50), 'base_estimator__ccp_alpha': uniform(0, 0.01)}},
    # {'name': 'RandomForest', 'model': RandomForestRegressor(random_state=1),
    #  'rs_params': {'n_estimators': randint(5, 50), 'ccp_alpha': uniform(0, 0.01)}},
    {'name': 'RegressionTree', 'model': DecisionTreeRegressor(random_state=1),
     'rs_params': {'ccp_alpha': uniform(0, 0.01)}},
]

datasets_in_log = {}
if not os.path.exists('results/results.csv'):
    with open('results/results.csv', 'w', newline='') as log_file:
        writer = csv.writer(log_file)
        header = ['Dataset Name', 'Algorithm Name', 'Cross Validation [1-10]', 'Hyper-Parameters Values',
                  'mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'r2_score',
                  'explained_variance_score', 'Training Time', 'Inference Time']
        writer.writerow(header)
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

                # fit model
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
                print('%d/%d dataset %d/%d (%s) fold %d/%d model %d/%d (%s) time: %s ETA: %s' %
                      (iteration, iterations, dataset_idx + 1, len(files), dataset_name, fold_idx + 1, len(folds),
                       model_idx + 1, len(models), model['name'], get_time_string(runtime), eta))
