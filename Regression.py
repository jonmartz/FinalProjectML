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


random_search_iters, random_search_cv = 10, 3
models = [
    {'name': 'AdaBoost', 'model': AdaBoostRegressor(DecisionTreeRegressor(random_state=1), random_state=1),
     'rs_params': {'n_estimators': randint(5, 50), 'base_estimator__ccp_alpha': uniform(0, 0.01)}},
    {'name': 'RandomForest', 'model': RandomForestRegressor(random_state=1),
     'rs_params': {'n_estimators': randint(5, 50), 'ccp_alpha': uniform(0, 0.01)}},
]

datasets_in_log = {}
if not os.path.exists('results.csv'):
    with open('results.csv', 'w', newline='') as log_file:
        writer = csv.writer(log_file)
        header = ['Dataset Name', 'Algorithm Name', 'Cross Validation [1-10]', 'Hyper-Parameters Values',
                  'Mean Squared Error', 'Mean Absolute Error', 'Median Absolute Error', 'R2 Score',
                  'Explained Variance Score', 'Training Time', 'Inference Time']
        writer.writerow(header)
else:  # load already logged results to avoid redundancy and shorten runtime
    for dataset_name, log_dataset in pd.read_csv('results.csv').groupby('Dataset Name'):
        folds_in_dataset = {}
        datasets_in_log[dataset_name] = folds_in_dataset
        for fold, log_fold in log_dataset.groupby('Cross Validation [1-10]'):
            folds_in_dataset[fold] = set(log_fold['Algorithm Name'])

num_folds = 10
files = os.listdir(os.fsencode('datasets'))
avg_runtime, iteration, iterations = 0, 0, len(files) * len(models) * num_folds
for dataset_idx, file in enumerate(files):

    # load and pre-process dataset
    print('%d/%d dataset = %s' % (dataset_idx + 1, len(files), file))
    dataset_name = os.fsdecode(file)[:-4]
    dataset = pd.read_csv('datasets/%s.csv' % dataset_name)
    dataset = pd.get_dummies(dataset)  # one-hot encode categorical features
    array = dataset.to_numpy()
    X, y = array[:, :-1], array[:, -1]

    # start k-fold cross validation
    folds = KFold(n_splits=10, shuffle=True, random_state=1).split(dataset)
    for fold_idx, fold in enumerate(list(folds)):
        iteration += 1
        start_time_fold = int(round(time() * 1000))

        # organize samples for this fold
        indexes_train, indexes_test = fold
        X_test, y_test = X[indexes_test], y[indexes_test]
        X_train, y_train = X[indexes_train], y[indexes_train]

        for model in models:

            # fit model
            best_model = RandomizedSearchCV(model['model'], model['rs_params'], random_state=1,
                                            n_iter=random_search_iters, cv=random_search_cv)
            start_time_train = int(round(time() * 1000))
            best_model.fit(X_train, y_train)
            runtime_train = (round(time() * 1000) - start_time_train) / 1000
            best_parameters = best_model.best_params_

            # test model
            start_time_test = int(round(time() * 1000))
            y_pred = best_model.predict(X_test)
            runtime_test = (round(time() * 1000) - start_time_test) / 1000

            # compute metrics
            mean_squared_error = metrics.mean_squared_error(y_test, y_pred)
            mean_absolute_error = metrics.mean_absolute_error(y_test, y_pred)
            median_absolute_error = metrics.median_absolute_error(y_test, y_pred)
            r2_score = metrics.r2_score(y_test, y_pred)
            explained_variance_score = metrics.explained_variance_score(y_test, y_pred)

            # save to log
            with open('results.csv', 'a', newline='') as log_file:
                writer = csv.writer(log_file)
                row = [dataset_name, model['name'], fold_idx + 1, best_parameters, mean_squared_error,
                       mean_absolute_error, median_absolute_error, r2_score, explained_variance_score,
                       runtime_train, runtime_test]
                writer.writerow(row)

            # print fold runtime and ETA
            runtime = (round(time() * 1000) - start_time_fold) / 1000
            avg_runtime = (avg_runtime * (iteration - 1) + runtime) / iteration
            runtime_string = get_time_string(runtime)
            eta = get_time_string((iterations - iteration) * avg_runtime)
            print('\tfold %d/%d time = %s ETA = %s' % (fold_idx + 1, len(folds), runtime_string, eta))
