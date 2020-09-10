import os
import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.metrics import auc
from time import time
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_score


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
random_search_iters, random_search_cv = 5, 3
adaboost_param_distributions = {'n_estimators': randint(5, 50), 'base_estimator__ccp_alpha': uniform(0, 0.01)}

files = os.listdir(os.fsencode('datasets'))
avg_runtime, num_runtimes, total_runtimes = 0, 0, len(files) * num_folds
for dataset_idx, file in enumerate(files):
    print('%d/%d dataset = %s' % (dataset_idx + 1, len(files), file))
    dataset = pd.read_csv('datasets/%s' % os.fsdecode(file))
    dataset = pd.get_dummies(dataset)  # one-hot encode categorical features
    array = dataset.to_numpy()
    X, y = array[:, :-1], array[:, -1]
    k_fold = KFold(n_splits=10, shuffle=True, random_state=1)
    folds = list(k_fold.split(dataset))
    for fold_idx, fold in enumerate(folds):
        print('\tfold %d/%d' % (fold_idx + 1, len(folds)))

        # get samples for fold
        start_time = int(round(time() * 1000))
        indexes_train, indexes_test = fold
        X_test, y_test = X[indexes_test], y[indexes_test]
        X_train, y_train = X[indexes_train], y[indexes_train]

        # fit model
        adaBoost = AdaBoostClassifier(DecisionTreeClassifier(random_state=1), random_state=1)
        best_adaBoost = RandomizedSearchCV(adaBoost, adaboost_param_distributions, random_state=1,
                                           n_iter=random_search_iters, cv=random_search_cv)
        best_adaBoost.fit(X_train, y_train)
        print('\t\tbest params: %s' % best_adaBoost.best_params_)

        # compute metrics
        y_pred = adaBoost.predict_proba(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        true_positive_rate = tp / (tp + fn)
        false_positive_rate = fp / (fp + tn)
        auc = roc_auc_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        # calculate runtime and ETA
        runtime = (round(time() * 1000) - start_time) / 1000
        num_runtimes += 1
        avg_runtime = (avg_runtime * (num_runtimes - 1) + runtime) / num_runtimes
        runtime_string = get_time_string(runtime)
        eta = get_time_string((total_runtimes - num_runtimes) * avg_runtime)
        print('\t\taccuracy = %.4f time = %s ETA = %s' % (accuracy, runtime_string, eta))

