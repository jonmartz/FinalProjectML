import pandas as pd
import xgboost as xgb


def create_meta_dataset():
    df_model_performance = pd.read_csv('average_results.csv')
    df_meta_features = pd.read_csv('meta_features.csv')
    df_meta_features = df_meta_features.fillna(df_meta_features.mean())
    model_names = list(df_model_performance.columns[1:])
    rows = []
    for row_idx in range(len(df_model_performance)):
        row_model_performance = df_model_performance.loc[row_idx]
        row_meta_features = df_meta_features.loc[row_idx]
        if row_model_performance['dataset'] != row_meta_features['dataset']:  # sanity check
            raise ValueError('dataset order is different on average_results.csv and meta_features.csv!')
        best_model_performance = row_model_performance[1:].min()  # set to min() because we measure MSE
        found_best = False
        for model_idx, model_name in enumerate(model_names):
            row_model_features = [0] * (len(model_names) + 1)
            row_model_features[model_idx] = 1  # set the one hot vector for model name
            if row_model_performance[model_name] == best_model_performance:
                row_model_features[-1] = 1  # set target class ("is best model") to 1
                found_best = True
            rows.append(list(row_meta_features) + row_model_features)
        if not found_best:  # sanity check
            raise ValueError('there was no best model in a dataset!')
    meta_dataset = pd.DataFrame(rows, columns=list(df_meta_features.columns) + model_names + ['is best'])
    meta_dataset.to_csv('meta_dataset.csv', index=False)


# create_meta_dataset()

# start the leave-one-out cross validation
