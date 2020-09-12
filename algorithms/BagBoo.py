import random
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator, RegressorMixin


class BagBoo(BaseEstimator, RegressorMixin):
    def __init__(self, n_boo=10, n_bag=10, bagging_ratio=0.1, rsm_ratio=1.0, ccp_alpha=0.01,
                 learning_rate=0.1, random_state=None):
        self.n_boo = n_boo
        self.n_bag = n_bag
        self.bagging_ratio = bagging_ratio
        self.rsm_ratio = rsm_ratio
        self.ccp_alpha = ccp_alpha
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, Y):
        self.boosting_list = []
        rsm_cnt = int(self.rsm_ratio * X.shape[1])
        bagging_cnt = int(self.bagging_ratio * X.shape[0])
        if bagging_cnt == 0:
            bagging_cnt += 1
        for bag_iter in range(self.n_bag):
            idx_bagging = range(X.shape[0])
            idx_rsm = range(X.shape[1])
            new_idx_bagging = list(idx_bagging)[:]
            if self.random_state is not None:
                random.seed(self.random_state)
            random.shuffle(new_idx_bagging)
            new_idx_rsm = list(idx_rsm)[:]
            if self.random_state is not None:
                random.seed(self.random_state)
            random.shuffle(new_idx_rsm)
            new_idx_bagging = new_idx_bagging[:bagging_cnt]
            new_idx_rsm = new_idx_rsm[:(rsm_cnt)]
            X_bag = X[new_idx_bagging][:, new_idx_rsm]
            Y_bag = Y[new_idx_bagging]
            new_boosting = GradientBoostingRegressor(n_estimators=self.n_boo, ccp_alpha=self.ccp_alpha,
                                                     learning_rate=self.learning_rate, random_state=self.random_state)
            new_boosting.fit(X_bag, Y_bag)
            self.boosting_list.append(new_boosting)
        return self

    def predict(self, X):
        y = np.array([0.0] * X.shape[0])
        for boosting in self.boosting_list:
            y += boosting.predict(X)
        return y / float(self.n_bag)
