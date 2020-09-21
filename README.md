# FinalProjectML 

What's in this project? (Based on a ML course by Pr. Lior Rokach, BGU)

This project includes the execution and the statistical comparison of these four algorithms:

COBRA - which proposes a new method for combining several initial estimators of the regression function. Instead of building a linear or convex optimized combination over a collection of basic estimators, it uses them as a collective indicator of the proximity between the training data and a test observation. Based on this paper: Biau, et al. "COBRA: A combined regression strategy." Journal of Multivariate Analysis 146 (2016): 18-28.

EWA - which obtains sharp oracle inequalities for convex aggregates via exponential weights, under general assumptions on the distribution of errors and on the functions to aggregate. In other words, it obtains a weighted average of the predictions of all the algorithms that form the ensemble. Based on this paper: Dalalyan, Arnak S., and Alexandre B. Tsybakov. "Aggregation by exponential weighting and sharp oracle inequalities." International Conference on Computational Learning Theory. Springer, Berlin, Heidelberg, 2007.

Boruta - which is built around a base algorithm (e.g: the random forest algorithm) with additional implementation for finding all relevant features and removing non-relevant ones. Boruta core idea is that a feature that is not relevant is not more useful for classification than its version with a permuted order of values. Based on this paper: Miron B. Kursa, Witold R. Rudnicki: “Feature Selection with the Boruta Package". Feature Selection with the Boruta Package. September 2010, Volume 36, Issue 11.

Adaboost - The well-known adaptive boosting meta-algorithm (formulated by Yoav Freund and Robert Schapire).

The statistical data analysis includes performing the Friedman test on the MSE metric results of the four algorithms and additional meta-learning model implementation using XGBoost that performs the binary classification task of determining whether an algorithm will be the best performing one (rank 1) given a predefined dataset’s meta-features.

Installation

pip install pycobra
pip install Boruta

Dependencies

Python 3.4+
numpy, scipy, scikit-learn, matplotlib, pandas, seaborn.

Execution:



Hyper-Parameters:

n_estimators - sets the number of estimators in the chosen ensemble method.
estimator - A supervised learning estimator, with a 'fit' method that returns the feature_importances_ attribute. Important features must correspond to high absolute values in the feature_importances_.
epsilon - for determining the "distance" between the initial estimators and the new estimator (used for COBRA).
beta - the "temperature" parameter, which is used to build the estimator fn based on data. (fur further explanation, look at EWA reference above).
machine_list - determines which list of initial estimators will be used for building the new estimator (used for COBRA & EWA).


Reference:

Biau, Fischer, Guedj and Malley (2016), COBRA: A combined regression strategy. Journal of Multivariate Analysis.
Dalalyan, Arnak S., and Alexandre B. Tsybakov. "Aggregation by exponential weighting and sharp oracle inequalities." International Conference on Computational Learning Theory. Springer, Berlin, Heidelberg, 2007.
Miron B. Kursa, Witold R. Rudnicki: “Feature Selection with the Boruta Package". Feature Selection with the Boruta Package. September 2010, Volume 36, Issue 11.
Guedj and Srinivasa Desikan (2020), Kernel-based ensemble learning in Python. Information.
Guedj and Srinivasa Desikan (2018), Pycobra: A Python Toolbox for Ensemble Learning and Visualisation. Journal of Machine Learning Research.




