# FinalProjectML

What's in this project?

This project includes the execution and the statistical comparison of these four algorithms:

COBRA - which proposes a new method for combining several initial estimators of the regression function. Instead of building a linear or convex optimized combination over a collection of basic estimators, it uses them as a collective indicator of the proximity between the training data and a test observation. Based on this paper: Biau, et al. "COBRA: A combined regression strategy." Journal of Multivariate Analysis 146 (2016): 18-28.

EWA - which obtains sharp oracle inequalities for convex aggregates via exponential weights, under general assumptions on the distribution of errors and on the functions to aggregate. In other words, it obtains a weighted average of the predictions of all the algorithms that form the ensemble. Based on this paper: Dalalyan, Arnak S., and Alexandre B. Tsybakov. "Aggregation by exponential weighting and sharp oracle inequalities." International Conference on Computational Learning Theory. Springer, Berlin, Heidelberg, 2007.

Boruta - which is built around a base algorithm (e.g: the random forest algorithm) with additional algorithm that includes an implementation for finding all relevant features and removing non-relevant ones. Boruta core idea is that a feature that is not relevant is not more useful for classification than its version with a permuted order of values. Based on this paper: Miron B. Kursa, Witold R. Rudnicki: “Feature Selection with the Boruta Package". Feature Selection with the Boruta Package. September 2010, Volume 36, Issue 11.

Adaboost - The well-known adaptive boosting meta-algorithm (formulated by Yoav Freund and Robert Schapire).

