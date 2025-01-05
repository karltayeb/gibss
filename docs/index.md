# Generalized iterative Bayesian stepwises selection

This is the documentation for the `gibss` python package, which implements generalized iterative Bayesian stepwise selection (GIBSS).
GIBSS provides a recipe for building a sparse additive model when given a large number of predictors.

GIBSS was motivated by the iterative Bayesian stepwise selection (IBSS) procedure used to fit the sum of single effects regression (SuSiE). 
GIBSS corresponds to IBSS when the base model is a univariate regression with a Gaussian prior on the effect.
In this setting IBSS is coordinate ascent of a variational objective in a particular variational family.
Although GIBSS was inspired by IBSS, GIBSS does not (to our knowledge) correspond to optimization of a particular objective. 
The algorithm is heuristic. 

