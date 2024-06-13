# Generalized IBSS

This package implements "Generalized IBSS"


## TODO

- [ ] Monitor convergence/run to convergence in Newton's method rather than running for a fixed number of iterations
- [ ] Use JAX looping instead of python loop to speed up compilation. Are there other changes that would improve compilation time?
- [ ] Implement cox model?
- [ ] Keep track of optimization state in univariate regression?
- [ ] Implement more additive components e.g. for fixed effects/covariates to be included
- [ ] Extend GIBSS to account for extra covariates `Z` so that we don't need to specify a custom additive model to include these
- [ ] Convert informal tests from `notebooks/` into unit tests that we can use as the code base develops.
- [ ] Remove dependence on flax and jaxopt-- it would be preferable for this to only depend on base jax!    

Could be useful for diagonising issues