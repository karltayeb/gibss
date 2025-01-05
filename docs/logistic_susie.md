# Logistic SuSiE

We have implemented two versions of logistic SuSiE. For the moment we recommend using the version in `gibss.logistic` module.
The most important design decision made in this module is to handle the intercept and other fixed covariates in their own component of the additive model.
This dramatically simplifies implementation of the SER, where each sub-problem corresponds to fitting a simple univariate logistic regression.
It is much easier to develop a fast, stable optimization scheme for this 1d problem.
This has the advantage that we can flexibly specify how to handle these effects seperate from how we do this SER.
For example, we could change the regularization/priors on the fixed effect parameters independently of the SER.

### The Logistic SER  

When you call `logistic.gibss.fit_logistic_susie` with `method='hermite'`, 
you fit an additive model where each component is an SER.

`gibss.logistic.logistic_ser_hermite` implements the logistic SER in two stages.

1. Compute the MAP estimates for each variable, given the current predictions from the other components.
1. Approximate the Bayes Factor/marginal likelihood for each variable using Gauss-Hermite quadrature. When using $m=1$ points, this corresponds to the usual Laplace approximation. 
1. Approximate the posterior mean for each effect using the same quadrature rule.
1. Return the expected prediction for each observation.

``` py 
def make_logistic_hermite_fitfun(X, y, kwargs=dict()):
    kwargs2 = dict(
        prior_variance = float(10.),
        newtonkwargs=dict(tol=1e-2, maxiter=5, alpha=0.2, gamma=-0.1)
    )
    kwargs2.update(kwargs)

    @jax.jit
    def fitfun(psi, old_fit):
        # update default arguments
        # fit SER
        if old_fit is None:
            coef_init = np.zeros((X.shape[0], 1))
        else:
            coef_init = old_fit.fits.state.x
        return logistic_ser_hermite(
            coef_init = coef_init,
            X = X, y=y, offset = psi,
            **kwargs2
        )
    return fitfun
```

#### Rationale for default hyperparameters

The GIBSS outer loop iteratively updates each SER.
Within each SER, we need to compute the MAP estimate for each variable.
Our approach here is to only run a small number of Newton steps (default 5) during each SER update.
For the first iteration we initialize at the null $b=0$. For subsequent iterations we initialize with the previous iterate.
Across several iterations of the outer loop, as `psi` stabilizes, the optimization problem in the inner loop remains unchanged.
Heuristically, we save computational effort by not optimizing very precisely the intermediate objectives that are liable to change drastically iteration to iteration,
and by leveraging the previous approximate optima when the problems are similar.

Note that for convex problems, Newton exhibits fast (quadratic) convergence within a neighborhood of the optima with stepsize $1$.
Away from the optimum, the Newton update is guarunteed to be a descent direction but the step size may need tuning.
We start with a stepsize and decay geometrically until the objective improves (or at least, does not increase too much).
We set the step size scaling factor to $0.2$, which gives a minimum stepsize of $0.2^5$.
In practice this minimum step size is small enough that we will improve the objective at each iteration.
