# Implement Logistic SER using JAX
# MLE based SERs-- either Wakefield or Laplace-MLE
import jax.numpy as jnp
from jax.scipy.stats import norm
import jaxopt
import jax
import numpy as np
from functools import partial, cache
from gibss.ser import SER
from gibss.additive import additive_model

def loglik(coef, x, y, offset):
    """Logistic log-likelihood"""
    psi = offset + coef[0] + x * coef[1]
    return jnp.sum(y * psi - jnp.logaddexp(0, psi))

def negative_loglik(coef, x, y, offset):
    return -loglik(coef, x, y, offset)

loglik_hess = jax.hessian(loglik)
loglik_vmap = jax.vmap(loglik, in_axes=(0, None, None, None))

@jax.jit
def logistic_regression_mle(x, y, offset, coef_init):
    solver = jaxopt.LBFGS(fun=negative_loglik, maxiter=100)
    params, state = solver.run(coef_init, x=x, y=y, offset=offset)
    return params

hermgauss = cache(np.polynomial.hermite.hermgauss)

@partial(jax.jit, static_argnames=['correct'])
@partial(jax.vmap, in_axes=(0, 0, None, None, None, None))
def logistic_regression(coef_init, x, y, offset, prior_variance, correct=False):
    """Univariate logistic regression"""
    # fit MAP
    coef_mle = logistic_regression_mle(x, y, offset, coef_init)

    # compute Hessian
    hessian = loglik_hess(coef_mle, x, y, offset)

    # approximate BF with wakefield
    # see appendix of Wakefield 2009 for justicatioin of why there is no dependence on the intercept
    s2 = -1/hessian[1,1]
    lbf = norm.logpdf(coef_mle[1], loc=0, scale=jnp.sqrt(s2 + prior_variance)) \
        - norm.logpdf(coef_mle[1], loc=0, scale=jnp.sqrt(prior_variance))

    # correct lbf: missing - logp(beta = 0) term
    if correct:
        lbf = lbf + norm.logpdf(coef_mle[1], 0, jnp.sqrt(s2)) \
            - norm.logpdf(0, 0, jnp.sqrt(s2)) \
            + loglik(coef_mle, x, y, offset)
    
    # shrink  the effect size
    beta = coef_mle[1] * prior_variance / (s2 + prior_variance)
    return beta, lbf, coef_mle

@jax.jit
def logistic_ser_mle(coef_init, X, y, offset, prior_variance):
    """Logistic SER"""
    # fit logistic regression for each column of X
    beta, lbf, params = logistic_regression(coef_init, X, y, offset, prior_variance, False)

    # compute PIPs
    alpha = jnp.exp(lbf - jax.scipy.special.logsumexp(lbf))
    
    # compute predictions
    # note we are not including the intercept!
    psi = X.T @ (alpha * beta)
    
    # return coef
    return SER(psi, alpha, params)


def logistic_susie_mle(X, y, L=5, prior_variance=1, maxiter=10):
    # attatch data, initialization, settings to the fit function once
    def fit_function(psi):
        # TODO: better initialization
        return logistic_ser_mle(np.zeros((X.shape[0], 2)), X, y, psi, prior_variance)
    
    fit_funs = [fit_function for _ in range(L)]
    psi_init = np.zeros_like(y)
    res = additive_model(psi_init, fit_funs, maxiter=maxiter)
    return res