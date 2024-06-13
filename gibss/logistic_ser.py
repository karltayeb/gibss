# Implement Logistic SER using JAX
import jax.numpy as jnp
from jax.scipy.stats import norm
import jaxopt
import jax
import numpy as np
from functools import partial, cache
from gibss.ser import SER
from gibss.additive import additive_model

def loglik(coef, x, y, offset, prior_variance):
    """Logistic log-likelihood"""
    psi = offset + x * coef
    return jnp.sum(y * psi - jnp.logaddexp(0, psi)) + norm.logpdf(coef, 0, jnp.sqrt(prior_variance))

def negative_loglik(coef, x, y, offset, prior_variance):
    return -loglik(coef, x, y, offset, prior_variance)

loglik_hess = jax.hessian(loglik)
loglik_vmap = jax.vmap(loglik, in_axes=(0, None, None, None, None))

@jax.jit
def logistic_regression_map(x, y, offset, prior_variance, coef_init):
    solver = jaxopt.LBFGS(fun=negative_loglik, maxiter=100)
    params, state = solver.run(coef_init, x=x, y=y, offset=offset, prior_variance=prior_variance)
    return params

hermgauss = cache(np.polynomial.hermite.hermgauss)


@jax.jit
@partial(jax.vmap, in_axes=(0, None, None, None, 0))
def logistic_regression(x, y, offset, prior_variance, coef_init):
    """Univariate logistic regression"""
    # fit MAP
    coef_map = logistic_regression_map(x, y, offset, prior_variance, coef_init)

    # compute Hessian
    hessian = loglik_hess(coef_map, x, y, offset, prior_variance)

    # set up quadrature
    m = 5 # number of quadrature points
    mu = coef_map
    sigma = jnp.sqrt(-1/hessian)
    nodes, weights = hermgauss(m)
    nodes = nodes * jnp.sqrt(2) * sigma + mu
    weights = weights / jnp.sqrt(jnp.pi)

    # compute logp
    y = loglik_vmap(nodes, x, y, offset, prior_variance)
    logp = jax.scipy.special.logsumexp(y - norm.logpdf(nodes, loc=mu, scale=sigma) + jnp.log(weights)) 

    # compute posterior mean of effect
    y2 = nodes * jnp.exp(y - logp)
    beta = jnp.sum(y2/norm.pdf(nodes, loc=mu, scale=sigma) * weights)    

    return beta, logp, coef_map

@jax.jit
def logistic_ser(X, y, offset, prior_variance, coef_init):
    """Logistic SER"""
    # fit logistic regression for each column of X
    coef, logp, coef_map = logistic_regression(X, y, offset, prior_variance, coef_init)
    alpha = jnp.exp(logp - jax.scipy.special.logsumexp(logp))

    # compute predictions
    # note we are not including the intercept!
    psi = X.T @ (alpha * coef)
    params = dict(coef = coef, coef_map = coef_map, logp=logp)

    # return coef
    return SER(psi, alpha, params)

@jax.jit
def logistic_regression_intercept(X, y, offset, prior_variance, coef_init):
    intercept = logistic_regression_map(1., y, offset, 1e10, 0.)
    # TODO: odd to put this in an SER dataclass, just need to be able to access via `res.psi` in additive_model
    return SER(jnp.ones_like(y) * intercept, None, None)

def logistic_susie(X, y, L=5, prior_variance=1, maxiter=10):
    fit_funs = [logistic_regression_intercept] + [logistic_ser for _ in range(L)]
    res = additive_model(X, y, fit_funs, maxiter=10)

def logistic_susie(X, y, L=5, prior_variance=1, maxiter=10):
    # attatch data, initialization, settings to the fit function once
    def intercept_function(psi):
        return logistic_regression_intercept(X, y, psi, 1e10, 0.)
    
    def fit_function(psi):
        # TODO: better initialization
        return logistic_ser(X, y, psi, prior_variance, np.zeros(X.shape[0]))
    
    fit_funs = [intercept_function] + [fit_function for _ in range(L)]
    res = additive_model(np.zeros_like(y), fit_funs, maxiter=maxiter)
    return res