# Implement Logistic SER using JAX
# for each variable maximize the intercept, "profile likelihood" method
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as mvnorm
from jax.scipy.stats import norm
from jax.scipy.special import logsumexp
import jax
import numpy as np
from functools import partial
from gibss.ser import ser, _ser
from jax.tree_util import Partial
from typing import Any
from dataclasses import dataclass
from gibss.newton import newton_factory
from gibss.gd_backtracking import gd_factory
from gibss.gibss import gibss

from gibss.logisticprofile import UnivariateRegression, nloglik_mle

@jax.jit
def nloglik(coef, x, y, offset, prior_variance):
    """Logistic log-likelihood"""
    psi = offset + coef[0] + x * coef[1]
    ll = jnp.sum(y * psi - jnp.logaddexp(0, psi)) + norm.logpdf(coef, np.zeros(2), jnp.sqrt(prior_variance)).sum()
    return -ll

nloglik_hess = jax.hessian(nloglik)
nloglik_vmap = jax.vmap(nloglik, in_axes=(0, None, None, None, None))

@partial(jax.vmap, in_axes=(0, None))
def update_prior_variance_lapmle(fit: UnivariateRegression, prior_variance: float) -> UnivariateRegression:
    # extract state
    betahat = fit.state.x[1]
    # s2 = 1/fit.state.h[1, 1]
    hessian = - fit.state.h
    s2 = - hessian[0 ,0] / (hessian[0,0] * hessian[1, 1] - hessian[1,0] * hessian[0,1])
    ll = -fit.state.f

    # compute wakefield lbf
    logp = compute_lapmle_logp(ll, betahat, s2, prior_variance)
    lbf = fit.lbf - fit.logp + logp
    beta = betahat * prior_variance / (s2 + prior_variance)
    return UnivariateRegression(logp, lbf, beta, fit.state)
    

def estimate_prior_variance_lapmle(fits: UnivariateRegression, prior_variance_init: float) -> float:
    """Estimate prior variance using Wakefield approximation"""
    def f(ln_prior_variance):
        return -logsumexp(compute_lapmle_logp(
            -fits.state.f, fits.state.x[:, 1], 1/fits.state.h[:, 1,1], jnp.exp(ln_prior_variance))
        )
    fopt = gd_factory(f, maxiter=100, init_ss=1.)
    return jnp.exp(fopt(jnp.atleast_1d(jnp.log(prior_variance_init))).x)[0]

@jax.jit
def nloglik0(b0, y, offset, prior_variance0):
    """Logistic log-likelihood"""
    psi = offset + b0
    ll = jnp.sum(y * psi - jnp.logaddexp(0, psi)) + norm.logpdf(b0, 0., jnp.sqrt(prior_variance0)).sum()
    return -ll

@jax.jit
def fit_null(y, offset, prior_variance0= 10., maxiter=50, tol=1e-3, alpha=0.5, gamma=0.0):
    """Logistic SER"""
    # we fit null model by giving a covariate with no variance and setting prior variance to ~=0
    # so that we can just reuse the code for fitting the full model
    solver = newton_factory(Partial(nloglik0, y=y, offset=offset, prior_variance0=prior_variance0), maxiter=maxiter, tol=tol, alpha=alpha, gamma=gamma)
    state = solver(jnp.zeros(1))
    params = state.x    
    b0 = params[0]
    s20 = 1/state.h[0, 0]
    logp0 = -state.f + 0.5 * jnp.log(2 * jnp.pi * s20) 
    return UnivariateRegression(logp0, 0, b0, state)

def laplace_map(coef_init, x, y, offset, nullfit, prior_variance, maxiter=50, tol=1e-3, alpha=0.5, gamma=0.0):
    solver = newton_factory(Partial(nloglik, x=x, y=y, offset=offset, prior_variance=prior_variance), maxiter=maxiter, tol=tol, alpha=alpha, gamma=gamma)
    state = solver(coef_init)
    logp = -0.5 * jnp.linalg.slogdet(state.h / (2 * jnp.pi)).logabsdet - state.f
    lbf = logp - nullfit.logp
    betahat = state.x[1]
    return UnivariateRegression(logp, lbf, betahat, state)

@jax.jit
def logistic_ser_lapmap(coef_init, X, y, offset, prior_variance, maxiter=50, tol=1e-3, alpha=0.5, gamma=0.0):
    vlapmle = jax.vmap(Partial(laplace_map, prior_variance=prior_variance, maxiter=maxiter, tol=tol, alpha=alpha, gamma=gamma), in_axes=(0, 0, None, None, None))
    fit_null_partial = Partial(fit_null, prior_variance0=prior_variance[0])
    return ser(coef_init, X, y, offset, vlapmle, fit_null_partial)

@jax.jit
def logistic_ser_lapmle_eb(coef_init, X, y, offset, prior_variance, maxiter=50, tol=1e-3, alpha=0.5, gamma=0.0):
    # 1. fit ser, choice of prior variance doesn't matter
    nullfit = fit_null(y, offset)
    vlapmle = jax.vmap(Partial(laplace_mle, maxiter=maxiter, tol=tol, alpha=alpha, gamma=gamma), in_axes=(0, 0, None, None, None, None))
    fits = vlapmle(coef_init, X, y, offset, prior_variance, nullfit)
    prior_variance = estimate_prior_variance_lapmle(fits, prior_variance)
    fits2 = update_prior_variance_lapmle(fits, prior_variance)
    return _ser(fits2, X)

def initialize_coef(X, y, offset, prior_variance):
    """Initialize univarate regression coefficients using null model"""
    return np.zeros((X.shape[0], 2)) 

def logistic_susie(X, y, L=5, prior_variance=1, maxiter=10, tol=1e-3, method='hermite', serkwargs: dict = {}):
    if method == 'lapmle':
        serfun = partial(logistic_ser_lapmle, **serkwargs)
    elif method == 'lapmle_eb':
        serfun = partial(logistic_ser_lapmle_eb, **serkwargs)
    else:
        raise ValueError(f"Unknown method {method}: must be one of 'hermite', 'wakefield', or 'lapmle'")
    return gibss(X, y, L, prior_variance, maxiter, tol, initialize_coef, serfun)