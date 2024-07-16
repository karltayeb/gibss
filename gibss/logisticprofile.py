# Implement Logistic SER using JAX
# for each variable maximize the intercept, "profile likelihood" method
import jax.numpy as jnp
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

@partial(jax.tree_util.register_dataclass,
         data_fields=['logp', 'lbf', 'beta', 'state'], meta_fields=[])
@dataclass
class UnivariateRegression:
    logp: float
    lbf: float
    beta: float
    state: Any

@jax.jit
def nloglik_mle(coef, x, y, offset):
    """Logistic log-likelihood"""
    psi = offset + coef[0] + x * coef[1]
    ll = jnp.sum(y * psi - jnp.logaddexp(0, psi))
    return -ll

nloglik_mle_hess = jax.hessian(nloglik_mle)
nloglik_mle_vmap = jax.vmap(nloglik_mle, in_axes=(0, None, None, None))

@jax.jit
def nloglik(coef, x, y, offset, prior_variance):
    """Logistic log-likelihood"""
    psi = offset + coef[0] + x * coef[1]
    ll = jnp.sum(y * psi - jnp.logaddexp(0, psi)) + norm.logpdf(coef[1], 0, jnp.sqrt(prior_variance))
    return -ll

nloglik_hess = jax.hessian(nloglik)
nloglik_vmap = jax.vmap(nloglik, in_axes=(0, None, None, None, None))

def compute_wakefield_lbf(betahat, s2, prior_variance):
    lbf = norm.logpdf(betahat, loc=0, scale=jnp.sqrt(s2 + prior_variance)) \
        - norm.logpdf(betahat, loc=0, scale=jnp.sqrt(s2))
    return lbf

def wakefield(coef_init, x, y, offset, prior_variance, nullfit):
    solver = newton_factory(Partial(nloglik_mle, x=x, y=y, offset=offset), niter=5)
    state = solver(coef_init)
    params = state.x
    hessian = -state.h

    # approximate BF with wakefield
    # see appendix of Wakefield 2009 for justicatioin of why there is no dependence on the intercept
    s2 = -1/hessian[1,1]
    lbf = compute_wakefield_lbf(params[1], s2, prior_variance)

    # shrink  the effect size
    # this gives an asymptotic approximation to the MAP estimate from the MLE
    # by combining the quadratic approximation of the likelihood at the MLE 
    # with a normal prior
    beta = params[1] * prior_variance / (s2 + prior_variance)
    # NOTE: the value for logp might not make sense
    return UnivariateRegression(lbf + nullfit.logp, lbf, beta, state)

def estimate_prior_variance_wakefield(fits, prior_variance_init):
    """Estimate prior variance using Wakefield approximation"""
    def f(ln_prior_variance):
        return -logsumexp(compute_wakefield_lbf(
            fits.state.x[:, 1],
            1/fits.state.h[:, 1,1], 
            jnp.exp(ln_prior_variance)))
    fopt = gd_factory(f, maxiter=100, init_ss=1.)
    return jnp.exp(fopt(jnp.atleast_1d(jnp.log(prior_variance_init))).x)[0]

@partial(jax.vmap, in_axes=(0, None))
def update_prior_variance_wakefield(fit: UnivariateRegression, prior_variance: float) -> UnivariateRegression:
    # extract state
    betahat = fit.state.x[1]
    s2 = 1/fit.state.h[1, 1]
    ll0 = fit.logp - fit.lbf

    # compute wakefield lbf
    lbf = compute_wakefield_lbf(betahat, s2, prior_variance)
    beta = betahat * prior_variance / (s2 + prior_variance)
    return UnivariateRegression(lbf + ll0, lbf, beta, fit.state)

# def compute_lapmle_logp(ll, betahat, tau1, tau):
#     logp = ll + \
#         0.5 * jnp.log(tau1/(tau1 + tau)) \
#         - 0.5 * tau*tau1/(tau + tau1) * betahat**2
#     return logp

def compute_lapmle_logp(ll, betahat, s2, prior_variance):
    logp = ll + \
        0.5 * jnp.log(2 * jnp.pi * s2) + \
        norm.logpdf(betahat, loc=0, scale=jnp.sqrt(s2 + prior_variance))
    return logp

def laplace_mle(coef_init, x, y, offset, prior_variance, nullfit):
    solver = newton_factory(Partial(nloglik_mle, x=x, y=y, offset=offset), niter=5)
    state = solver(coef_init)
    params = state.x
    hessian = -state.h

    # compute wakefield lbf
    s2 = -1/hessian[1,1]
    ll = - state.f
    betahat = params[1]
    logp = compute_lapmle_logp(ll, betahat, s2, prior_variance)
    lbf = logp - nullfit.logp
    beta = betahat * prior_variance / (s2 + prior_variance)
    return UnivariateRegression(logp, lbf, beta, state)

@partial(jax.vmap, in_axes=(0, None))
def update_prior_variance_lapmle(fit: UnivariateRegression, prior_variance: float) -> UnivariateRegression:
    # extract state
    betahat = fit.state.x[1]
    s2 = 1/fit.state.h[1, 1]
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

# gauss-hermite quadrature nodes and weights
def hermite_factory(m):
    base_nodes, base_weights = np.polynomial.hermite.hermgauss(m)
    def hermite(coef_init, x, y, offset, prior_variance, nullfit):
        solver = newton_factory(Partial(nloglik, x=x, y=y, offset=offset, prior_variance=prior_variance), niter=5)
        state = solver(coef_init)
        params = state.x
        hessian = -state.h

        # set up quadrature
        mu = params[1]
        sigma = jnp.sqrt(-1/hessian[1,1])
        nodes = base_nodes * jnp.sqrt(2) * sigma + mu
        weights = base_weights / jnp.sqrt(jnp.pi)

        # compute logp
        coef_nodes = jnp.stack([jnp.ones_like(nodes) * params[0], nodes], axis=1)
        y = -nloglik_vmap(coef_nodes, x, y, offset, prior_variance)
        logp = jax.scipy.special.logsumexp(y - norm.logpdf(nodes, loc=mu, scale=sigma) + jnp.log(weights)) 
        lbf = logp - nullfit.logp

        # compute posterior mean of effect
        y2 = nodes * jnp.exp(y - logp)
        beta = jnp.sum(y2/norm.pdf(nodes, loc=mu, scale=sigma) * weights)
        return UnivariateRegression(logp, lbf, beta, state)
    
    hermite_jit = jax.jit(hermite)
    return hermite_jit

@jax.jit
def fit_null(y, offset):
    """Logistic SER"""
    # we fit null model by giving a covariate with no variance and setting prior variance to ~=0
    # so that we can just reuse the code for fitting the full model
    x = jnp.zeros_like(y)
    prior_variance = 1e-10
    solver = newton_factory(Partial(nloglik, x=x, y=y, offset=offset, prior_variance=prior_variance), niter=20)
    state = solver(jnp.zeros(2))
    params = state.x    
    ll0 = -nloglik_mle(params, x, y, offset)
    b0 = params[0]
    return UnivariateRegression(ll0, 0, b0, state)


# use Partial so that you can pass to jitted ser function
# see https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.Partial.html
vwakefield= Partial(jax.vmap(wakefield, in_axes=(0, 0, None, None, None, None)))
vlapmle = Partial(jax.vmap(laplace_mle, in_axes=(0, 0, None, None, None, None)))
pfitnull = Partial(fit_null)

@jax.jit
def logistic_ser_wakefield(coef_init, X, y, offset, prior_variance):
    return ser(coef_init, X, y, offset, prior_variance, vwakefield, pfitnull)

@jax.jit
def logistic_ser_wakefield_eb(coef_init, X, y, offset, prior_variance):
    # 1. fit ser, choice of prior variance doesn't matter
    nullfit = pfitnull(y, offset)
    fits = vwakefield(coef_init, X, y, offset, prior_variance, nullfit)
    prior_variance = estimate_prior_variance_wakefield(fits, prior_variance)
    fits2 = update_prior_variance_wakefield(fits, prior_variance)
    return _ser(fits2, prior_variance, X)

@jax.jit
def logistic_ser_lapmle(coef_init, X, y, offset, prior_variance):
    return ser(coef_init, X, y, offset, prior_variance, vlapmle, pfitnull)

@jax.jit
def logistic_ser_lapmle_eb(coef_init, X, y, offset, prior_variance):
    # 1. fit ser, choice of prior variance doesn't matter
    nullfit = pfitnull(y, offset)
    fits = vlapmle(coef_init, X, y, offset, prior_variance, nullfit)
    prior_variance = estimate_prior_variance_lapmle(fits, prior_variance)
    fits2 = update_prior_variance_lapmle(fits, prior_variance)
    return _ser(fits2, prior_variance, X)


@partial(jax.jit, static_argnames = ['m'])
def logistic_ser_hermite(coef_init, X, y, offset, prior_variance, m):
    vhermite = Partial(jax.vmap(hermite_factory(m), in_axes=(0, 0, None, None, None, None)))
    return ser(coef_init, X, y, offset, prior_variance, vhermite, pfitnull)


def initialize_coef(X, y, offset, prior_variance):
    """Initialize univarate regression coefficients using null model"""
    nullfit = fit_null(y, offset)
    coef_init = np.zeros((X.shape[0], 2))
    coef_init[:, 0] = nullfit.beta
    return coef_init


def logistic_susie(X, y, L=5, prior_variance=1, maxiter=10, tol=1e-3, method='hermite', m=5):
    if method == 'hermite':
        serfun = partial(logistic_ser_hermite, m=m)
    elif method == 'wakefield':
        serfun = logistic_ser_wakefield
    elif method == 'wakefield_eb':
        serfun = logistic_ser_wakefield_eb
    elif method == 'lapmle':
        serfun = logistic_ser_lapmle
    elif method == 'lapmle_eb':
        serfun = logistic_ser_lapmle_eb
    else:
        raise ValueError(f"Unknown method {method}: must be one of 'hermite', 'wakefield', or 'lapmle'")
    return gibss(X, y, L, prior_variance, maxiter, tol, initialize_coef, serfun)