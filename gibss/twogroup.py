from jax.tree_util import Partial
from typing import Any
from gibss.logistic_sparse import make_sparse_logistic_ser1d
from gibss.logisticprofile import make_logisticprofile_hermite_ser
from gibss.logistic_sparse import make_fixed_fitfun
from gibss.logistic import SusieFit
from gibss.additive import AdditiveModel, AdditiveState
from gibss.utils import tree_stack
import jax.numpy as jnp

Array = Any


def sigmoid(x: Array):
    return 1 / (1 + jnp.exp(-x))


def update_twogroup_additive_model(
    log_marginals: Array, model: Any, maxiter=100, tol=1e-3
) -> AdditiveModel:
    # initialization, use data from existing model if it's there
    fitfuns = model.fit_functions
    maxiter = int(maxiter)
    tol = float(tol)
    L = len(fitfuns)

    if model.components is None:
        psi = 0.0
        components = []
        for l in range(L):
            y = sigmoid(log_marginals + psi)
            fit = fitfuns[l](psi, None, y)
            psi = psi + fit.psi

            # for _ in range(50):
            #     y = sigmoid(log_marginals + psi)
            #     psi = psi - fit.psi
            #     fit = fitfuns[l](psi, None, y)
            #     psi = psi + fit.psi

            components.append(fit)
    else:
        components = model.components
        psi = model.psi

    # for monitoring convergance
    diff = jnp.inf
    state = AdditiveState(diff, tol, False, maxiter, 0)
    for i in range(maxiter):
        psi_old = psi
        for l in range(L):
            y = sigmoid(log_marginals + psi)
            psi = psi - components[l].psi
            components[l] = fitfuns[l](psi, components[l], y)
            psi = psi + components[l].psi

        diff = jnp.abs(psi - psi_old).max()
        state = AdditiveState(diff, tol, diff < tol, maxiter, i + 1)
        if diff < tol:
            break
    return AdditiveModel(psi, components, model.fit_functions, state)


def estimate_prior_log_odds_em(log_bf, tolerance=1e-5, maxiter=100):
    """
    Estimate the prior odds using the Expectation-Maximization (EM) algorithm.

    Parameters:
    - log_bf: Array of log Bayes factors.
    - tolerance: Convergence criterion for the change in log prior odds.
    - maxiter: Maximum number of iterations for the EM algorithm.

    Returns:
    - log_prior_odds: Estimated log prior odds.
    """
    log_prior_odds = 0.0  # Initialize log prior odds
    for i in range(maxiter):
        lpo_old = log_prior_odds
        pi1 = sigmoid(log_bf + log_prior_odds).mean()
        log_prior_odds = jnp.log(pi1) - jnp.log(1 - pi1)
        if jnp.abs(lpo_old - log_prior_odds) < tolerance:
            break
    return log_prior_odds


# Example usage:
# log_bf = jnp.array([...])
# result = estimate_prior_odds_em(log_bf, tolerance=1e-5, maxiter=100)

from collections import namedtuple

Susie = namedtuple("SER", ["fixed_effect", "sers", "state"])


def fit_twogroup_logistic_sparse_susie(
    X_sp, log_bf, L, prior_variance=1.0, alpha=0.8, gamma=0.0, **kwargs
):
    """
    log_bf is log p(bhat | z=1) - log p(bhat | z=0)
    """
    # now the fitfuns take y as last argument
    fitfun = make_sparse_logistic_ser1d(X_sp, prior_variance, alpha, gamma)
    fixedfitfun = make_fixed_fitfun()
    fitfuns = [fixedfitfun] + [fitfun for _ in range(L)]
    model = AdditiveModel(None, None, fitfuns, None)
    model = update_twogroup_additive_model(log_bf, model, **kwargs)

    fixed_effect = model.components[0]
    sers = tree_stack(model.components[1:])
    fit = SusieFit(fixed_effect, sers, model.state)
    return fit


def fit_twogroup_logisticprofile_hermite_susie(X, log_bf, L, prior_variance=1.0, m=1):
    fitfun = make_logisticprofile_hermite_ser(X, prior_variance, m)
    fixedfitfun = make_fixed_fitfun()
    fitfuns = [fixedfitfun] + [fitfun for _ in range(L)]
    model = AdditiveModel(None, None, fitfuns, None)
    model = update_twogroup_additive_model(log_bf, model)

    fixed_effect = model.components[0]
    sers = tree_stack(model.components[1:])
    fit = Susie(fixed_effect, sers, model.state)
    return fit
