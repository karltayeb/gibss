import jax.numpy as jnp
import jax
from functools import partial
from dataclasses import dataclass
from jax.typing import ArrayLike
from jax.scipy.special import logsumexp
from gibss.logistic import make_fixed_fitfun
from gibss.additive import fit_additive_model, AdditiveComponent
from scipy import sparse
from gibss.utils import tree_stack, ensure_dense_and_float, npize
from gibss.logistic import SusieFit, SusieSummary
from gibss.credible_sets import compute_cs
import numpy as np


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["x", "f", "g", "h", "stepsize"],
    meta_fields=[],
)
@dataclass
class OptState:
    x: ArrayLike
    f: ArrayLike
    g: ArrayLike
    h: ArrayLike
    stepsize: ArrayLike


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "b",
        "llr",
        "lbf",
        "alpha",
        "lbf_ser",
        "pi",
        "prior_variance",
        "optstate",
    ],
    meta_fields=[],
)
@dataclass
class SparseSER:
    b: ArrayLike
    llr: ArrayLike
    lbf: ArrayLike
    alpha: ArrayLike
    lbf_ser: float
    pi: ArrayLike
    prior_variance: float
    optstate: OptState


def _get_sizes(partition):
    return (jnp.roll(partition, -1) - partition)[:-1]


def make_sparse_logistic_ser1d(X_sp, y, prior_variance=1.0):
    partition = X_sp.indptr
    sizes = _get_sizes(partition)
    indices = X_sp.indices
    xlong = X_sp.data
    ylong = y[X_sp.indices]

    def _splitsum(long):
        cumsum = jnp.cumsum(long)[partition[1:] - 1]
        carry, splitsum = jax.lax.scan(lambda carry, x: (x, x - carry), 0, cumsum)
        return splitsum

    def _compute_llr(blong, xlong, ylong, psilong, psi0long, tau=1.0):
        psilong = psi0long + blong
        lrlong = (
            ylong * (psilong - psi0long)
            - jnp.log(1 + jnp.exp(psilong))
            + jnp.log(1 + jnp.exp(psi0long))
        )
        return _splitsum(lrlong)

    def decay_stepsize_opt_state(old_opt_state, alpha=0.5):
        return OptState(
            old_opt_state.x,
            old_opt_state.f,
            old_opt_state.g,
            old_opt_state.h,
            old_opt_state.stepsize * alpha,
        )

    def merge_optstate(old_state, new_state):
        return jax.lax.cond(
            (old_state.f < new_state.f),
            lambda old, new: new,
            lambda old, new: decay_stepsize_opt_state(old, 0.5),
            old_state,
            new_state,
        )

    merge_opstate_vmap = jax.vmap(merge_optstate, 0, 0)

    @jax.jit
    def update_b(state, psi0):
        # propose newton step
        psi0long = psi0[indices]
        tau = 1 / state.prior_variance
        blong = jnp.repeat(state.b, sizes)
        psilong = psi0long + blong
        plong = 1 / (1 + jnp.exp(-psilong))
        gradlong = (ylong - plong) * xlong
        hesslong = -plong * (1 - plong) * xlong * xlong
        g = _splitsum(gradlong) - tau * state.b  # gradient of log p(y, b)
        h = _splitsum(hesslong) - tau  # hessian of logp(y, b) < 0
        update_direction = -g / h
        b = state.b + update_direction * state.optstate.stepsize
        llr = (
            _compute_llr(blong, xlong, ylong, psilong, psi0long, tau)
            - 0.5 * tau * b**2
            + jnp.log(tau)
            - jnp.log(2 * jnp.pi)
        )

        # pick between states
        optstate = OptState(b, llr, g, h, jnp.ones_like(llr))
        optstate = merge_opstate_vmap(state.optstate, optstate)

        # ser computations
        lbf = optstate.f + 0.5 * (jnp.log(2 * jnp.pi) - jnp.log(-optstate.h))
        lbf_ser = logsumexp(lbf + jnp.log(state.pi))
        alpha = jnp.exp(lbf + jnp.log(state.pi) - lbf_ser)
        state = SparseSER(
            optstate.x,
            optstate.f,
            lbf,
            alpha,
            lbf_ser,
            state.pi,
            state.prior_variance,
            optstate,
        )
        return state

    def logistic_ser_1d(psi, fit):
        if fit is None:
            p, n = X_sp.shape
            opt0 = OptState(
                jnp.ones(p),
                -jnp.inf * jnp.ones(p),
                jnp.ones(p),
                jnp.ones(p),
                jnp.ones(p),
            )
            state0 = SparseSER(
                jnp.zeros(p), 0, 0, 0, -jnp.inf, jnp.ones(p) / p, prior_variance, opt0
            )
            state = update_b(state0, psi)
        else:
            state = update_b(fit.fit, psi)
        psi = (state.alpha * state.b) @ X_sp
        return AdditiveComponent(psi, state)

    return logistic_ser_1d


def fit_logistic_susie(X_sp, y, L, prior_variance=1.0, **kwargs):
    fitfun = make_sparse_logistic_ser1d(X_sp, y, prior_variance)
    fixedfitfun = make_fixed_fitfun(X_sp, y)
    fitfuns = [fixedfitfun] + [fitfun for _ in range(L)]
    model = fit_additive_model(fitfuns, **kwargs)

    fixed_effect = model.components[0]
    sers = tree_stack(model.components[1:])
    fit = SusieFit(fixed_effect, sers, model.state)
    return fit


def summarize_susie(fit):
    fixed_effects = np.array(fit.fixed_effect.fit.x)
    alpha = np.array(fit.sers.fit.alpha)
    lbf = np.array(fit.sers.fit.lbf)
    beta = np.array(fit.sers.fit.b)
    prior_variance = np.array(fit.sers.fit.prior_variance)
    lbf_ser = np.array(fit.sers.fit.lbf_ser)
    credible_sets = [compute_cs(a).__dict__ for a in alpha]
    res = SusieSummary(
        fixed_effects,
        alpha,
        lbf,
        beta,
        prior_variance,
        lbf_ser,
        credible_sets,
        npize(fit.state.__dict__),
    )
    return res


def fit_logistic_susie2(X, y, L=10, prior_variance=1.0, maxiter=50, tol=1e-3, **kwargs):
    X_sp = sparse.csr_matrix(X)
    y = ensure_dense_and_float(y)
    fit = fit_logistic_susie(
        X, y, L=L, prior_variance=prior_variance, maxiter=maxiter, tol=tol, **kwargs
    )
    summary = summarize_susie(fit)
    return summary
