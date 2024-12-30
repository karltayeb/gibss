# A simple newton method
# 1. run for a fixed number of *function evaluations*
# 2. try taking a newton step with stepsize = 1
# 3. if the likelihood increasses, keep the proposed move and set the stepsize to 1
# 4. if the likelihood decreases, reject the proposed move and halve the step size
# Here we assume `f` is a function to MINIMIZE
import jax
import jax.numpy as jnp
from functools import partial
from dataclasses import dataclass
import numpy as np
from jax.tree_util import Partial
from typing import Callable
@partial(jax.tree_util.register_dataclass,
         data_fields=['x', 'f', 'g', 'h', 'direction', 'stepsize', 'nd', 'tol', 'converged', 'alpha', 'gamma', 'maxiter', 'iter'], meta_fields=[])
@dataclass
class NewtonState:
    x: jax.Array
    f: float
    g: jax.Array
    h: jax.Array
    direction: jax.Array
    stepsize: float
    nd: float
    tol: int
    converged: bool
    alpha: float
    gamma: float 
    maxiter: int
    iter: int

def newton_step(state, fun, grad, hess):
    # take newton step proposed by state
    x = state.x + state.stepsize * state.direction

    # compute likelihood at new value
    f = fun(x)

    # update state if objective decreases
    def f1(x, dir, f, fcur):
        g = grad(x)
        H = hess(x)
        nd = (g @ H @ g)/2
        return NewtonState(
            x, f, g, H, -jax.scipy.linalg.solve(H, g), 
            1.,
            nd, state.tol, nd < state.tol,
            state.alpha, state.gamma,
            state.maxiter, state.iter + 1
        )
    # shrink stepsize by factro alpha if objective does not decrease
    def f2(x, dir, f, fcur):
        return NewtonState(
            state.x, state.f, state.g, state.h, state.direction, 
            state.stepsize * state.alpha,
            state.nd, state.tol, False,
            state.alpha, state.gamma,
            state.maxiter, state.iter + 1
        )
    # update state if we accept move, backtracking/Armijo-Goldstein with c=0
    new_state = jax.lax.cond(f <= (state.f - state.gamma * jnp.sum(state.g**2)), f1, f2, x, state.direction, f, state.f)
    # new_state = jax.lax.cond(True, f1, f2, x, f, state.f)
    return new_state

def newton(x0, fun, grad, hess, tol=1e-3, maxiter=10, alpha=0.5, gamma=-0.1):
    def converged_or_maxiter_reached(state):
        return jax.lax.cond(
            (state.nd <= state.tol)| (state.iter >= state.maxiter),
            lambda: False,
            lambda: True
        )
    body_fun = Partial(newton_step, fun=fun, grad=grad, hess=hess)
    f = fun(x0)
    g = grad(x0)
    H = hess(x0)
    direction = -jax.scipy.linalg.solve(H, g)
    state = NewtonState(x0, f, g, H, direction, 1.0, np.inf, tol, False, alpha, gamma, maxiter, 0)
    state = jax.lax.while_loop(converged_or_maxiter_reached, body_fun, state)
    return state

def newton_factory(f: Callable, maxiter: int=50, tol: float=1e-3, alpha: float=0.5, gamma: float=-jnp.inf) -> Callable:
    """Create a Newton optimizer for a function

    Args:
        f (Callable): the function to be minimized
        maxiter (int, optional): Maximum number of iteration. Defaults to 50.
        tol (float, optional): tolerance to Newton's decrement to declare convergence. Defaults to 1e-3.
        alpha (float, optional): stepsize scaling parameter for backtracking line search takes values in (0, 1). Defaults to 0.5.
        gamma (float, optional): sufficient decrease parameter for backtracking line search, takes values in (0, 1). By setting to -inf we always accept the step. Defaults to -jnp.inf.

    Returns:
        Callable: an optimizer function which accepts an initialization of the parameters
    """
    
    fp = Partial(f)
    grad = Partial(jax.grad(f))
    hess = Partial(jax.hessian(f))
    return partial(newton, fun=fp, grad=grad, hess=hess, maxiter=maxiter, tol=tol, alpha=alpha, gamma=gamma)

@partial(jax.jit)
def newton_lite(x0, fun, grad, hess, tol=0, niter=10, alpha=0.5, gamma=0.):
    body_fun = Partial(newton_step, fun=fun, grad=grad, hess=hess)
    f = fun(x0)
    g = grad(x0)
    H = hess(x0)
    direction = -jax.scipy.linalg.solve(H, g)
    state = NewtonState(x0, f, g, H, direction, 1.0, np.inf, tol, False, alpha, gamma, niter, 0)
    state = jax.lax.fori_loop(0, niter, lambda i, s: body_fun(s), state)
    return state

def newton_lite_factory(f: Callable, niter: int=10, tol: float=1e-3, alpha: float=0.5, gamma: float=-jnp.inf) -> Callable:
    fp = Partial(f)
    grad = Partial(jax.grad(f))
    hess = Partial(jax.hessian(f))
    return partial(newton_lite, fun=fp, grad=grad, hess=hess, niter=niter, tol=tol, alpha=alpha, gamma=gamma)

def newton_save_iterates(x0, fun, niter=10):
    grad = jax.grad(fun)
    hess = jax.hessian(fun)
    step = partial(newton_step, fun=fun, grad=grad, hess=hess)
    state = NewtonState(x0, fun(x0), grad(x0), hess(x0), 1.0, np.inf, 1e-3, False, niter, 0)
    states = []
    for i in range(niter):
        state = step(state)
        states.append(state)
    return states
