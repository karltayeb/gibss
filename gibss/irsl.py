# iteratively reweighted least squares (IRLS)
# which is equivalent to pure Newton method
# we'll see if we can implement it to be faster than `gibss.newton`
import jax.numpy as jnp
import jax

def wls(X, z, w, prior_variance):
    # weighted least squares with l2 regularization
    WX = w[:, None] * X
    Lambda = jnp.diag(0.5 / prior_variance)
    H = X.T @ WX + Lambda
    beta = jnp.linalg.solve(H, WX.T @ z)
    return beta, H

def sigmoid(x):
    return 1/(1 + jnp.exp(-x))

def irls_step(b, X, y, offset, prior_variance):
    psi = X @ b + offset
    p = sigmoid(psi)
    w = p * (1-p)
    z = psi + (y - p) / w
    return wls(X, z, w, prior_variance)

def irls(b_init, X, y, offset, prior_variance, niter=10):
    b = b_init
    for _ in range(niter):
        b, H = irls_step(b, X, y, offset, prior_variance)
    return b, H

virls = jax.vmap(irls, in_axes=(0, 0, None, None, None, None))
virls_jit = jax.jit(virls, static_argnames=['niter'])