from gibss.additive import additive_model
import numpy as np

def gibss(X, y, L=5, prior_variance=1.0, maxiter=100, tol=1e-3, initfun = None, serfun = None):
    def fit_ser(psi, old_component):
        # TODO: can we estimate the prior variance here using the last fit?
        # So at each iteration we fit with a fixed prior variance,
        # but we give the opportuity to update the prior variance before the next iteration.
        coef_init = old_component.fits.params
        return serfun(coef_init, X, y, psi, prior_variance)
    coef_init = initfun(X, y, 0., prior_variance)

    # first iteration to build up the components
    components = []
    psi = np.zeros_like(y)
    for _ in range(L):
        coef_init = initfun(X, y, psi, prior_variance)
        component = serfun(coef_init, X, y, psi, prior_variance)
        components.append(component)
        psi = psi + component.psi

    # subsequent iterations to refine the components
    return additive_model(psi, components, [fit_ser for _ in range(L)])
