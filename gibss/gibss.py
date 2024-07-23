from gibss.additive import additive_model, Monitor
import numpy as np

def gibss(X, y, L=5, prior_variance=1.0, maxiter=100, tol=1e-3, initfun = None, serfun = None):
    def fit_ser(psi, old_component):
        # TODO: can we estimate the prior variance here using the last fit?
        # So at each iteration we fit with a fixed prior variance,
        # but we give the opportuity to update the prior variance before the next iteration.
        # TODO: maybe there should be a `params` attribute to component
        # that way we don't depend on the details of how the component is fit
        coef_init = old_component.fits.state.x
        return serfun(coef_init, X, y, psi, old_component.prior_variance)
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
    monitor = Monitor(components, tol)
    return additive_model(psi, components, [fit_ser for _ in range(L)], maxiter=maxiter, monitor=monitor)
