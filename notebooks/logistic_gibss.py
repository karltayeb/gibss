#%%
import numpy as np
from gibss.logisticprofile import logistic_susie

X = np.random.normal(size = (100, 1000))
x = X[0]
y = np.random.binomial(1, 1/(1 + np.exp(-(-1 + 0.5 * x))))
offset = 0.
prior_variance = 1.


logistic_susie(X, y, offset, L=3, maxiter=3)
# %%
from gibss.logisticprofile import initialize_coef, logistic_ser_hermite
coef_init = initialize_coef(X, y, 0., prior_variance)
component = logistic_ser_hermite(coef_init, X, y, np.zeros_like(y), 1e-10)

def fit_ser(psi, old_component):
    coef_init = old_component.fits.params
    # TODO: can we estimate the prior variance here using the last fit?
    # So at each iteration we fit with a fixed prior variance,
    # but we give the opportuity to update the prior variance before the next iteration.
    return logistic_ser_hermite(coef_init, X, y, psi, prior_variance)

component2 = fit_ser(-component.psi, component)