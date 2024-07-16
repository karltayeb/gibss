#%%
import numpy as np
from gibss.logisticprofile import logistic_susie
from gibss.logisticprofile import logistic_ser_hermite
from gibss.logisticprofile import initialize_coef
from gibss.logisticprofile import fit_null
from gibss.utils import todict

X = np.random.normal(size = (500, 2000))
x = X[0]
y = np.random.binomial(1, 1/(1 + np.exp(-(-1 + 0.5 * x))))
offset = 0.
prior_variance = 1.

nullfit = fit_null(y, 0.)

#%%
coef_init = initialize_coef(X, y, 0, 1.)
serfit = logistic_ser_hermite(coef_init, X, y, 0., 1.0, 5)

#%%
from gibss.logisticprofile import logistic_ser_lapmle
from gibss.logisticprofile import logistic_ser_lapmle_eb
serfit_eb = logistic_ser_lapmle_eb(coef_init, X, y, 0., 100.0)
serfit = logistic_ser_lapmle(coef_init, X, y, 0., 100.0)

#%%
from gibss.logisticprofile import logistic_ser_wakefield
from gibss.logisticprofile import logistic_ser_wakefield_eb
serfit_eb_wakefield = logistic_ser_lapmle_eb(coef_init, X, y, 0., 100.0)
serfit_wakefield = logistic_ser_lapmle(coef_init, X, y, 0., 100.0)

#%%
y2 = np.random.binomial(1, np.mean(y) * np.ones_like(y))
serfit_eb = logistic_ser_lapmle_eb(coef_init, X, y2, 0., 100.0)
serfit = logistic_ser_lapmle(coef_init, X, y2, 0., 100.0)

#%%
susiefit = logistic_susie(X, y, L=1, maxiter=1, tol=1e-5)
susiedict = todict(susiefit) 

#%%
susiefit = logistic_susie(X, y, L=3, maxiter=50, tol=1e-5)
susiedict = todict(susiefit) 

#%%
susiefit = logistic_susie(X, y, L=10, maxiter=50, tol=1e-5)
susiedict = todict(susiefit) 

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