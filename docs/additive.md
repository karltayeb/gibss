# Additive models 

At it's core GIBSS is fitting an additive model. Each component of the additive model produces a prediction $\psi_l$ which contributes to the total predictions $\psi = \sum \psi_l$. We can estimate each component of the additive model in an iterative fashion, estimating one $\psi_l$ while holding the other $\psi_j$ $j \neq l$ fixed.

When we fit SuSiE via GIBSS, each component corresponds to an SER, and $\psi_l$ are the posterior means of the linear predictor $\mathbb E [X b]$. However, there is no requirement that the additive model consist of homogenous components. For example, we might include an additive component for the intercept and any other covariates that we want to include in the model.


## Additive model interface

```
fit_additive(fitfuns: List[Callable], tol: float=1e-3, maxiter: int=10, keep_intermediate=False) -> (List[Any], AdditiveState)
```


A minimal version of this function could be implemented as

``` py title="fit_additive pseudocode"
def fit_additive_core(fitfuns: List[Callable]) -> (List[AdditiveComponent], AdditiveState):
	psi = 0.
	components = []
	for fun in enumerate(fitfuns, i):
		component = fun(psi, None)
		psi = psi + component.psi
		components.append(component)
	
	while {not converged}
		for fun in enumerate(fitfuns, i):
			psi = psi - components[i].psi
			components[i] = fun(psi, components[i])
			psi = psi + components[i].psi
	return components, state
```

`fitfuns` is a list of functions with signature `fitfun(psi: Array, old_fit: Union[AdditiveComponent, None]) -> AdditiveComponent`.

All of the work goes into designing the additive components.  `fitfun` knows what data it should use, how to initialize itself either with an old fit or from scratch. `fit_additive` is meant just to handle the basic logic of iteratively fitting an additive model. 

There are a few features we add. 
- When the argument `keep_intermediate = True` the function will save all the intermediate states of `components` after each run of the loop.
- We also add arguments for controlling how convergence is monitored. 

## Using `fit_additive` to fit new SuSiE models

The functions that we can pass to `fit_additive` need to have a specific type signature 

```py
def fun(psi: Array, fit: Union[Fit, None]) -> Fit
```
Where `Fit` type represents the output of the fit function.
Importantly, it must be able to handle the case where `fit = None`, as is the case in the first iteration.
The function must know what data, parameters, etc should be used.


To facilitate the development of new additive models, we provide a helper function for building functions that are compatible with `additive_model`

``` py 
def make_fitfun(X: np.ndarray, y: np.ndarray, fun: Callable, initfun: Callable, kwargs: dict) -> Callable:
    """
    Produces a function for fitting an additive component using the provided data and settings. 
    The returned function has a signature compatible with the additive_model interface.

    Args:
        X (np.ndarray): A p x n matrix where p is the number of variables and n is the number of observations.
        y (np.ndarray): An n-dimensional vector of observations corresponding to the rows of X.
        fun (Callable): A function that takes `coef_init`, `X`, `y`, `psi`, and possibly other arguments specified in `kwargs`.
        initfun (Callable): A function with the signature `(X, y, psi, fit) -> Array` used to initialize `coef_init`. 
                            It must be able to handle `fit` as `None`.
        kwargs (dict): A dictionary of additional keyword arguments to pass to `fun`.

    Returns:
        Callable: A function `fitfun(psi: Array, fit: Union[None, Fit]) -> Fit`, used to fit the SER model.
    """
    @jax.jit
    def fitfun(psi, fit):
        coef_init = initfun(X, y, psi, fit)
        return fun(
            coef_init = coef_init,
            X = X, y=y, offset = psi,
            **kwargs
        )
    return fitfun
```

`make_fitfun` assumes that the function `fun` has positional arguments `coef_init`, `X`, `y``, and `psi`. Other arguments can be specified in the dictionary `kwargs` which will be passed as keyword arguments to `fun`. 


For example we could construct a function for fitting the logistic SER with
``` py
from gibss.logistic import logistic_ser_hermite

initfun = lambda X, y, psi, fit: jnp.zeros((X.shape[0], 1))
fitfun = make_fitfun(X, y, logistic_ser_hermite, initfun)
fitfun(0., None)
```

And fit an additive model (logistic SuSiE) with

``` py
fitfuns = [fitfun for _ in range(5)]
fit, state = additive_model(fitfuns)
```

This framework makes it simple to implement new variations of SuSiE. 

* Want to do SuSiE with a new likelihood? Implement the SER for that new likelihood. We have implementations for logistic regression and the Cox proportional hazards model.
* Want to estimate the prior variance? Implement a version of the SER that does that. 
* Want to include fixed effects in the model? Implement a separate additive component that handles estimation of the fixed effect. This is how we handle the intercept in `gibss.logistic.fit_logistic_susie`. 
