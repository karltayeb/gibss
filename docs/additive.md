# Additive models 

At it's core GIBSS is fitting an additive model. Each component of the additive model produces a prediction $\psi_l$ which contributes to the total predictions $\psi = \sum \psi_l$. We can estimate each component of the additive model in an iterative fashion, estimating one $\psi_l$ while holding the other $\psi_j$ $j \neq l$ fixed.

When we fit SuSiE via GIBSS, each component corresponds to an SER, and $\psi_l$ are the posterior means of the linear predictor $\mathbb E [X b]$. However, there is no requirement that the additive model consist of homogenous components. For example, we might include an additive component for the intercept and any other covariates that we want to include in the model.


### Additive model interface

`fit_additive(fitfuns: List[Callable], tol: float=1e-3, maxiter: int=10, keep_intermediate=False) -> (List[Any], AdditiveState)`

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

### Using `fit_additive` to fit new SuSiE models

We will assume that we have a function for fitting a single effect regression.
The details of how this function are implemented are unimportant, except that it takes as an argument an offset. This is important because each additive component should be sensitive to the contributions of the other components.

For example, we will use `gibss.logistic.logistic_ser_hermite`. This function implements the logistic single effect regression. It has a signature `logistic_ser_hermite(coef_init, X, y, offset, m=1, prior_variance=1.0, newtonkwargs=dict())`. 

To construct a valid `fitfun` for use with `fit_additive` we will construct a new function that handles initialization of `coef_init` from the value provided for `old_fit`, and evaluates `logistic_ser_hermite` with this initialization, the data, and other arguments at a given choice of `psi`. 

``` py
def make_fitfun(X, y, **kwargs):
	def fitfun(psi: Array, old_fit: Union[AdditiveComponent, None]):
		# 1. handle different initializations
		if old_fit is None:
			# initialize the coefficients, e.g. using the data (X, y)
			coef_init = ... 
		else:
			# optinally, provide a different initialization if we already have a fit AdditiveComponent
			coef_init = ... 
		# 2. return partial evaluation
		return logistic_ser_hermite(coef_init, X, y, psi, **kwargs)
		
	return fitfun
```

Then a basic implementation of logistic SuSiE looks like

``` py
def logistic_susie_simple(X, y, L):
	fitfun = makefitfun(X, y)
	fitfuns = [fitfun for _ in range(L)]
	components, state = fit_additive(fitfuns)
	return components, state
```

This framework makes it simple to implement new variations of SuSiE. Want to do SuSiE with a new likelihood? Implement the SER for that likelihood and iterate in the additive model. Want to estimate the prior variance? Implement a version of the SER that does that. Want to include fixed effects in the model? Implement a separate additive component that handles estimation of the fixed effect (this is how we handle the intercept in `gibss.logistic`). 
