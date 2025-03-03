# Generalized IBSS

This package implements Generalized iterative Bayesian stepwise selection (GIBSS), which is a heuristic for approximating the posterior distribution to the generalized sum of single effects regression (GSuSiE).
The idea is simple: accurately estimate the single effect regression, which can be done through various approximation techniques (Laplace, quadrature, etc.), conditional on point estimates from other effects.
A natural criticism of the approach is that it does not propagate uncertainty in the point estimates. However, we find empirically it works well in many situations. 

We have implemented several variations of logistic regression (exploiting sparsity, different treatments of the intercept, different approximations of the Bayes factor), Cox regression.

## Installation
`gibss` is hosted on PyPI

```bash
python3 -m venv venv
source venv/bin/activate # On Windows use `venv\Scripts\activate`
pip install gibss
```

Alternatively, you can clone this package and install from local

```bash
git clone https://github.com/yourusername/gibss.git
cd gibss
python3 -m venv venv
source venv/bin/activate # On Windows use `venv\Scripts\activate`
pip install .
```

## Minimal example

```python
from gibss.logistic1d import fit_logistic1d_susie
import numpy as np

n = 1000 # number of observations
p = 50 # number of variables
X = np.random.normal(size=(p, n))
logit = X[0] - 1
y = 1 / (1 + np.exp(-logit))

# fit with 5 effects
fit = fit_logistic1d_susie(X, y, L=5)
```

