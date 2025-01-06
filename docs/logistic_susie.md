# Logistic SuSiE

We have implemented two versions of logistic SuSiE. For the moment we recommend using the version in `gibss.logistic` module.
The most important design decision made in this module is to handle the intercept and other fixed covariates in their own component of the additive model.
This dramatically simplifies implementation of the SER, where each sub-problem corresponds to fitting a simple univariate logistic regression.
It is much easier to develop a fast, stable optimization scheme for this 1d problem.


Typically, we would also want to estimate an intercept, or the effect of other covariates that we would want to include in the model (e.g. genotype PCs for GWAS or eQTL studies). The univariate approach has the advantage that we can flexibly specify how to handle these effects independently from the SER.
For example, we could change the regularization/priors on the fixed effect without needing to implement a new SER.

### Fitting a single effect regression (SER) model

The single effect regression is a simple model where exactly one of $p$ variables has a non-zero effect. Compared to more general variable selection problems inference in the SER is tractable because we can enumerate all possible configurations of non-zero effects. Inference in the SER can be carried out as follows.

1. For each variable  $j = 1 \dots p$

    1. Compute the MAP estimate for the effect $b_j$, $\hat b_j$.
    1. Approximate the Bayes Factor $\text{BF}_j$, $\widehat{\text{BF}}_j$

1. Compute the (approximate) posterior inclusion probabilities $\alpha_j \propto \widehat{\text {BF}}_j$
1. For each observation $i = 1, \dots, n$, compute the (approximate) posterior mean predictions $\psi_i = \sum \alpha_j \hat b_j x_{ij}$.

### The logistic SER

When you call `logistic.gibss.fit_logistic_susie` with `method='hermite'`, 
you fit a logistic SER where the posterior mean and Bayes factor are approximated using adaptive Gauss-Hermite quadrature. When the quadrature rule uses $m=1$ points, this corresponds to the usual Laplace approximation. 

#### Estimating the prior variance

We offer the option to estimate the prior variance. A simple way to estimate the prior variance is to evaluate the SER along a fixed grid of prior variance settings, and then selecting the SER with the highest marginal likelihood.

We think that this approach is sufficient, compared to more precise maximization of the prior variance. It is implemented in `gibss.logistic.logistic_ser_hermite_grid`, which takes an argument `prior_variance_grid`. 



#### Rationale for default optimization hyperparameters

We use the following defaults in `gibss.logistic.` when `method = 'hermite'`

``` py
defaultserkwargs  = dict(
    prior_variance = float(10.),
    newtonkwargs=dict(tol=1e-2, maxiter=5, alpha=0.2, gamma=-0.1)
)
```

1. **Prior variance** We take set the `pior_variance = 10.`. The appropriate choice of prior variance is of course dependent on the scale of the covariates. If we assume that the user are providing covariates where a unit increase in the covariate is non-negligble (e.g. increased dosage of the derived allele, inclusion in a gene set), then a prior variance of $10$ corresponds to pretty weak regularization of the effect. I think setting the default prior variance to a rather large value is safe in the sense that we will get smaller Bayes factors which will cause us to be conservative when evaluating evidence for the presence of a non-zero effect in the SER.
1. **Number of iteration** We estimate the MAP using Newton's method with backtracking line search. We use just $5$ iterations of Newton's method for each SER, but use the estimates from this iteration to initialize the next iteration. That is, we initialize component $l$ at iteration $t+1$ with the estimates from component $l$ at iteration $t$. 
Across several iterations of the outer loop, as `psi` stabilizes, the optimization problem in the inner loop remains unchanged.
Heuristically, we save computational effort by not optimizing very precisely the intermediate objectives that are liable to change drastically iteration to iteration, and by leveraging the previous approximate optima when the problems are similar.
1. **Settings for backtracking line search** For convex problems, Newton exhibits fast (quadratic) convergence within a neighborhood of the optima with stepsize $1$.Away from the optimum, the Newton update is guarunteed to be a descent direction but the step size may need tuning. We start with a stepsize of one and decay geometrically until the objective improves (or at least, does not increase too much). Since we are allowing just $5$ evaluations of the objective, and we would like to ensure that our effect gets updated at each stage, we set the step size scaling factor to $0.2$, which gives a minimum stepsize of $0.2^5$.
In practice this minimum step size is small enough that we will improve the objective at each iteration.
1. **Sufficient decrease parameter**. The sufficient decrease parameter `gamma` says to accept a move if $f(x_{t+1}) < f(x_t) + \gamma ||g||_2^2$ Where $g$ is the gradient. We actually allow slight decrease in the objective function by setting $\gamma = -0.1$. The optimization is implemented in JAX, and we find that with 32bit floats, and optimization by the compiler it is not uncommon to dramatic decrease in the sub-optimality of a solution while actually seeing very slight increases in the objective. Therefore to avoid the optimization procedure getting stuck we allow for slight increases in the objective. It would be good to better understand the cause of these numerical issues.


#### Adjusting the optimization hyperparameters

The choice of optimization hyper-parameters can have a dramatic effect on performance.
Newton's method with backtracking line search is guaranteed to converge, 
but because we are repeatedly computing MAP estimates for many sub-problems, it pays to be mindful of the balance between accuracy and computation.
