
# SMASHY: Synchronized Model-Assisted Hyperband

[![check](https://github.com/mlr-org/smashy/actions/workflows/check.yml/badge.svg)](https://github.com/mlr-org/smashy/actions/workflows/check.yml)
[![Coverage](https://codecov.io/github/mlr-org/smashy/branch/master/graphs/badge.svg)](https://codecov.io/github/mlr-org/smashy)

## Project Status

Although `smashy` is currently still evolving, it can already be used for optimization. All exported functions (should) work and are [documented](https://mlr-org.github.io/smashy/reference/index.html).

`smashy` itself is functionally complete and will find its way to CRAN. In its current shape, it contains many auxiliary
classes that will be part of the [`miesmuschel`](https://github.com/mlr-org/miesmuschel) package instead, however.

## Installation

Install the github version, as well as some dependencies, using `remotes`. 

```r
remotes::install_github("mlr-org/paradox@expression_params")
remotes::install_github("mlr-org/smashy")
```

## Example Usage

The most convenient way to use `smashy` is through the `configure_smashy()`-function. 
`smashy` is designed with parallel evaluation in mind. Therefore, the following
also enables parallelization using `future`.
```r
library("smashy")
# Define the objective to optimize
# The 'budget' here simulates averaging 'b' samples from a noisy function
objective <- ObjectiveRFun$new(
  fun = function(xs) {
    z <- exp(-xs$x^2 - xs$y^2) + 2 * exp(-(2 - xs$x)^2 - (2 - xs$y)^2)
    z <- z + rnorm(1, sd = 1 / sqrt(xs$b))
    list(Obj = z)
  },
  domain = ps(x = p_dbl(-2, 4), y = p_dbl(-2, 4), b = p_int(1)),
  codomain = ps(Obj = p_dbl(tags = "maximize"))
)

search_space_proto = objective$domain$search_space(list(
  x = to_tune(),
  y = to_tune(),
  b = to_tune(p_int(1, 2^10, tags = "budget"))
))

# Need to put the "budget" component to log-scale for smashy:
search_space = budget_to_logscale(search_space_proto)

# Get a new OptimInstance. Here we determine that the optimizatoin goes
# for 100 full budget evaluations (100 * 2^100)
oi <- OptimInstanceSingleCrit$new(objective,
  search_space = search_space,
  terminator = trm("budget", budget = 100 * 2^10)
)

future::plan("multisession")

# Ideally, set `mu` to a low multiple of the number of available CPU cores.
# Most other default values are chosen to give relatively good, robust
# performance in many settings while being fast and simple.

smashy = configure_smashy(search_space, budget_log_step = log(4), mu = 6)

smashy$optimize(oi)
```

## Getting Help

`smashy` provides optimizers that work with the [bbotk](https://github.com/mlr-org/bbotk) package:
A [`OptimInstanceSingleCrit`](https://bbotk.mlr-org.com/reference/OptimInstanceSingleCrit.html) needs to be defined, which itself needs an [`Objective`](https://bbotk.mlr-org.com/reference/Objective.html) and a [`ParamSet`](https://paradox.mlr-org.com/reference/ParamSet.html) that defines a search space.

`smashy` can also be used for hyperparameter optimization (HPO) of [`mlr3`](https://mlr3.mlr-org.com/) machine learning algorithms:
See the [mlr3book](https://mlr3book.mlr-org.com/04-optimization.html) chapter for an introduction of how HPO works in [`mlr3`].

It is easiest to use `smashy`, as in the example above, by using [`configure_smashy()`](https://mlr-org.github.io/smashy/reference/configure_smashy.html) (making sure the budget parameter is in log-scale with [`budget_to_logscale()`](https://mlr-org.github.io/smashy/reference/budget_to_logscale.html) beforehand).
Functions that emulate [Hyperband](https://arxiv.org/abs/1603.06560?context=cs) and [BOHB](https://www.automl.org/blog_bohb/) are also provided: [`smashy_as_hyperband()`](https://mlr-org.github.io/smashy/reference/smashy_as_hyperband.html) and [`smashy_as_bohb()`](https://mlr-org.github.io/smashy/reference/smashy_as_bohb.html). 

You can also see the [reference](https://mlr-org.github.io/smashy/reference/index.html) for other classes and methods, although these are mostly for internal use and not of interest for the typical user.
