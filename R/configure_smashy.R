
# --- possible surrogate learners
prepare_learnerlist = function() {
# Imputation pipeline; needs to be robust against NAs that only occur during prediction
imputepl <- po("imputeoor", offset = 1, multiplier = 10) %>>% po("fixfactors") %>>% po("imputesample")
#library("mlr3learners")

# learners used in the paper
learnerlist <- list(
  ranger = GraphLearner$new(imputepl %>>% mlr3::lrn("regr.ranger", fallback = mlr3::lrn("regr.featureless"), encapsulate = c(train = "evaluate", predict = "evaluate"))),
  knn1 = GraphLearner$new(imputepl %>>% mlr3::lrn("regr.kknn", k = 1, fallback = mlr3::lrn("regr.featureless"), encapsulate = c(train = "evaluate", predict = "evaluate"))),
  knn7 = GraphLearner$new(imputepl %>>% mlr3::lrn("regr.kknn", k = 7, fallback = mlr3::lrn("regr.featureless"), encapsulate = c(train = "evaluate", predict = "evaluate"))),
  bohblrn = GraphLearner$new(
    po("colapply", id = "colapply0", applicator = as.factor, affect_columns = selector_type("character")) %>>%
    po("fixfactors") %>>%
    po("removeconstants", id = "priorremoveconstants", affect_columns = selector_type("factor")) %>>%  # prevent 0-level-columns
    po("colapply", applicator = as.numeric, affect_columns = selector_type("integer")) %>>%
    po("stratify", predict_choice = "exact_or_less") %>>%
    list(
      po("densitysplit") %>>%
      # would love to use a multiplicity here, but nested mults have problems when the outer one is empty, which can actually happen here.
      list(
          po("removeconstants", id = "goodremoveconstants") %>>%
          po("imputesample", id = "goodimputesample") %>>%
          mlr3::lrn("density.np", id = "gooddensity", bwmethod = "normal-reference-numeric", min_bandwidth = 1e-3),
          po("removeconstants", id = "badremoveconstants") %>>%
          po("imputesample", id = "badimputesample") %>>%
          mlr3::lrn("density.np", id = "baddensity", bwmethod = "normal-reference-numeric", min_bandwidth = 1e-3)
      ) %>>%
      po("densityratio") %>>%
      po("predictionunion", collect_multiplicity = TRUE),
      mlr3::lrn("regr.featureless")
    ) %>>% po("predictionunion", id = "fallback_union")
  )
)

# --- set up for usage in surrogate pipeline

# prevent k > n segfault, since CRAN version of KKNN is ancient
# (https://github.com/KlausVigo/kknn/issues/25)
learnerlist$knn7$graph$pipeops$regr.kknn$param_set$context_available = "task"
learnerlist$knn7$param_set$values$regr.kknn.k = ContextPV(function(task) if (task$nrow < 8) stop("need 8 samples") else 7)

# bohb emulation
learnerlist$bohblrn$graph$pipeops$stratify$param_set$context_available = "inputs"
learnerlist$bohblrn$graph$pipeops$densitysplit$param_set$context_available = "inputs"
learnerlist$bohblrn$graph$pipeops$stratify$param_set$values$min_size = ContextPV(function(inputs) inputs[[1]]$ncol + 2)
learnerlist$bohblrn$graph$pipeops$densitysplit$param_set$values$min_size = ContextPV(function(inputs) inputs[[1]]$ncol + 1)
learnerlist$bohblrn$graph$pipeops$stratify$param_set$values$stratify_feature = ContextPV(function(inputs) stop("needs to be set to the budget param"))

# GraphLearner doesn't know that it is a regression learner
# we have to set this explicitly to pass assertions later.
learnerlist <- lapply(learnerlist, function(x) { class(x) <- c("LearnerRegr", class(x)) ; x })

learnerlist
}

# --- bohb-like sampling
generate_design_bohb = ContextPV(function(inst) function(param_set, n) {
  target = inst$archive$codomain$ids()
  if (!nrow(inst$archive$data)) return(paradox::generate_design_random(param_set, n))
  task = mlr3::TaskRegr$new("archive", inst$archive$data[, c(inst$archive$search_space$ids(), target), with = FALSE], target = target)
  sampler = SamplerKD$new(param_set, task, inst$archive$codomain$tags[[1]] == "minimize", alpha = .15, min_points_in_model = 0, bandwidth_factor = 3, min_bandwidth = 1e-3)
  sampler$sample(n)
})

# --- ContextPV helpers

# for simulated annealing
get_progress <- function(inst) {
  prog = inst$terminator$status(inst$archive)
  if (prog["max_steps"] <= 0) return(1)
  min(1, max(0, prog["current_steps"] / prog["max_steps"]))
}

# interpolate context param value
interpolate_cpv <- function(beginning, end, logscale = FALSE, round = FALSE) {
  ContextPV(function(inst) {
    prog = get_progress(inst)
    result = if (logscale) {
      exp(prog * log(end) + (1 - prog) * log(beginning))
    } else {
      prog * end + (1 - prog) * beginning
    }
    if (round) result = round(result)
    result
  }, beginning, end, get_progress, logscale, round)
}

# --- objective construction

#' @title Configure Full Smashy Optimizer.
#'
#' @description
#' Create a [`OptimizerSmashy`] (or [`TunerSmashy`]) object with extended functionality from
#' basic building blocks. While basic Smashy-optimizers can be created with `opt("smashy")`
#' and `tnr("smashy")`, setting them up for functionality such as BOHB-like surrogate
#' model filtering or change of filter configuration over the course of optimization
#' can be difficult, hence this helper function. `configure_smashy()` should be used
#' when the extended functionality is desired.
#'
#' @section Simulated Annealing:
#' For the function arguments `filter_factor_first`, `filter_factor_last`,
#' `select_per_tournament`, and `random_interleave_fraction`, it is possible to also give
#' corresponding `*.end` arguments. 
#' When the corresponding `*.end` argument differs from the given value, then the effective
#' value changes over the course of optimization, starting at the given value and progressing
#' geometrically (for `filter_factor_first`, `filter_factor_last`, and `select_per_tournament`) or
#' linearly (for `random_interleave_fraction`) towards the `*.end`-value at the end.
#'
#' @param search_space ([`ParamSet`][paradox::ParamSet])\cr
#'   search space, has one parameter tagged 'budget'.
#' @param budget_log_step (`numeric(1)`)\cr
#'   `log()` of budget fidelity steps to make. E.g. `log(2)` for doubling. Must always be on log scale,
#'   even when `budget_is_logscale` is `FALSE`.
#' @param mu (`integer(1)`)\cr
#'   Population size, must be at least 2.
#' @param survival_fraction (`numeric(1)`)\cr
#'   Fraction of individuals that survive each round. Must be between 0 and 1. If `mu` is very large,
#'   it is decreased so that `round(mu * survival_fraction)` is < `survival_fraction`. Defaults to `1 / 3`.
#' @param sample (`character(1)`)\cr
#'   Sampling mechanism: Must be either `"random"` (sample uniformly at random), or `"bohb"` (use
#'   BOHB sampling from kernel density estimate of well performing points). Default `"bohb"`.
#' @param batch_method (`character(1)`)\cr
#'   Fidelity schedule, one of `"smashy"` or `"hb"`. `"smashy"` uses synchronized batches, `"hb"`
#'   does scheduling similar to Hyperband, but generalized for possibly disagreeing values of
#'   `budget_log_step` and `survival_fraction`. For true Hyperband (up to rounding), set `budget_log_step` to
#'   `log(survival_fraction)` and `mu` to `survival_fraction ^ -fidelity_steps` (where `fidelity_steps`
#'   counts from 0 for "no step, only highest budget is evaluated"). Defaults to `"smashy"`.
#' @param filter_algorithm (`character(1)`)\cr
#'   One of `"tournament"` (using [`FiltorSurrogateTournament`]) or `"progressive"`
#'   (using [`FiltorSurrogateProgressive`]). Defaults to `"tournament"`.
#' @param surrogate_learner (`character(1)`)\cr
#'   Surrogate learner to use for filtering. One of `"knn1"`, `"knn7"` (KNN with `k` = 1 or `k` = 7, respectively),
#'   `"ranger"` (random forest) or `"bohblrn"` (emulating the TPE used in the BOHB paper). Defaults to `"knn1"`.
#' @param filter_with_max_budget (`logical(1)`)\cr
#'   Whether to use the surrogate learner to predict performance at the maximum budget value that has been
#'   evaluated thus far, or at the budget value being currently evaluated. Behavour of these only disagrees
#'   once at least one full-budget configuration has been evaluated. Default `TRUE`.
#' @param filter_factor_first (`numeric(1)`)\cr
#'   When filtering the first configuration using the [`FiltorSurrogateTournament`] or [`FiltorSurrogateProgressive`],
#'   how many random samples to consider. Corresponds to `filter.tournament_size/filter.per_tournament` in
#'   [`FiltorSurrogateTournament`]
#'   and `filter.pool_factor` in [`FiltorSurrogateProgressive`]. Default 100.\cr
#'   Note the *Simulated Annealing* section about `filter_factor_first.end`.
#' @param filter_factor_last (`numeric(1)`)\cr
#'   When filtering the last configuration using the [`FiltorSurrogateTournament`] or [`FiltorSurrogateProgressive`],
#'   how many random samples to consider. Corresponds to `filter.tournament_size_last/filter.per_tournament` in
#'   [`FiltorSurrogateTournament`]
#'   and `filter.pool_factor_last` in [`FiltorSurrogateProgressive`]. Default to the value given to
#'   `filter_factor_first`.
#'   Note the *Simulated Annealing* section about `filter_factor_last.end`.
#' @param filter_select_per_tournament (`integer(1)`)\cr
#'   Only when `filter_algorithm` is `"tournament"`: How many configurations to draw per tournament. Corresponds
#'   to `filter.per_tournament` of [`FiltorSurrogateTournament`]. Must be at least 1. Defaults to 1.
#'   Note the *Simulated Annealing* section about `filter_select_per_tournament.end`.
#' @param random_interleave_fraction (`numeric(1)`)\cr
#'   How many configurations to not filter using the surrogate in each generation and to instead draw
#'   randomly using the `sample` method. Must be between 0 and 1, where a value of 1 means no surrogate-based filtering
#'   is done at all, all values are drawn randomly. Defaults to `1 / 3`.\cr
#'   Note the *Simulated Annealing* section about `random_interleave_fraction.end`.
#' @param filter_factor_first.end (`numeric(1)`)\cr
#'   Corresponds to `filter_factor_first`, see the *Simulated Annealing* section.
#' @param filter_factor_last.end (`numeric(1)`)\cr
#'   Corresponds to `filter_factor_last`, see the *Simulated Annealing* section.
#' @param filter_select_per_tournament.end (`integer(1)`)\cr
#'   Corresponds to `filter_select_per_tournament`, see the *Simulated Annealing* section.
#' @param random_interleave_fraction.end (`numeric(1)`)\cr
#'   Corresponds to `random_interleave_fraction`, see the *Simulated Annealing* section.
#' @param random_interleave_random (`logical(1)`)\cr
#'   Whether during each sampling, a fixed fraction of `random_interleave_fraction` individuals
#'   is drawn randomly (`FALSE`), or the number of individuals to draw randomly is itself random (`TRUE`).
#'   In the latter case, each new configuration has an independent chance of being drawn randomly with
#'   probability `random_interleave_fraction`. Default `TRUE`.
#' @param budget_is_logscale (`logical(1)`)\cr
#'   Whether the component of `search_space` tagged as `"budget"` is in log-scale (`TRUE`) or not (`FALSE`).
#'   When `budget_is_logscale` is given as `TRUE`, then evaluations are done with linear steps of `budget_log_step`.
#'   Otherwise, exponential steps of `exp(budget_log_step)` are performed. Default `FALSE`.
#' @param type (`character(1)`)\cr
#'   One of `"Optimizer"` or `"Tuner"`. What class to return: An [`Optimizer`][bbotk::Optimizer] for
#'   optimizing [`OptimInstanceSingleCrit`][bbotk::OptimInstanceSingleCrit] objects, or a
#'   [`Tuner`][mlr3tuning::Tuner] for tuning [`mlr3::Learner`]s. Defaults to `"Optimizer"`.
#' @return [`OptimizerSmashy`] or [`TunerSmashy`], depending on the `type` argument: The configured *Smashy* optimizer.
#' @examples
#' library("bbotk")
#'
#' # Define the objective to optimize
#' # The 'budget' here simulates averaging 'b' samples from a noisy function
#' objective <- ObjectiveRFun$new(
#'   fun = function(xs) {
#'     z <- exp(-xs$x^2 - xs$y^2) + 2 * exp(-(2 - xs$x)^2 - (2 - xs$y)^2)
#'     z <- z + rnorm(1, sd = 1 / sqrt(xs$b))
#'     list(Obj = z)
#'   },
#'   domain = ps(x = p_dbl(-2, 4), y = p_dbl(-2, 4), b = p_int(1)),
#'   codomain = ps(Obj = p_dbl(tags = "maximize"))
#' )
#'
#' search_space = objective$domain$search_space(list(
#'   x = to_tune(),
#'   y = to_tune(),
#'   b = to_tune(p_int(1, 2^10, tags = "budget"))
#' ))
#'
#' # Get a new OptimInstance. Here we determine that the optimizatoin goes
#' # for 10 full budget evaluations (10 * 2^100)
#' oi <- OptimInstanceSingleCrit$new(objective,
#'   search_space = search_space,
#'   terminator = trm("budget", budget = 10 * 2^10)
#' )
#'
#' # smashy is designed with parallel evaluation in mind. It is
#' # recommended to run
#' # > future::plan("multisession") (windows, or in RStudio)
#' # or
#' # > future::plan("multicore") (Linux without RStudio)
#' # and then set `mu` to a multiple of the number of CPU cores.
#'
#' smashy = configure_smashy(search_space, budget_log_step = log(4), mu = 6)
#'
#' smashy$optimize(oi)
#'
#' @export 
configure_smashy <- function(search_space, budget_log_step, mu,
    survival_fraction = 1 / 3, sample = "bohb", batch_method = "smashy",
    filter_algorithm = "tournament", surrogate_learner = "knn1", filter_with_max_budget = TRUE,
    filter_factor_first = 100, filter_factor_last = filter_factor_first, filter_select_per_tournament = 1, random_interleave_fraction = 1 / 3,
    filter_factor_first.end = filter_factor_first, filter_factor_last.end = filter_factor_last,
    filter_select_per_tournament.end = filter_select_per_tournament, random_interleave_fraction.end = random_interleave_fraction,
    random_interleave_random = TRUE,
    budget_is_logscale = FALSE,
    type = "Optimizer") {


  # Objective Parameters
  assert_r6(search_space, "ParamSet")

  # HB Parameters
  assert_number(budget_log_step, lower = 0)
  assert_int(mu, lower = 2)
  assert_number(survival_fraction, lower = 0, upper = 1)
  assert_choice(sample, c("random", "bohb"))
  assert_choice(batch_method, c("smashy", "hb"))

  learnerlist = prepare_learnerlist()
  
  
  # Surrogate Options
  assert_choice(filter_algorithm, c("tournament", "progressive"))  # The two implemented filter algorithms
  assert_choice(surrogate_learner, names(learnerlist))


  # Whether to use surrogate predictions at the largest budget so far evaluated, or at the budget of the last evaluated budget.
  # (This only makes a difference after HB "restarts", i.e. when max-budget configs were already evaluated and HB samples new low-budget individuals.)
  assert_flag(filter_with_max_budget)
  # How big is the pool from which the first individual / of the last individual is sampled from? (Relative to select_per_tournament)
  assert_number(filter_factor_first, lower = 1)
  assert_number(filter_factor_first.end, lower = 1)
  assert_number(filter_factor_last, lower = 1)
  assert_number(filter_factor_last.end, lower = 1)
  assert_int(filter_select_per_tournament, lower = 1)  # tournament size, only really used if `filter_algorithm` is "tournament"
  assert_int(filter_select_per_tournament.end, lower = 1)

  assert_number(random_interleave_fraction, lower = 0, upper = 1)  # fraction of individuals sampled with random interleaving
  assert_number(random_interleave_fraction.end, lower = 0, upper = 1)  # fraction of individuals sampled with random interleaving
  assert_flag(random_interleave_random)  # whether the number of random interleaved individuals is drawn from a binomial distribution, or the same each generation

  assert_flag(budget_is_logscale)
  assert_choice(type, c("Optimizer", "Tuner"))

  budget_param = search_space$ids(tags = "budget")
  if (!search_space$params[[budget_param]]$is_number) stop("parameter of search_space tagged as 'budget' must be numeric (ParamInt or ParamDbl)")
  if (!budget_is_logscale) {
    # budget needs to be log-scale internally, so we re-write the search space here
    prevtrafo <- search_space$trafo %??% function(x, ...) x
    if (search_space$params[[budget_param]]$class == "ParamInt") {
      # ParamInt is a bit more complicated: need to make it Dbl towards the outside, and trafo must do rounding
      budgetupper <- search_space$params[[budget_param]]$upper
      search_space$.__enclos_env__$private$.params[[budget_param]] <- ParamDbl$new(budget_param,
        lower = log(search_space$params[[budget_param]]$lower),
        upper = log(search_space$params[[budget_param]]$upper + 1),
        tags = "budget"
      )
      search_space$trafo <- mlr3misc::crate(function(x, param_set) {
        x <- prevtrafo(x, param_set)
        x[[budget_param]] <- min(floor(exp(x[[budget_param]])), budgetupper)
        x
      }, prevtrafo, budget_param, budgetupper)
    } else {
      search_space$params[[budget_param]]$lower = log(search_space$params[[budget_param]]$lower)
      search_space$params[[budget_param]]$upper = log(search_space$params[[budget_param]]$upper)
      search_space$trafo <- mlr3misc::crate(function(x, param_set) {
        x <- prevtrafo(x)
        x[[budget_param]] <- exp(x[[budget_param]])
        x
      }, prevtrafo, budget_param)

    }
  }

  # We change the lower limit of the budget parameter:
  # suppose: budget_step is 2, budget param goes from 1 to 6
  # we want steps of length 2, and highest step should be 6, so we want to eval with 6, 4, 2
  # --> there are 2 budget_steps. lower bound needs to be adjusted to 6 - 2 (# of budget steps) * 2 (budget step size) --> 2

  fidelity_steps = floor((search_space$upper[budget_param] - search_space$lower[budget_param]) / budget_log_step)
  search_space$params[[budget_param]]$lower = search_space$upper[budget_param] - fidelity_steps * budget_log_step

  learnerlist$bohblrn$graph$pipeops$stratify$param_set$values$stratify_feature = budget_param
  surrogate_learner = learnerlist[[surrogate_learner]]

  survivors = max(round(survival_fraction * mu), 1)
  lambda = mu - survivors
  if (lambda < 1) {
    # return("infeasible: no new samples per generation")
    survival_fraction <- 1 - 1 / mu
  }


  # scalor: scalarizes multi-objective results. "one": take the single objective.
  scalor = scl("one")

  # selector: take the best, according to scalarized objective
  selector = sel("best", scalor)

  # filtor: use surtour or surprog, depending on filter_algorithm config argument
  filtor = switch(filter_algorithm,
    tournament = ftr("surtour", surrogate_learner = surrogate_learner, surrogate_selector = selector,
      filter.per_tournament = interpolate_cpv(filter_select_per_tournament, filter_select_per_tournament.end, logscale = TRUE, round = TRUE),
      filter.tournament_size = interpolate_cpv(
        filter_factor_first * filter_select_per_tournament,
        filter_factor_first.end * filter_select_per_tournament.end,
        logscale = TRUE
      ),
      filter.tournament_size_last = interpolate_cpv(
        filter_factor_last * filter_select_per_tournament,
        filter_factor_last.end * filter_select_per_tournament.end,
        logscale = TRUE
      )
    ),
    progressive = ftr("surprog", surrogate_learner = surrogate_learner, surrogate_selector = selector,
      filter.pool_factor = interpolate_cpv(filter_factor_first, filter_factor_first.end, logscale = TRUE),
      filter.pool_factor_last = interpolate_cpv(filter_factor_last, filter_factor_last.end, logscale = TRUE)
    )
  )

  random_interleave_fraction_cpv  = interpolate_cpv(1 - random_interleave_fraction, 1 - random_interleave_fraction.end)  # linear scale

  interleaving_filtor = ftr("maybe", filtor, p = random_interleave_fraction_cpv, random_choice = random_interleave_random)

  sampling_fun = switch(sample, random = paradox::generate_design_random, bohb = generate_design_bohb)

  if (type == "Optimizer") {
    result = bbotk::opt("smashy", filtor = interleaving_filtor, selector = selector,
      mu = mu, survival_fraction = survival_fraction,
      fidelity_steps = fidelity_steps + 1, synchronize_batches = batch_method == "smashy",
      filter_with_max_budget = filter_with_max_budget
    )
    optimizer = result
  } else {
    result = mlr3tuning::tnr("smashy", filtor = interleaving_filtor, selector = selector,
      mu = mu, survival_fraction = survival_fraction,
      fidelity_steps = fidelity_steps + 1, synchronize_batches = batch_method == "smashy",
      filter_with_max_budget = filter_with_max_budget
    )
    optimizer = result$.__enclos_env__$private$.optimizer
  }

  optimizer$.__enclos_env__$private$.own_param_set$context_available = "inst"
  optimizer$param_set$values$sampling = sampling_fun

  result
}

#' @title Emulate the Hyperband optimizer.
#'
#' @description
#' Create a [`OptimizerSmashy`] (or [`TunerSmashy`]) object which behaves as
#' the Hyperband optimizer presented by Li et al. (2018).
#'
#' @param search_space ([`ParamSet`][paradox::ParamSet])\cr
#'   search space, has one parameter tagged 'budget'.
#' @param eta (`numeric(1)`)\cr
#'   Eta-parameter of Hyperband:
#'   Factor of budget increase and at the same time, one over fraction of configurations that survive, for
#'   each batch evaluation. Default 3.
#' @param budget_is_logscale (`logical(1)`)\cr
#'   Whether the component of `search_space` tagged as `"budget"` is in log-scale (`TRUE`) or not (`FALSE`).
#'   When `budget_is_logscale` is given as `TRUE`, then evaluations are done with linear steps of `budget_log_step`.
#'   Otherwise, exponential steps of `exp(budget_log_step)` are performed. Default `FALSE`.
#' @param type (`character(1)`)\cr
#'   One of `"Optimizer"` or `"Tuner"`. What class to return: An [`Optimizer`][bbotk::Optimizer] for
#'   optimizing [`OptimInstanceSingleCrit`][bbotk::OptimInstanceSingleCrit] objects, or a
#'   [`Tuner`][mlr3tuning::Tuner] for tuning [`mlr3::Learner`]s. Defaults to `"Optimizer"`.
#' @return [`OptimizerSmashy`] or [`TunerSmashy`], depending on the `type` argument: The configured *Smashy* optimizer.
#' @examples
#' library("bbotk")
#'
#' # Define the objective to optimize
#' # The 'budget' here simulates averaging 'b' samples from a noisy function
#' objective <- ObjectiveRFun$new(
#'   fun = function(xs) {
#'     z <- exp(-xs$x^2 - xs$y^2) + 2 * exp(-(2 - xs$x)^2 - (2 - xs$y)^2)
#'     z <- z + rnorm(1, sd = 1 / sqrt(xs$b))
#'     list(Obj = z)
#'   },
#'   domain = ps(x = p_dbl(-2, 4), y = p_dbl(-2, 4), b = p_int(1)),
#'   codomain = ps(Obj = p_dbl(tags = "maximize"))
#' )
#'
#' search_space = objective$domain$search_space(list(
#'   x = to_tune(),
#'   y = to_tune(),
#'   b = to_tune(p_int(1, 2^10, tags = "budget"))
#' ))
#'
#' # Get a new OptimInstance. Here we determine that the optimizatoin goes
#' # for 10 full budget evaluations (10 * 2^100)
#' oi <- OptimInstanceSingleCrit$new(objective,
#'   search_space = search_space,
#'   terminator = trm("budget", budget = 10 * 2^10)
#' )
#'
#' hb = smashy_as_hyperband(search_space)
#'
#' hb$optimize(oi)
#'
#' @export 
smashy_as_hyperband <- function(search_space, eta = 3, budget_is_logscale = FALSE, type = "Optimizer") {
  budget_param = search_space$ids(tags = "budget")
  trafo = if (budget_is_logscale) identity else log
  fidelity_steps = floor((trafo(search_space$upper[budget_param]) - trafo(search_space$lower[budget_param])) / log(eta))

  configure_smashy(search_space,
    budget_log_step = log(eta),
    survival_fraction = 1 / eta,
    mu = eta ^ fidelity_steps,
    sample = "random",
    batch_method = "hb",
    random_interleave_fraction = 1,
    budget_is_logscale = budget_is_logscale,
    type = type,
   ## - mandatory arguments that don't have an effect with interleave fraction == 1
    filter_algorithm = "tournament",  # doesn't matter
    surrogate_learner = "ranger",  # doesn't matter
    filter_with_max_budget = FALSE,  # doesn't matter
    filter_factor_first = 1,  # doesn't matter
    random_interleave_random = FALSE  # doesn't matter
  )
}

#' @title Emulate the BOHB optimizer.
#'
#' @description
#' Create a [`OptimizerSmashy`] (or [`TunerSmashy`]) object which behaves as
#' the BOHB optimizer presented by Falkner et al. (2018).
#'
#' @param search_space ([`ParamSet`][paradox::ParamSet])\cr
#'   search space, has one parameter tagged 'budget'.
#' @param eta (`numeric(1)`)\cr
#'   Eta-parameter of BOHB:
#'   Factor of budget increase and at the same time, one over fraction of configurations that survive, for
#'   each batch evaluation. Default 3.
#' @param rho (`numeric(1)`)\cr
#'   Rho-parameter of BOHB:
#'   fraction of configurations sampled randomly without the aid of the surrogate model.\cr
#'   Note that this implementation differs from the implementation by Falkner et al. (2018),
#'   since the kernel density sampler of BOHB is used even for the randomly interleaved configurations
#'   here, whereas Falkner et al. (2018) samples these points uniformly at random.
#' @param ns (`integer(1)`)\cr
#'   Surrogate random search rate: How many randomly sampled configurations to evaluate on the surrogate model
#'   to choose a single configuration to evaluate on the true objective function.
#' @param budget_is_logscale (`logical(1)`)\cr
#'   Whether the component of `search_space` tagged as `"budget"` is in log-scale (`TRUE`) or not (`FALSE`).
#'   When `budget_is_logscale` is given as `TRUE`, then evaluations are done with linear steps of `budget_log_step`.
#'   Otherwise, exponential steps of `exp(budget_log_step)` are performed. Default `FALSE`.
#' @param type (`character(1)`)\cr
#'   One of `"Optimizer"` or `"Tuner"`. What class to return: An [`Optimizer`][bbotk::Optimizer] for
#'   optimizing [`OptimInstanceSingleCrit`][bbotk::OptimInstanceSingleCrit] objects, or a
#'   [`Tuner`][mlr3tuning::Tuner] for tuning [`mlr3::Learner`]s. Defaults to `"Optimizer"`.
#' @return [`OptimizerSmashy`] or [`TunerSmashy`], depending on the `type` argument: The configured *Smashy* optimizer.
#' @examples
#' library("bbotk")
#'
#' # Define the objective to optimize
#' # The 'budget' here simulates averaging 'b' samples from a noisy function
#' objective <- ObjectiveRFun$new(
#'   fun = function(xs) {
#'     z <- exp(-xs$x^2 - xs$y^2) + 2 * exp(-(2 - xs$x)^2 - (2 - xs$y)^2)
#'     z <- z + rnorm(1, sd = 1 / sqrt(xs$b))
#'     list(Obj = z)
#'   },
#'   domain = ps(x = p_dbl(-2, 4), y = p_dbl(-2, 4), b = p_int(1)),
#'   codomain = ps(Obj = p_dbl(tags = "maximize"))
#' )
#'
#' search_space = objective$domain$search_space(list(
#'   x = to_tune(),
#'   y = to_tune(),
#'   b = to_tune(p_int(1, 2^10, tags = "budget"))
#' ))
#'
#' # Get a new OptimInstance. Here we determine that the optimizatoin goes
#' # for 10 full budget evaluations (10 * 2^100)
#' oi <- OptimInstanceSingleCrit$new(objective,
#'   search_space = search_space,
#'   terminator = trm("budget", budget = 10 * 2^10)
#' )
#'
#' bohb = smashy_as_bohb(search_space)
#'
#' bohb$optimize(oi)
#'
#' @export 
smashy_as_bohb <- function(search_space, eta = 3, rho = 1 / 3, ns = 64, budget_is_logscale = FALSE, type = "Optimizer") {

  budget_param = search_space$ids(tags = "budget")
  trafo = if (budget_is_logscale) identity else log
  fidelity_steps = floor((trafo(search_space$upper[budget_param]) - trafo(search_space$lower[budget_param])) / log(eta))

  configure_smashy(
    budget_log_step = log(eta),
    survival_fraction = 1 / eta,
    mu = eta ^ fidelity_steps,
    sample = "bohb",
    batch_method = "hb",
    random_interleave_fraction = rho,
    filter_algorithm = "tournament",
    surrogate_learner = "bohblrn",
    filter_with_max_budget = TRUE,
    filter_factor_first = ns,
    random_interleave_random = TRUE,
    budget_is_logscale = budget_is_logscale,
    type = type
  )
}
