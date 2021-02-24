#' @title Gaussian Distribution Mutator
#'
#' @include Mutator.R
#'
#' @name dict_mutators_gauss
#'
#' @description
#' Individuals are mutated with an independent normal random variable on each component.
#'
#' @section Hyperparameters:
#' * `sdev` :: `numeric`\cr
#'   Standard deviation of normal distribuion. This is absolute if `sdev_is_relative` is `FALSE`, and
#'   multiplied with each individual component's range (upper - lower) if `sdev_is_relative` is `TRUE`.
#'   This may either be a scalar, in which case it is applied to all input components, or a vector,
#'   in which case it must have the length of the input and applies to components in order in which
#'   they appear in the priming [`ParamSet`][paradox::ParamSet]. Initialized to 1.
#' * `sdev_is_relative` :: `logical(1)`\cr
#'   Whether `sdev` is absolute (`FALSE`) or relative to component range (`TRUE`). Initialized to `FALSE`.
#' * `truncated_normal` :: `logical(1)`\cr
#'   Whether to draw individuals from a normal distribution that is truncated at the bounds of each
#'   component (`TRUE`), or to draw from a normal distribution and then restrict to bounds afterwards
#'   (`FALSE`). The former (`TRUE`) will lead to very few to no samples landing on the exact bounds
#'   (analytically it would be none almost surely, but this is subject to machine precision), the latter
#'   (`FALSE`) can lead to a substantial number of samples landing on the exact bounds. Initialized to `FALSE`.
#'
#' @templateVar id gauss
#' @template autoinfo_prepare_mut
#' @template autoinfo_operands
#' @template autoinfo_dict
#'
#' @family mutators
#' @export
MutatorGauss = R6Class("MutatorGauss",
  inherit = MutatorNumeric,
  public = list(
    #' @description
    #' Initialize the `MutatorGauss` object.
    initialize = function() {
      param_set = ps(sdev = p_uty(custom_check = crate(function(x) check_numeric(x,
        lower = tol_bound(0, "lower"), any.missing = FALSE, min.len = 1), .parent = topenv()), tags = "required"),
        sdev_is_relative = p_lgl(tags = "required"), truncated_normal = p_lgl(tags = "required"))
      param_set$values = list(sdev = 1, sdev_is_relative = FALSE, truncated_normal = FALSE)
      super$initialize("ParamDbl", param_set)
    }
  ),
  private = list(
    .mutate_numeric = function(values, lowers, uppers) {
      params = self$param_set$get_values()
      sdev = params$sdev
      sdev = pmax(sdev, 0)
      if (length(sdev) == 1) {
        sdev = rep(sdev, length(values))  # make the ifelse() further down work
      }
      if (length(sdev) != length(values)) {
        stop("sdev must have either length 1, or length of input.")
      }
      if (params$sdev_is_relative) {
        assert_numeric(lowers, finite = TRUE, any.missing = FALSE)
        assert_numeric(uppers, finite = TRUE, any.missing = FALSE)
        sdev = sdev * (uppers - lowers)
      }
      if (params$truncated_normal) {
        mutated <- ifelse(sdev == 0, values,
          qnorm(runif(length(values),
            pnorm(lowers, values, sdev),
            pnorm(uppers, values, sdev)),
            values, sdev)
        )
      } else {
        mutated <- rnorm(length(values), values, sdev)
      }
      pmax(pmin(mutated, uppers), lowers)
    }
  )
)
dict_mutators$add("gauss", MutatorGauss)