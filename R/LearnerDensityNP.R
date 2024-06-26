#' @title Kernel Density Estimator using the "np" Package
#'
#' @name mlr_learners_density.np
#'
#' @description
#' Perform kernel density estimation using the `np` package.
#'
#' @section Hyperparameters:
#' * `bwmethod` :: `character(1)`\cr
#'   Bandwidth selection method. One of `"cv.ml"`, `"cv.ls"`, `"normal-reference"`, `"normal-reference-numeric"`. Default `"cv.ml"`.
#'   `"normal-reference-numeric"` emulates (buggy, see `https://github.com/statsmodels/statsmodels/issues/3790`) behaviour of python statsmodels:
#'   non-numeric columns are treated as numeric for bandwidth estimation.
#' * `bwtype` :: `character(1)`\cr
#'   Continuous variable bandwidth type, one of `"fixed"` (default), `"generalized_nn"`, `"adaptive_nn"`.
#' * `ckertype` :: `character(1)`\cr
#'   Continuous kernel type. One of `"gaussian"` (default), `"epanechnikov"`, `"uniform"`.
#' * `ckerorder` :: `integer(1)`\cr
#'   Continuous Kernel order (only for gaussian and epanechnikov). One of 2 (default), 4, 6, 8.
#' * `ukertype` :: `character(1)`\cr
#'   Unordered categorical kernel type. One of `"aitchisonaitken"` (default) or `"liracine"`.
#' * `okertype` :: `character(1)`\cr
#'   Ordered kernel type. One of `"liracine"` (default), `"wangvanryzin"`.
#' * `nmulti` :: `integer(1)`\cr
#'   Number of random restarts for cross validation likelihood optimization. Default: number of features, at most 5. 0 disables restarts.
#' * `remin` :: `logical(1)`\cr
#'   Restart from located minima; Default `TRUE`.
#' * `itmax` :: `integer(1)`\cr
#'   Max optimization iters. Default 10000.
#' * `ftol` :: `numeric(1)`\cr
#'   Optimization y-value relative tolerance, default `10 * sqrt(.Machine$double.eps)`.
#' * `tol` :: `numeric(1)`\cr
#'   Optimization x-value relative tolerance, default `10000 * sqrt(.Machine$double.eps)`.
#' * `small` :: `numeric(1)`\cr
#'   Optimization x-value absolute tolerance (?), default `1000 * sqrt(.Machine$double.eps)`.
#' * `min_bandwidth` :: `numeric(1)`\cr
#'   Minimum bandwidth (all kerneltypes). Not part of np::npudensbw. Default 0. BOHB has this at `1e-3`.
#' * `sampling_bw_factor` :: `numeric(1)`\cr
#'   Oversmoothing bandwidth factor for sampling. Not part of np::npudensbw. Default 1. BOHB has this at 3.
#'
#' TODO: could also implement `lb{c,d}.{dir,init}`, `{c,d}fac.{dir,init}`, `dfc.dir`, `hbd.dir`, `hb{c,d}.init`, `init{c,d}.dir`, `scale.init.categorical.sample`.
#'
#' TODO: `normal-reference` seems to disable optimization, consider putting that as a dependency on the optimization-related params.
#'
#' TODO: could add dependency for ckerorder on not-uniform.
#'
#' @export
#' @family density estimation classes
LearnerDensityNP = R6Class("LearnerDensityNP", inherit = LearnerDensity,
  public = list(
    #' @description
    #' Initialize the `LearnerDensityNP` object.
    initialize = function() {
      param_set = ps(
        bwmethod = p_fct(c("cv.ml", "cv.ls", "normal-reference", "normal-reference-numeric"), default = "cv.ml", tags = c("train", "npudensbw")),
        bwtype = p_fct(c("fixed", "generalized_nn", "adaptive_nn"), default = "fixed", tags = c("train", "npudensbw")),
        ckertype = p_fct(c("gaussian", "epanechnikov", "uniform"), default = "gaussian", tags = c("train", "npudensbw")),
        ckerorder = p_int(2, 8, default = 2, tags = c("train", "npudensbw")),
        ukertype = p_fct(c("aitchisonaitken", "liracine"), default = "aitchisonaitken", tags = c("train", "npudensbw")),
        okertype = p_fct(c("liracine", "wangvanryzin"), default = "liracine", tags = c("train", "npudensbw")),
        nmulti = p_int(0, tags = c("train", "npudensbw")),
        remin = p_lgl(default = TRUE, tags = c("train", "npudensbw")),
        itmax = p_int(0, default = 10000, tags = c("train", "npudensbw")),
        ftol = p_dbl(0, default = 1e1 * sqrt(.Machine$double.eps), tags = c("train", "npudensbw")),
        tol = p_dbl(0, default = 1e4 * sqrt(.Machine$double.eps), tags = c("train", "npudensbw")),
        small = p_dbl(0, default = 1e3 * sqrt(.Machine$double.eps), tags = c("train", "npudensbw")),
        min_bandwidth = p_dbl(0, tags = "train", default = 0),  # not part of the package.
        sampling_bw_factor = p_dbl(0, tags = "predict")  # oversmoothing bandwidth factor for sampling
      )
      super$initialize(
        id = "density.np",
        feature_types = c("numeric", "factor", "ordered"),
        predict_types = "prob",
        packages = "np",
        param_set = param_set,
        properties = "sample",
        man = "smashy::mlr_learners_density.np"
      )
    }
  ),
  private = list(
    .train = function(task) {
      pv = self$param_set$get_values(tags = "train")
      pv$min_bandwidth <- pv$min_bandwidth %??% 0
      dat = task$data()

      jitter <- function(d) {
        # TODO: hack: do something with constant values, npudensbw can't handle them otherwise.
        for (col in seq_along(d)) {
          if (is.numeric(d[[col]]) && diff(range(d[[col]])) == 0) d[[col]][[1]] = d[[col]][[1]] * (1 - 2 * .Machine$double.eps) + .Machine$double.eps
        }
        d
      }
      np::npseed(as.integer(runif(1, 0, 161803398)))  # https://github.com/JeffreyRacine/R-Package-np/issues/47
      args = self$param_set$get_values(tags = "npudensbw")
      numericize = identical(args$bwmethod, "normal-reference-numeric")
      if (numericize) {
        args$bwmethod = "normal-reference"
        bw_numeric = invoke(np::npudensbw, dat = jitter(dat[, lapply(.SD, as.numeric)]), .args = args)
      }
      bw = invoke(np::npudensbw, dat = jitter(dat), .args = args)
      if (numericize) {
        bw$bw = bw_numeric$bw
        bw$bandwidth$x = bw_numeric$bandwidth$x
        # using the numeric kernels breaks everything when bandwidths are larger than 1 (and don't make much sense, besides).
        # instead, we get the probability that a normal rv with given bandwidth is not in the interval -0.5...0.5.
        rebandwidth = !sapply(dat, is.numeric)
        bw$bw[rebandwidth] = 2 - 2 * pnorm(0.5, sd = bw$bw[rebandwidth])
        bw$bandwidth$x[rebandwidth] = 2 - 2 * pnorm(0.5, sd = bw$bandwidth$x[rebandwidth])
      }

      bw$call = NULL

      bw$bw[bw$bw < pv$min_bandwidth] <- pv$min_bandwidth
      bw$bandwidth$x[bw$bandwidth$x < pv$min_bandwidth] <- pv$min_bandwidth

      list(bw = bw, dat = dat)
    },
    .predict = function(task) {
      prob = stats::fitted(np::npudens(bws = self$model$bw, tdat = self$model$dat, edat = task$data(cols = colnames(self$model$dat))))
      prob[prob < .Machine$double.xmin] <- .Machine$double.xmin
      list(prob = prob)
    },
    .sample = function(n, lower, upper) {
      bw = self$model$bw
      pv = self$param_set$get_values(tags = "predict")
      bw_factor <- pv$sampling_bw_factor %??% 1

      bwenlarge <- function(bw, factor) {
        rebandwidth = !sapply(self$model$dat, is.numeric)
        bw[rebandwidth] = 0.5 / qnorm(1 - bw[rebandwidth] / 2)
        bw = bw * factor
        bw[rebandwidth] = 2 - 2 * pnorm(0.5, sd = bw[rebandwidth])
        bw
      }

      bw$bw <- bwenlarge(bw$bw, bw_factor)
      bw$bandwidth$x <- bwenlarge(bw$bandwidth$x, bw_factor)

      if (!identical(bw$ckerorder, 2)) stop("Can only sample with kernel order 2.")
      # gaussian kernel: rnorm
      # epanechnikov kernel: repanechnikov
      indices = sample.int(nrow(self$model$dat), n, replace = TRUE)
      prototypes = self$model$dat[indices]
      pfun = switch(bw$ckertype, epanechnikov = pepanechnikov, gaussian = pnorm)
      qfun = switch(bw$ckertype, epanechnikov = qepanechnikov, gaussian = qnorm)
      dt = as.data.table(Map(function(dim, dimname, bandwidth, type, lx, ux) {
        if (type == "numeric") {
          rq = runif(n, pfun(lx, dim, bandwidth), pfun(ux, dim, bandwidth))
          result = qfun(rq, dim, bandwidth)
          result[result < lx] = lx
          result[result > ux] = ux
        } else {
          result = dim
          cons = switch(type, factor = factor, ordered = ordered, stopf("Unsupported feature type %s", type))
          samplefrom = cons(levels(dim), levels = levels(dim))
          for (l in levels(dim)) {
            xlevel = which(dim == l)
            if (!length(xlevel)) next
            protorow = xlevel[[1]]
            origin = prototypes[protorow]
            rep1s = rep(1, length(samplefrom))
            exdat = origin[rep1s]
            exdat[[dimname]] = samplefrom
            sweights = np::npksum(bw, txdat = origin, exdat = exdat)$ksum
            sweights[sweights < 0] = 0  # happens when the bandwidth is set too high, e.g. through bw_factor or normal-reference-numeric
            indices = sample.int(length(sweights), length(xlevel), replace = TRUE, prob = sweights)
            result[xlevel] = samplefrom[indices]
          }
        }
        result
      }, prototypes, names(prototypes), bw$bw, self$state$train_task$col_info[colnames(prototypes)]$type, lower, upper))
      colnames(dt) = colnames(self$model$dat)
      dt
    }
  )
)


depanechnikov <- function(x, location = 0, scale = 1) {
  z = (x - location) / scale
  3 * (1 - z^2 / 5) / (4 * sqrt(5) * scale)
}

pepanechnikov <- function(q, location = 0, scale = 1) {
  z = (q - location) / scale
  inrange = abs(z) < sqrt(5)
  result = as.numeric(z > 0)
  z = z[inrange]
  result[inrange] = z * (3 - z^2 / 5) / (4 * sqrt(5)) + 0.5
  result
}

qepanechnikov <- function(p, location = 0, scale = 1) {
  z <- p * 2 - 1
  theta <- atan2(sqrt(1 - z^2), -z) / 3
  sqrt(5) * (sqrt(3) * sin(theta) - cos(theta)) * scale + location
}

repanechnikov <- function(n, location = 0, scale = 1) {
  qepanechnikov(runif(n), location = location, scale = scale)
}
