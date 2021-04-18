% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/Filtor.R
\name{Filtor}
\alias{Filtor}
\title{Filtor Base Class}
\description{
Base class representing filter operations, inheriting from \code{\link{MiesOperator}}.

A \code{\link{Filtor}} gets a table of individuals that are to be filtered, as well as a table of individuals that were already evaluated,
along with information on the latter individuals' performance values. Furthermore, the
number of individuals to return is given. The \code{Filtor} returns a vector of unique integers indicating which individuals were selected.

Filter operations are performed in ES algorithms to facilitate concentration towards individuals that likely perform well with regard to the
fitness measure, without evaluating the fitness measure, for example through a surrogate model.

Fitness values are always \emph{maximized}, both in single- and multi-criterion optimization.

Unlike most other operator types inheriting from \code{\link{MiesOperator}}, the \verb{$operate()} function has four arguments, which are passed on to \verb{$.filter()}
\itemize{
\item \code{values} :: \code{data.frame}\cr
Individuals to filter. Must pass the check of the \code{\link[paradox:ParamSet]{Param}} given in the last \verb{$prime()} call
and may not have any missing components.
\item \code{known_values} :: \code{data.frame}\cr
Individuals to use for filtering. Must pass the check of the \code{\link[paradox:ParamSet]{Param}} given in the last \verb{$prime()} call
and may not have any missing components. Note that \code{known_values} may be empty.
\item \code{fitnesses} :: \code{numeric} | \code{matrix}\cr
Fitnesses for each individual given in \code{old_values}. If this is a \code{numeric}, then its length must be equal to the number of rows in \code{values}. If
this is a \code{matrix}, if number of rows must be equal to the number of rows in \code{values}, and it must have one column when doing single-crit optimization
and one column each for each  "criterion" when doing multi-crit optimization.
\item \code{n_filter} :: \code{integer(1)}\cr
Number of individuals to select. Some \code{Filtor}s select individuals with replacement, for which this value may be greater than the number of
rows in \code{values}.
}

The return value for an operation will be a numeric vector of integer values of ength \code{n_filter} indexing the individuals that were selected. \code{Filtor}
must always return unique integers, i.e. select every individual at most once.
}
\section{Inheriting}{

\code{Filtor} is an abstract base class and should be inherited from. Inheriting classes should implement the private \verb{$.filter()}
function. The user of the object calls \verb{$operate()}, and the arguments are passed on to private \verb{$.filter()} after checking that
the operator is primed, that the \code{values} and \code{known_values} arguments conforms to the primed domain and that other values match.

The \code{private$.needed_input()} function should also be overloaded. It is a function that gets a single input, \code{output_size}, a positive integer indicating
the number of individuals that the caller desires. The function should calculate the number of \code{values} that are required to
filter down to \code{output_size}, given the current configuraiton parameter settings. The needed input should always be at least \code{output_size}.

Typically, the \verb{$initialize()} function should also be overloaded, and optionally the \verb{$prime()} function; they should call their \code{super} equivalents.
}

\seealso{
Other base classes: 
\code{\link{MiesOperator}},
\code{\link{MutatorDiscrete}},
\code{\link{MutatorNumeric}},
\code{\link{Mutator}},
\code{\link{OperatorCombination}},
\code{\link{Recombinator}},
\code{\link{Scalor}},
\code{\link{Selector}}

Other filtors: 
\code{\link{dict_filtor_maybe}},
\code{\link{dict_filtors_null}},
\code{\link{dict_filtors_proxy}},
\code{\link{dict_filtors_surprog}}
}
\concept{base classes}
\concept{filtors}
\section{Super class}{
\code{\link[miesmuschel:MiesOperator]{miesmuschel::MiesOperator}} -> \code{Filtor}
}
\section{Active bindings}{
\if{html}{\out{<div class="r6-active-bindings">}}
\describe{
\item{\code{supported}}{(\code{character})\cr
Optimization supported by this \code{Filtor}, can be \code{"single-crit"}, \code{"multi-crit"}, or both.}
}
\if{html}{\out{</div>}}
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-new}{\code{Filtor$new()}}
\item \href{#method-needed_input}{\code{Filtor$needed_input()}}
\item \href{#method-clone}{\code{Filtor$clone()}}
}
}
\if{html}{
\out{<details open ><summary>Inherited methods</summary>}
\itemize{
\item \out{<span class="pkg-link" data-pkg="miesmuschel" data-topic="MiesOperator" data-id="operate">}\href{../../miesmuschel/html/MiesOperator.html#method-operate}{\code{miesmuschel::MiesOperator$operate()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="miesmuschel" data-topic="MiesOperator" data-id="prime">}\href{../../miesmuschel/html/MiesOperator.html#method-prime}{\code{miesmuschel::MiesOperator$prime()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="miesmuschel" data-topic="MiesOperator" data-id="print">}\href{../../miesmuschel/html/MiesOperator.html#method-print}{\code{miesmuschel::MiesOperator$print()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="miesmuschel" data-topic="MiesOperator" data-id="repr">}\href{../../miesmuschel/html/MiesOperator.html#method-repr}{\code{miesmuschel::MiesOperator$repr()}}\out{</span>}
}
\out{</details>}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-new"></a>}}
\if{latex}{\out{\hypertarget{method-new}{}}}
\subsection{Method \code{new()}}{
Initialize base class components of the \code{Filtor}.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{Filtor$new(
  param_classes = c("ParamLgl", "ParamInt", "ParamDbl", "ParamFct"),
  param_set = ps(),
  supported = c("single-crit", "multi-crit")
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{param_classes}}{(\code{character})\cr
Classes of parameters that the operator can handle. May contain any of \code{"ParamLgl"}, \code{"ParamInt"}, \code{"ParamDbl"}, \code{"ParamFct"}.
Default is all of them.\cr
The \verb{$param_classes} field will reflect this value.}

\item{\code{param_set}}{(\code{\link[paradox:ParamSet]{ParamSet}} | \code{list} of \code{expression})\cr
Strategy parameters of the operator. This should be created by the subclass and given to \code{super$initialize()}.
If this is a \code{\link[paradox:ParamSet]{ParamSet}}, it is used as the \code{MiesOperator}'s \code{\link[paradox:ParamSet]{ParamSet}}
directly. Otherwise it must be a \code{list} of expressions e.g. created by \code{alist()} that evaluate to \code{\link[paradox:ParamSet]{ParamSet}}s,
possibly referencing \code{self} and \code{private}.
These \code{\link[paradox:ParamSet]{ParamSet}} are then combined using a \code{\link[paradox:ParamSetCollection]{ParamSetCollection}}.
Default is the empty \code{\link[paradox:ParamSet]{ParamSet}}.\cr
The \verb{$param_set} field will reflect this value.}

\item{\code{supported}}{(\code{character})\cr
Subset of \code{"single-crit"} and \code{"multi-crit"}, indicating wether single and / or multi-criterion optimization is supported.
Default both of them.\cr
The \verb{$supported} field will reflect this value.}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-needed_input"></a>}}
\if{latex}{\out{\hypertarget{method-needed_input}{}}}
\subsection{Method \code{needed_input()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{Filtor$needed_input(output_size)}\if{html}{\out{</div>}}
}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-clone"></a>}}
\if{latex}{\out{\hypertarget{method-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{Filtor$clone(deep = FALSE)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{deep}}{Whether to make a deep clone.}
}
\if{html}{\out{</div>}}
}
}
}