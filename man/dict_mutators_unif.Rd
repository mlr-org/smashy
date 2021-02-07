% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/Mutator.R
\name{dict_mutators_unif}
\alias{dict_mutators_unif}
\alias{MutatorDiscreteUniform}
\title{Uniform Discrete Mutator}
\description{
Discrete components are mutated by sampling from a uniform distribution, either from all possible
values of each component, or from all values except the original value.

Since the information loss is very high, this should in most cases be combined with \code{\link{MutatorCmpMaybe}}.
}
\section{Hyperparameters}{

\itemize{
\item \code{can_mutate_to_same} :: \code{logical(1)}\cr
Whether to sample from entire range of each parameter (\code{TRUE}) or from all values except the
current value (\code{FALSE}). Initialized to \code{TRUE}.
}
}

\section{Supported Operand Types}{


Supported \code{\link[paradox:Param]{Param}} classes are: \code{\link[paradox:ParamLgl]{ParamLgl}}, \code{\link[paradox:ParamFct]{ParamFct}}
}

\section{Dictionary}{

This \code{\link{Mutator}} can be created with the short access form \code{\link[=mut]{mut()}}
(\code{\link[=muts]{muts()}} to get a list), or through the the \link[mlr3misc:Dictionary]{dictionary}
\code{\link{dict_mutators}} in the following way:\preformatted{# preferred:
mut("unif")
muts("unif")  # takes vector IDs, returns list of [`Mutator`]s

# long form:
dict_mutators$get("unif")
}
}

\seealso{
Other mutators: 
\code{\link{MutatorDiscrete}},
\code{\link{MutatorNumeric}},
\code{\link{Mutator}},
\code{\link{OperatorCombination}},
\code{\link{dict_mutators_cmpmaybe}},
\code{\link{dict_mutators_gauss}},
\code{\link{dict_mutators_maybe}},
\code{\link{dict_mutators_null}}
}
\concept{mutators}
\section{Super classes}{
\code{\link[Miesmuschel:MiesOperator]{Miesmuschel::MiesOperator}} -> \code{\link[Miesmuschel:Mutator]{Miesmuschel::Mutator}} -> \code{\link[Miesmuschel:MutatorDiscrete]{Miesmuschel::MutatorDiscrete}} -> \code{MutatorDiscreteUniform}
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-new}{\code{MutatorDiscreteUniform$new()}}
\item \href{#method-clone}{\code{MutatorDiscreteUniform$clone()}}
}
}
\if{html}{
\out{<details open ><summary>Inherited methods</summary>}
\itemize{
\item \out{<span class="pkg-link" data-pkg="Miesmuschel" data-topic="MiesOperator" data-id="operate">}\href{../../Miesmuschel/html/MiesOperator.html#method-operate}{\code{Miesmuschel::MiesOperator$operate()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="Miesmuschel" data-topic="MiesOperator" data-id="prime">}\href{../../Miesmuschel/html/MiesOperator.html#method-prime}{\code{Miesmuschel::MiesOperator$prime()}}\out{</span>}
}
\out{</details>}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-new"></a>}}
\if{latex}{\out{\hypertarget{method-new}{}}}
\subsection{Method \code{new()}}{
Initialize the \code{MutatorDiscreteUniform} object.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{MutatorDiscreteUniform$new()}\if{html}{\out{</div>}}
}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-clone"></a>}}
\if{latex}{\out{\hypertarget{method-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{MutatorDiscreteUniform$clone(deep = FALSE)}\if{html}{\out{</div>}}
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