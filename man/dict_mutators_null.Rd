% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/MutatorNull.R
\name{dict_mutators_null}
\alias{dict_mutators_null}
\alias{MutatorNull}
\title{Null-Mutator}
\description{
Null-mutator that does not perform any operation on its input. Useful in particular with operator-wrappers such as \code{\link{MutatorMaybe}} or \code{\link{MutatorCombination}}.
}
\section{Configuration Parameters}{

This operator has no configuration parameters.
}

\section{Supported Operand Types}{


Supported \code{\link[paradox:Param]{Param}} classes are: \code{\link[paradox:ParamLgl]{ParamLgl}}, \code{\link[paradox:ParamInt]{ParamInt}}, \code{\link[paradox:ParamDbl]{ParamDbl}}, \code{\link[paradox:ParamFct]{ParamFct}}
}

\section{Dictionary}{

This \code{\link{Mutator}} can be created with the short access form \code{\link[=mut]{mut()}}
(\code{\link[=muts]{muts()}} to get a list), or through the the \link[mlr3misc:Dictionary]{dictionary}
\code{\link{dict_mutators}} in the following way:

\if{html}{\out{<div class="sourceCode">}}\preformatted{# preferred:
mut("null")
muts("null")  # takes vector IDs, returns list of Mutators

# long form:
dict_mutators$get("null")
}\if{html}{\out{</div>}}
}

\examples{
mn = mut("null")
p = ps(x = p_int(-5, 5), y = p_dbl(-5, 5), z = p_lgl())
data = data.frame(x = rep(0, 5), y = rep(0, 5), z = rep(TRUE, 5))

mn$prime(p)
mn$operate(data)
}
\seealso{
Other mutators: 
\code{\link{MutatorDiscrete}},
\code{\link{MutatorNumeric}},
\code{\link{Mutator}},
\code{\link{OperatorCombination}},
\code{\link{dict_mutators_cmpmaybe}},
\code{\link{dict_mutators_erase}},
\code{\link{dict_mutators_gauss}},
\code{\link{dict_mutators_maybe}},
\code{\link{dict_mutators_proxy}},
\code{\link{dict_mutators_sequential}},
\code{\link{dict_mutators_unif}}
}
\concept{mutators}
\section{Super classes}{
\code{smashy::MiesOperator} -> \code{smashy::Mutator} -> \code{MutatorNull}
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-MutatorNull-new}{\code{MutatorNull$new()}}
\item \href{#method-MutatorNull-clone}{\code{MutatorNull$clone()}}
}
}
\if{html}{\out{
<details open><summary>Inherited methods</summary>
<ul>
<li><span class="pkg-link" data-pkg="smashy" data-topic="MiesOperator" data-id="operate"><a href='../../smashy/html/MiesOperator.html#method-MiesOperator-operate'><code>smashy::MiesOperator$operate()</code></a></span></li>
<li><span class="pkg-link" data-pkg="smashy" data-topic="MiesOperator" data-id="prime"><a href='../../smashy/html/MiesOperator.html#method-MiesOperator-prime'><code>smashy::MiesOperator$prime()</code></a></span></li>
<li><span class="pkg-link" data-pkg="smashy" data-topic="MiesOperator" data-id="print"><a href='../../smashy/html/MiesOperator.html#method-MiesOperator-print'><code>smashy::MiesOperator$print()</code></a></span></li>
<li><span class="pkg-link" data-pkg="smashy" data-topic="MiesOperator" data-id="repr"><a href='../../smashy/html/MiesOperator.html#method-MiesOperator-repr'><code>smashy::MiesOperator$repr()</code></a></span></li>
</ul>
</details>
}}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-MutatorNull-new"></a>}}
\if{latex}{\out{\hypertarget{method-MutatorNull-new}{}}}
\subsection{Method \code{new()}}{
Initialize the \code{MutatorNull} object.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{MutatorNull$new()}\if{html}{\out{</div>}}
}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-MutatorNull-clone"></a>}}
\if{latex}{\out{\hypertarget{method-MutatorNull-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{MutatorNull$clone(deep = FALSE)}\if{html}{\out{</div>}}
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
