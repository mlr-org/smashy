% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/FiltorNull.R
\name{dict_filtors_null}
\alias{dict_filtors_null}
\alias{FiltorNull}
\title{Null-Filtor}
\description{
Null-filtor that does not perform filtering. Its \code{needed_input()} is always  the \code{output_size}, and \code{operate()} selects the first \code{n_filter} values from its input.

Useful in particular with operator-wrappers such as \code{\link{FiltorProxy}}, and to make filtering optional.
}
\section{Configuration Parameters}{

This operator has no configuration parameters.
}

\section{Supported Operand Types}{


Supported \code{\link[paradox:Param]{Param}} classes are: \code{\link[paradox:ParamLgl]{ParamLgl}}, \code{\link[paradox:ParamInt]{ParamInt}}, \code{\link[paradox:ParamDbl]{ParamDbl}}, \code{\link[paradox:ParamFct]{ParamFct}}
}

\section{Dictionary}{

This \code{\link{Filtor}} can be created with the short access form \code{\link[=ftr]{ftr()}}
(\code{\link[=ftrs]{ftrs()}} to get a list), or through the the \link[mlr3misc:Dictionary]{dictionary}
\code{\link{dict_filtors}} in the following way:

\if{html}{\out{<div class="sourceCode">}}\preformatted{# preferred:
ftr("null")
ftrs("null")  # takes vector IDs, returns list of Filtors

# long form:
dict_filtors$get("null")
}\if{html}{\out{</div>}}
}

\examples{
fn = ftr("null")

p = ps(x = p_dbl(-5, 5))
known_data = data.frame(x = 1:5)
fitnesses = 1:5

new_data = data.frame(x = c(2.5, 4.5))

fn$prime(p)

fn$needed_input(1)

fn$operate(new_data, known_data, fitnesses, 1)

}
\seealso{
Other filtors: 
\code{\link{FiltorSurrogate}},
\code{\link{Filtor}},
\code{\link{dict_filtor_maybe}},
\code{\link{dict_filtors_proxy}},
\code{\link{dict_filtors_surprog}},
\code{\link{dict_filtors_surtour}}
}
\concept{filtors}
\section{Super classes}{
\code{smashy::MiesOperator} -> \code{smashy::Filtor} -> \code{FiltorNull}
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-FiltorNull-new}{\code{FiltorNull$new()}}
\item \href{#method-FiltorNull-clone}{\code{FiltorNull$clone()}}
}
}
\if{html}{\out{
<details open><summary>Inherited methods</summary>
<ul>
<li><span class="pkg-link" data-pkg="smashy" data-topic="MiesOperator" data-id="operate"><a href='../../smashy/html/MiesOperator.html#method-MiesOperator-operate'><code>smashy::MiesOperator$operate()</code></a></span></li>
<li><span class="pkg-link" data-pkg="smashy" data-topic="MiesOperator" data-id="prime"><a href='../../smashy/html/MiesOperator.html#method-MiesOperator-prime'><code>smashy::MiesOperator$prime()</code></a></span></li>
<li><span class="pkg-link" data-pkg="smashy" data-topic="MiesOperator" data-id="print"><a href='../../smashy/html/MiesOperator.html#method-MiesOperator-print'><code>smashy::MiesOperator$print()</code></a></span></li>
<li><span class="pkg-link" data-pkg="smashy" data-topic="MiesOperator" data-id="repr"><a href='../../smashy/html/MiesOperator.html#method-MiesOperator-repr'><code>smashy::MiesOperator$repr()</code></a></span></li>
<li><span class="pkg-link" data-pkg="smashy" data-topic="Filtor" data-id="needed_input"><a href='../../smashy/html/Filtor.html#method-Filtor-needed_input'><code>smashy::Filtor$needed_input()</code></a></span></li>
</ul>
</details>
}}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-FiltorNull-new"></a>}}
\if{latex}{\out{\hypertarget{method-FiltorNull-new}{}}}
\subsection{Method \code{new()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{FiltorNull$new()}\if{html}{\out{</div>}}
}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-FiltorNull-clone"></a>}}
\if{latex}{\out{\hypertarget{method-FiltorNull-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{FiltorNull$clone(deep = FALSE)}\if{html}{\out{</div>}}
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
