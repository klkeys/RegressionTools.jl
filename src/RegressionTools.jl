module RegressionTools

using Distances: euclidean
using StatsBase: logistic, softplus, logit

export threshold!
export update_residuals! 
export update_indices!
export update_col!
export update_weights!
export update_xk!
export fill_perm!
export compute_loglik
export update_xb!
export count_partialnz
export fill_partial!
export mce
export mse
export loggrad!
export logistic!
export update_y2!
export log2xb!
export fit_logistic
export selectperm!
export selectperm
export update_partial_residuals!
export difference! 
export ypatzmw! 

include("regtool32.jl")
include("regtool64.jl")
include("logistic.jl")

end	# end module RegressionTools
