module RegressionTools

using Distances: euclidean
using StatsFuns: logistic, softplus, logit
using PLINK

import Base.vecnorm

export threshold!
export threshold
export update_residuals!
export update_indices!
export update_col!
export update_weights!
export update_xk!
export fill_perm!
export update_xb!
export count_partialnz
export fill_partial!
export update_partial_residuals!
export difference!
export ypatzmw!
export project_k!
export selectpermk!
export selectpermk
export mce
export logistic_loglik 
export update_x!
export df_norm
export logistic_grad!
export logistic!
export log2xb!
export fit_logistic
export mask!
export cv_get_folds
export vecnorm
export issymmetric
export vec!

const Float = Union{Float32, Float64}

depwarn("NOTE: As of Julia v0.6, this package is deprecated and no longer maintained.\n")

# do not load any functions; deprecated package will now be empty
#include("regtool.jl")
#include("logistic.jl")

end # end module RegressionTools
