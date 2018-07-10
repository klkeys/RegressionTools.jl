# RegressionTools.jl

***NOTE: As of Julia v0.6, this package is deprecated and no longer maintained.***

A Julia package with various numerical subroutines to facilitate calculations in regression and optimization.
RegressionTools.jl optimizes for speed and low memory profiles instead of readability or wide applicability.
It only works with `Float32` and `Float64` arithmetic.
Where possible, routines use `@inbounds` and `@simd`. 
Many of the routines are simply hand-coded loops.
These loops are not necessarily faster than their vectorized counterparts,
though they often scale better to large dimensions.
