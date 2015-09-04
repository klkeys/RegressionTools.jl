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


###################
### SUBROUTINES ###
###################

# DIFFERENCE OF TWO VECTORS
#
# Compute the difference x = y - z, overwriting x.
function difference!{T <: Union(Float32, Float64)}(x::DenseArray{T,1}, y::DenseArray{T,1}, z::DenseArray{T,1}; n::Integer = length(x), a::T = 1.0, b::T = 1.0)
	@inbounds for i = 1:n
		x[i] = a*y[i] - b*z[i]
	end
	return nothing
end

# AIHT UPDATE FOR Z
#
# Compute x = y + a(z - w) in one pass, overwriting x. Equivalent to 
# > copy!(x,w)
# > axpy!(n, 1.0, z, 1, -1.0, x)
# > axpy!(n, 1.0, y, 1, a, x)
#
# Name is "Y Plus A Times Z minus W".
function ypatzmw!{T <: Union(Float32, Float64)}(x::DenseArray{T,1}, y::DenseArray{T,1}, a::T, z::DenseArray{T,1}, w::DenseArray{T,1}; n::Integer = length(x))
	@inbounds for i = 1:n
		x[i] = y[i] + a*(z[i] - w[i])
	end
	return nothing
end



# PROJECT ONTO K LARGEST COMPONENTS OF A VECTOR IN MAGNITUDE
#
# This function projects onto the set S = { x : || x ||_0 <= k }.
# It does so by first finding the indices of the k largest components of x in magnitude.
# Those components are saved, and then b is filled with 0.0.
# The nonzero components are then returned in their proper indices.
#
# Arguments:
# -- b is the vector to project.
# -- bk is a vector to store the values of the largest k components of b in magnitude.
# -- perm is an index vector for b.
# -- k is the number of components of b to preserve.
#
# Optional Arguments:
# -- p is the length of b
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function project_k!{T <: Union(Float32, Float64, Int32, Int64)}(
	b::DenseArray{T,1}, 
	bk::DenseArray{T,1}, 
#	sortk::DenseArray{Int,1}, 
	perm::DenseArray{Int,1}, 
	k::Integer; 
#	p::Integer = length(b)
)
#	sortk = select!(perm, 1:k, by = (i)->abs(b[i]), rev = true)
	kk = k == 1 ? 1 : 1:k
	select!(perm, kk, by = (i)->abs(b[i]), rev = true)
	fill_perm!(bk, b, perm, k=k)	# bk = b[sortk]
	fill!(b,0.0)
	@inbounds for i = 1:k
#		b[sortk[i]] = bk[i]
		b[perm[i]] = bk[i]
	end
	return nothing 
end



# PARTIAL PERMUTATION SORT ON INDICES OF A VECTOR
# This subroutine replaces sortperm to get the top k components of a vector in magnitude.
# By performing only a partial sort, it saves in compute time and memory.
# Feed selectperm() a preallocated vector z of indices for optimal performance.
#
# Arguments:
# -- x is the vector to use for sorting. Note that x itself will not be sorted.
# -- k is the number of elements to choose from the partial sort.
#
# Optional Arguments:
# -- p is the number of elements to sort. Defaults to length(x).
# -- z is the vector of indices to use in the actual sorting process.
#    Defaults to collect(1:p).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function selectperm!(z::DenseArray{Int,1}, x::DenseArray{Float64,1}, k::Integer; p::Int = length(x)) 
    k <= p                 || throw(ArgumentError("selectperm: k cannot exceed length of x!"))
    length(z) == length(x) || throw(DimensionMismatch("Arguments z and x do not have the same length")) 
	kk = k == 1 ? 1 : 1:k
    return select!(z, kk, by = (i)->abs(x[i]), rev = true)
end 


function selectperm(x, k::Integer; p::Integer = length(x), z::Array{Int,1} = collect(1:p))
	k <= p || throw(ArgumentError("selectperm: k cannot exceed length of x!"))
	kk = k == 1 ? 1 : 1:k
	return select!(z, kk, by = (i)->abs(x[i]), rev = true)
end 


# calculate residuals (Y - XB)  piecemeal
# first line does residuals = - X * x_mm
# next line does residuals = residuals + Y = Y - X*x_mm
function update_residuals!(r::Array{Float64,1}, X::Array{Float64,2}, Y::Array{Float64,1}, b::Array{Float64,1})

    # ensure conformable arguments
    length(r) == length(Y) || DimensionMismatch("update_residuals!: output vector must have same length as Y")
    length(b) == size(X,2) || DimensionMismatch("update_residuals!: X and beta not conformable")

    copy!(r, Y)
    BLAS.gemv!('N', -1.0, X, b, 1.0, r)
end



# THRESHOLD THE ELEMENTS OF A VECTOR AGAINST A TOLERANCE
# This subroutine compares the absolute values of the components of a vector x
# against a thresholding tolerance. All elements below the threshold are sent to 0.  
#
# Arguments:
# -- x is the vector to compare to the threshold.
# -- tol is the thresholding tolerance
#
# Optional Arguments:
# -- n is the length of x
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function threshold!(x, tol; n::Int = length(x))
    @inbounds for i = 1:n
        x[i] = ifelse(abs(x[i]) < tol, 0.0, x[i])
    end  
    return x
end


# COMPUTE RESIDUALS (Y - XB)
# This subroutine calculates the residuals (Y - XB)  piecemeal. 
# The first line does residuals = Y. The last line does residuals = Y - XB.
# Note that this subroutine does no explicit error checking.
# Instead, it relies on copy! and BLAS.gemv! to ensure conformable arguments.
#
# Arguments:
# -- r is the preallocated vector of n residuals to overwrite.
# -- x is the nxp design matrix.
# -- y is the n-vector of responses
# -- b is the p-vector of effect sizes
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function update_residuals!(r::DenseArray{Float64,1}, x::DenseArray{Float64,2}, y::DenseArray{Float64,1}, b::DenseArray{Float64,1})
	copy!(r, y)
	BLAS.gemv!('N', -1.0, x, b, 1.0, r)
end

function update_residuals!(r::DenseArray{Float64,1}, x::DenseArray{Float64,2}, y::DenseArray{Float64,1}, b::DenseArray{Float64,1}; xb::DenseArray{Float64,1} = BLAS.gemv('N', 1.0, x, b), n::Int = length(y))
	@inbounds @simd for i = 1:n
		r[i] = y[i] - xb[i]
	end

	return r
end

# COMPUTE WEIGHTED RESIDUALS (Y - 1/2 - diag(W)XB) IN LOGISTIC REGRESSION
# This subroutine, in contrast to the previous update_residuals!() function, 
# will compute WEIGHTED residuals in ONE pass.
# For optimal performance, feed it a precomputed vector of x*b.
# 
# Arguments:
# -- r is the preallocated vector of n residuals to overwrite.
# -- x is the nxp design matrix.
# -- y is the n-vector of responses
# -- b is the p-vector of effect sizes
# -- w is the n-vector of residual weights
#
# Optional Arguments:
# -- xb is the n-vector x*b of predicted responses
#    If x*b is precomputed, then this function will compute the weights much more quickly.
# -- n is the number of samples. Defaults to length(w).
# -- p is the number of predictors. Defaults to length(b).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function update_residuals!(r::DenseArray{Float64,1}, x::DenseArray{Float64,2}, y::DenseArray{Float64,1}, b::DenseArray{Float64,1}, w::DenseArray{Float64,1}; xb::DenseArray{Float64,1} = BLAS.gemv('N', 1.0, x, b), n::Int = length(y), p::Int = length(b))
	(n,p) == size(x) || throw(DimensionMismatch("update_residuals!: nonconformable arguments!"))
	@inbounds @simd for i = 1:n
		r[i] = y[i] - 0.5 - w[i] * xb[i] 
	end
	return r
end




# UPDATE PARTIAL RESIDUALS BASED ON PERMUTATION VECTOR
function update_partial_residuals!(r::DenseArray{Float64,1}, Y::DenseArray{Float64,1}, X::DenseArray{Float64,2}, perm::DenseArray{Int,1}, b::DenseArray{Float64,1}, k::Int; n::Int = length(r), p::Int = length(b))
	k <= p || throw(error("update_partial_residuals!: k cannot exceed the length of b!"))
	copy!(r, Y)
	@inbounds for j = 1:k
		idx = perm[j]
		@inbounds @simd for i = 1:n
			r[i] += -b[idx]*X[i,idx]
		end
	end
	return r
end



# UPDATE SUPPORT OF BETA
# Calculate the vector of booleans indicating nonzero status of betas.
#
# Arguments:
# -- x is the vector of p elements to index.
# -- idx is a BitArray of p Booleans. It contains true for each nonzero component of x.
#    Every other component is false.
#
# Optional Arguments:
# -- p is the number of elements in both x and idx. Defaults to length(x).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function update_indices!(idx::BitArray{1}, x::DenseArray{Float64,1}; p::Int = length(x))
	length(idx) == p || throw(DimensionMismatch("update_indices!: arguments idx and x must have same length!"))
	@inbounds @simd for i = 1:p
		idx[i] = ifelse(x[i] != 0.0, true, false)
	end 
end




# UPDATE ONE WEIGHT
# This subroutine computes the weight of one residual for the surrogate for logistic regression.
# Use this in conjunction with update_weights!().
function one_weight(j, xb)
	0.5*tanh(0.5*xb[j]) / xb[j] 
end

# UPDATE WEIGHTS FOR SURROGATE FUNCTION IN LOGISTIC REGRESSION
#
# This function calculates a vector of weights
#
#     w = 0.5*diag( tanh(0.5 * x * b) ./ x*b )
#
# for the logistic loglikelihood surrogate function by applying the one_weight() function over the entire vector of weights.
# Note that w is actually defined as 0.25 for each component of x*b that equals zero,
# even though the formula above would yield an undefined quantity.
#
# Arguments:
# -- w is the n-vector of weights for the predicted responses.
# -- x is the nxp design matrix.
# -- b is the p-vector of effect sizes.
#
# Optional Arguments:
# -- xb is the n-vector x*b of predicted responses. 
#    If x*b is precomputed, then this function will compute the weights much more quickly.
# -- n is the number of samples. Defaults to length(w).
# -- p is the number of predictors. Defaults to length(b).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function update_weights!(w::DenseArray{Float64,1}, x::DenseArray{Float64,2}, b::DenseArray{Float64,1}; xb::DenseArray{Float64,1} = BLAS.gemv('N', 1.0, x, b), n::Int = length(w), p::Int = length(b))
	(n,p) == size(x) || throw(DimensionMismatch("update_weights!: nonconformable arguments!"))
	@inbounds @simd for i = 1:n
		w[i] = ifelse(xb[i] == 0.0, 0.25, one_weight(i, xb))
	end
	return w
end






# UPDATE A COLUMN VECTOR
# This subroutine updates a temporary 1D array z from a matrix X at column j and scale by a scalar "a".
# Compare this to
# > z = a*x[:,j]
#
# Arguments:
# -- z is the temporary array of length n which we will overwrite.
# -- x is the nxp array from which we will draw
# -- j is the index of the column of x that we will use to fill z
#
# Optional Arguments:
# -- n is the leading dimension of x. Defaults to n = size(x,1)
# -- p is the trailing dimension of x. Defaults to p = size(x, 2)
# -- a is an optional scalar to use for scaling the entries of z. Defaults to 1.0 (no scaling).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function update_col!(z::DenseArray{Float64,1}, x::DenseArray{Float64,2}, j::Int; n::Int=size(x,1), p::Int = size(x,2), a::Float64 = 1.0) 
	length(z) == n || throw(DimensionMismatch("update_col!: arguments z and X must have same number of rows!"))	
	j <= p || throw(DimensionMismatch("update_col!: index j must not exceed number of columns p!")) 
	@inbounds @simd for i = 1:n
		z[i] = a*x[i,j]
	end
	return z
end

# UPDATE A COLUMN VECTOR OF A TEMPORARY ARRAY 
# This subroutine updates a temporary array x from a vector z at column j, scaled by a scalar "a".
# Note that this variant admits a 1D array z and a 2D matrix x. 
# Compare this to
# > x[:,q] = a*z
#
# Arguments:
# -- z is the temporary array of length n which we will put into x 
# -- x is the nxp array into which we will insert z 
# -- j is the index of the column of x that we will use to fill z
#
# Optional Arguments:
# -- n is the leading dimension of x. Defaults to n = size(x,1)
# -- p is the trailing dimension of x. Defaults to p = size(x, 2)
# -- a is an optional scalar to use for scaling the entries of z. Defaults to 1.0 (no scaling).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function update_col!(x::DenseArray{Float64,2}, z::DenseArray{Float64,1}, j::Int; n::Int=size(x,1), p::Int = size(x,2), a::Float64 = 1.0) 
	length(z) == n || throw(DimensionMismatch("update_col!: arguments z and X must have same number of rows!"))	
	j <= p || throw(DimensionMismatch("update_col!: index j must not exceed number of columns p!")) 
	@inbounds @simd for i = 1:n
		x[i,j] = a*z[i]
	end

	return z
end


# UPDATE A COLUMN VECTOR OF A TEMPORARY ARRAY 
# This subroutine updates a temporary array z from a matrix x at column j, scaled by a scalar "a".
# Note that this variant admits a 2D array z and a 2D matrix x. 
# It also requires one to declare which column of x goes into which column of z. 
# Compare this to
# > Xk[:,q] = a*X[:,j]
#
# Arguments:
# -- z is the nxk temporary array. 
# -- x is the nxp array from which we will draw the column to insert into z.
# -- j is the index of the column of x that we will use to fill z
# -- q is the index of the column of z that will receive the column from x.
#
# Optional Arguments:
# -- n is the leading dimension of x. Defaults to n = size(x,1)
# -- p is the trailing dimension of x. Defaults to p = size(x, 2)
# -- a is an optional scalar to use for scaling the entries of z. Defaults to 1.0 (no scaling).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function update_col!(z::DenseArray{Float64,2}, x::DenseArray{Float64,2}, j::Int, q::Int; n::Int=size(x,1), p::Int = size(x,2), a::Float64 = 1.0) 
	size(z,1) == n || throw(DimensionMismatch("update_col!: arguments z and X must have same number of rows!"))	
	j <= p         || throw(DimensionMismatch("update_col!: index j must not exceed number of columns p!")) 
	@inbounds @simd for i = 1:n
		z[i,q] = a*x[i,j]
	end
	return z
end



# UPDATE A REDUCED MATRIX
# This subroutine updates a submatrix Xk containing k columns of X as selected by an index vector idxvec.
# Compare this to
# > Xk = a*X[:,idxvec]
#
# Arguments:
# -- xk is the nxk temporary array to fill from x.
# -- x is the nxp array from which we will fill xk.
# -- idxvec is a BitArray that indexes the columns of x to send into xk. It should have a true for each 
#    column that goes into xk and a false otherwise.
#
# Optional Arguments:
# -- k is the trailing dimension of xk. Defaults to size(xk,2).
# -- p is the trailing dimension of x. Defaults to length(idxvec) and is used to test conformability with x.
# -- n is the leading dimension of x and xk. Defaults to size(x,1).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function update_xk!(xk::DenseArray{Float64,2}, x::DenseArray{Float64,2}, idxvec::BitArray{1}; k::Int = size(xk,2), p::Int = length(idxvec), n::Int = size(x,1))
	k >= sum(idxvec) || throw(DimensionMismatch("update_xk!: number of trues in idxvec must not exceed the number of columns in xk!"))	
	p == size(x,2)   || throw(DimensionMismatch("update_xk!: number of components in idxvec must equal the number of columns in x!"))	

	# counter j is used to track the number of trues in idxvec
	j = 0

	# loop over entire vector idxvec
	@inbounds @simd for i = 1:p

		# if current component of idxvec is a true, then increment j and fill column
		if idxvec[i]
			j += 1
			update_col!(xk, x, i, j, n=n, p=p)
		end
	end
	return xk
end

# FILL A VECTOR VIA A PERMUTATION
# This subroutine fills a vector x of length k from a vector y of length p via an index vector idx.
# This variant admits BitArray index vectors.
#
# Arguments:
# -- x is the k-vector that we will fill.
# -- y is the p-vector that we use to fill x.
# -- idx is a BitArray that indexes the components of y to put into x
#
# Optional Arguments:
# -- k = length(x).
# -- p = length(idx). 
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function fill_perm!(x::DenseArray{Float64,1}, y::DenseArray{Float64,1}, idx::BitArray{1}; k::Int = length(x), p::Int = length(idx))
#	k <= sum(idx) || throw(DimensionMismatch("fill_perm!: length(x) != sum(idx)")) 

	# counter j is used to track the number of trues in idx
	j = 0

	# loop over entire vector idx
	@inbounds for i = 1:p

		# if current component of idx is a true, then increment j and fill x from y 
		if idx[i]
			j += 1
			x[j] = y[i]
		end

		# once x has k components, then it is completely filled and we return it
		j == k && return x
	end

	return x
end

# FILL A VECTOR VIA A PERMUTATION
# This subroutine fills a vector x of length k from a vector y of length p via an index vector idx.
# This variant admits Int index vectors.
#
# Arguments:
# -- x is the k-vector that we will fill.
# -- y is the p-vector that we use to fill x.
# -- idx is an Int array that indexes the components of y to put into x
#
# Optional Arguments:
# -- k = length(x).
# -- p = length(idx). 
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function fill_perm!(x::DenseArray{Float64,1}, y::DenseArray{Float64,1}, idx::DenseArray{Int,1}; k::Int = length(x))
	k <= length(idx) || throw(DimensionMismatch("fill_perm!: length(x) != length(idx)")) 
	@inbounds for i = 1:k
			x[i] = y[idx[i]]
	end
	return x
end

# COMPUTE THE LOGISTIC LOGLIKELIHOOD (Y - 0.5)'XB - LOG(COSH(0.5*XB)) 
# This subroutine computes the logistic likelihood in one pass.
# For optimal performance, supply this function with a precomputed x*b. 
# 
# Arguments:
# -- y is the n-vector of responses
# -- x is the nxp design matrix.
# -- b is the p-vector of effect sizes.
#
# Optional Arguments:
# -- n is the number of samples. Defaults to length(y).
# -- xb is the n-vector x*b of predicted responses
#    If x*b is precomputed, then this function will compute the loglikelihood more quickly. 
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function compute_loglik(y::DenseArray{Float64,1}, x::DenseArray{Float64,2}, b::DenseArray{Float64,1}; n::Int = length(y), xb::Array{Float64,1} = BLAS.gemv('N',1.0,x,b)) 
	n == length(xb) || throw(DimensionMismatch("compute_loglik: y and X*b must have same length!"))

	# initialize return value
	s = 0.0

	# each part accumulates sum s
	@inbounds @simd for i = 1:n
		s += y[i]*xb[i] - softplus(xb[i])
	end

	return s
end


# UPDATE X*B, OR UPDATE A MATRIX-VECTOR PRODUCT BY AN INDEX VECTOR
#
# This function performs the matrix-vector product x*b, where b has k nonzeroes.
# The nonzeroes are encoded in the first k elements of the index vector indices.
#
# Arguments:
# -- xb is the array that stores x*b, which we will overwrite.
# -- x is the nxp design matrix.
# -- b is the px1 parameter vector.
# -- indices is a vector of integers that indexes b. The first k should correspond to the k nonzeroes.
# -- k is the number of nonzeroes in b.
#
# Optional Arguments:
# -- p = length(b)
# -- n = length(xb) = size(x,1)
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function update_xb!(xb::DenseArray{Float64,1}, x::DenseArray{Float64,2}, b::DenseArray{Float64,1}, indices::DenseArray{Int,1}, k::Int; p::Int = length(b), n::Int = size(x,1))

	fill!(xb, 0.0)
	for i = 1:k
#		println("update_xb!: i = ", i)
		idx = indices[i]
#		println("update_xb!: idx = ", idx)
		@inbounds for j = 1:n
#			println("update_xb!: j = ", j)
			xb[j] += b[idx]*x[j,idx] 
		end
	end

	return xb
end

# UPDATE X*B, OR UPDATE A MATRIX-VECTOR PRODUCT BY AN INDEX VECTOR
#
# This function performs the matrix-vector product x*b, where b has k nonzeroes.
# The nonzeroes are encoded in the first k elements of the index vector indices.
# This function wraps update_xb!.
#
# Arguments:
# -- x is the nxp design matrix.
# -- b is the px1 parameter vector.
# -- indices is a vector of integers that indexes b. The first k should correspond to the k nonzeroes.
# -- k is the number of nonzeroes in b.
#
# Optional Arguments:
# -- p = length(b)
# -- n = length(xb) = size(x,1)
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function update_xb(x::DenseArray{Float64,2}, b::DenseArray{Float64,1}, indices::DenseArray{Int,1}, k::Int; p::Int = length(b), n::Int = size(x,1))
	xb = zeros(n)
	return update_xb!(xb, x, b, indices, k, p=p, n=n)
end

# COUNT NONZEROES ON A SUBVECTOR
# This subroutine counts nonzeroes starting from index k0 and ending at k (inclusive) 
function count_partialnz(x::DenseArray{Float64,1}, k0::Int, k::Int)
	k0 <= k || throw(ArgumentError("Start index must not exceed end index"))
	s::Int = 0
	for i = k0:k
		if x[i] != 0.0
			s += 1
		end
	end
	return s
end

# PARTIALLY FILL A VECTOR WITH ONE VALUE
# This subroutine performs a partial fill on a vector.
# It only works if the components are filled to the same value!
# For more complicated partial fills, consider fill_perm!(). 
function fill_partial!(x::DenseArray{Float64,1}, a::Float64, k0::Int, k::Int)
	1 <= k0 <= k || throw(ArgumentError("fill_partial!: Start index must lie between 1 and end index"))
	@inbounds @simd for i = k0:k
		x[i] = a
	end
	return x
end


# COMPUTE THE MEAN CLASSIFICATION ERROR
# This subroutine computes the average number of misclassified responses in logistic regression.
function mce(y::DenseArray{Float64,1}, x::DenseArray{Float64,2}, b::DenseArray{Float64,1}; xb::Array{Float64,1} = BLAS.gemv('N', 1.0, x, b))
	n == length(xb) || throw(DimensionMismatch("y and xb must have same length"))
	s = 0.0
	t = 0.0
	@inbounds @simd for i = 1:n
		t  = (round(y[i]) - round(logistic(xb[i])))
		t *= t
		s += t 
	end
	return s / n
end

# COMPUTE THE MEAN SQUARED ERROR
# This subroutine computes the average of the residuals in least squares regression.
function mse(y::DenseArray{Float64,1}, x::DenseArray{Float64,2}, b::DenseArray{Float64,1}; xb::Array{Float64,1} = BLAS.gemv('N', 1.0, x, b), n::Int = length(y))
	n == length(xb) || throw(DimensionMismatch("y and xb must have same length"))
	s = 0.0
	t = 0.0
	@inbounds @simd for i = 1:n
		t  = (y[i] - xb[i])
		t *= t
		s += t 
	end
	return s / n
end


# COMPUTE THE MEAN SQUARED ERROR
# This subroutine computes the average of the residuals in least squares regression.
# This variant works on a matrix B of betas. 
# It computes the MSE from ONE column of B.
function mse(y::DenseArray{Float64,1}, x::DenseArray{Float64,2}, B::DenseArray{Float64,1}, j::Int; xB::Array{Float64,2} = BLAS.gemm('N', 'N', 1.0, x, B), n::Int = length(y), k::Int = size(B,2))
	n == size(xB,1) || throw(DimensionMismatch("y and xb must have same length"))
	j <= k          || throw(DimensionMismatch("Column index exceeds number of columns in B"))
	s = 0.0
	t = 0.0
	@inbounds @simd for i = 1:n
		t  = (y[i] - xB[i,j])
		t *= t
		s += t 
	end
	return s / n
end


# COMPUTE THE MEAN SQUARED ERROR
# This subroutine computes the average of the residuals in least squares regression.
# This variant works on a matrix B of betas. 
# It computes the MSE from ONE column of B.
function mse(y::DenseArray{Float64,1}, x::DenseArray{Float64,2}, B::SparseMatrixCSC{Float64,Int}, j::Int; xB::DenseArray{Float64,2} = x*B, n::Int = length(y), k::Int = size(B,2))
	n == size(xB,1) || throw(DimensionMismatch("y and xb must have same length"))
	j <= k          || throw(DimensionMismatch("Column index exceeds number of columns in B"))
	s = 0.0
	t = 0.0
	@inbounds @simd for i = 1:n
		t  = (y[i] - xB[i,j])
		t *= t
		s += t 
	end
	return s / n
end


# COMPUTE THE MEAN SQUARED ERROR
# This subroutine computes the average of the residuals in least squares regression.
# This variant works on a matrix of betas. 
function mses(y::DenseArray{Float64,1}, x::DenseArray{Float64,2}, B::DenseArray{Float64,1}; xB::Array{Float64,2} = BLAS.gemm('N', 'N', 1.0, x, B), n::Int = length(y), k::Int = size(B,2))
	n == size(xB,1) || throw(DimensionMismatch("y and xb must have same length"))
	output = zeros(k)
	@inbounds @simd for i = 1:k
		output[i] = mse(y, x, B, i, xB=xB, n=n, k=k)
	end
	return output 
end


# COMPUTE THE MEAN SQUARED ERROR
# This subroutine computes the average of the residuals in least squares regression.
# This variant works on a *sparse* matrix of betas. 
function mses(y::DenseArray{Float64,1}, x::DenseArray{Float64,2}, B::SparseMatrixCSC{Float64,Int}; xB::Array{Float64,2} = x*B, n::Int = length(y), k::Int = size(B,2))
	n == size(xB,1) || throw(DimensionMismatch("y and xb must have same length"))
	output = zeros(k)
	@inbounds @simd for i = 1:k
		output[i] = mse(y, x, B, i, xB=xB, n=n, k=k)
	end
	return output 
end


# COMPUTE THE PARTIAL GRADIENT OF THE LOGISTIC LOGLIKELIHOOD 
#
# This routine computes an approximate gradient for the loglikelihood based on the first k elements of the vector indices.
# N.B. there is little evidence that this function is useful! Partial gradients seem to lead to instability in the algorithm.
function loggrad(x::DenseArray{Float64,2}, y::DenseArray{Float64,1}, b::DenseArray{Float64,1}, indices::DenseArray{Int,1}, k::Int; n::Int = length(y), p::Int = size(x,2), xb::DenseArray{Float64,1} = BLAS.gemv('N', 1.0, x, b))
	size(x,1) == n  || throw(DimensionMismatch("x and y must have same number of rows"))	
	length(xb) == n || throw(DimensionMismatch("xb and y must have same number of rows"))	

	lg = zeros(p)
	return loggrad!(lg, x, y, b, indices, k, n=n, p=p, xb=xb) 
end

# calculate gradient of loglikelihood based on indices over smaller subspace
function loggrad!(lg::DenseArray{Float64,1}, x::DenseArray{Float64,2}, y::DenseArray{Float64,1}, b::DenseArray{Float64,1}, indices::DenseArray{Int,1}, k::Int; n::Int = length(y), p::Int = size(x,2), xb::DenseArray{Float64,1} = update_xb(x,b,indices,k, n=n, p=p)) 
	size(x,1) == n       || throw(DimensionMismatch("x and y must have same number of rows"))	
	length(lg) == p      || throw(DimensionMismatch("length(lg) != p"))	
	length(indices) >= k || throw(DimensionMismatch("length(indices) < k"))	

	fill!(lg, 0.0)
	for i = 1:k
		idx = indices[i]
		for j = 1:n
			lg[i] += (y[j] - logistic(xb[j]))*x[j,idx]
		end
	end

	return lg 
end




# compute the logistic function over a vector, save to another vector
function logistic!(y::DenseArray{Float64,1}, x::DenseArray{Float64,1}; n::Int = length(y))
	length(x) == n || throw(DimensionMismatch("length(y) != length(x)"))
	@inbounds @simd for i = 1:n
		y[i] = logistic(x[i])
	end
	return y
end

# compute the logistic function over a vector for Newton's method, save to another vector
function logistic!(y::DenseArray{Float64,1}, xb::DenseArray{Float64,1}, xlg::DenseArray{Float64,1}, mu::Float64; n::Int = length(y))
	length(xb) == n  || throw(DimensionMismatch("length(y) != length(xb)"))
	length(xlg) == n || throw(DimensionMismatch("length(y) != length(xlg)"))
	@inbounds @simd for i = 1:n
		y[i] = logistic(xb[i] + mu*xlg[i])
	end
	return y
end

# compute y - logistic(xb), based on precomputed logistic(xb), and save to another vector
function update_y2!(y2::DenseArray{Float64,1}, y::DenseArray{Float64,1}, lxb::DenseArray{Float64,1}; n::Int = length(y2))
	@inbounds @simd for i = 1:n
		y2[i] = y[i] - lxb[i]
	end
	return y2
end


# COMPUTE LOGISTIC PROBABILITIES
#
# For logistic probabilities pi = logistic(x*b), this subroutine computes pi and pi (1 - pi) in one pass.
# It relies on a precomputed xb = x*b.
#
# Arguments:
# -- lxb  = logistic(xb) = pi. The return value is actually -pi so that we can quickly compute (y - pi) with BLAS.axpy!
# -- l2xb = pi (1 - pi). The function computes lxb and then reuses it for l2xb, so the actual calculation is (-pi)[1 - (-pi)].
# -- xb   = x*b (precomputed!)
#
# Optional Arguments:
# -- n = length(xb)
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function log2xb!(lxb, l2xb, xb; n::Int = length(xb))
	@inbounds for i = 1:n
		lxb[i]  = -logistic(xb[i])
		l2xb[i] = -lxb[i]*( 1 + lxb[i])
	end
end


# FIT A COMPLETE LOGISTIC MODEL
#
# This function computes a logistic fit using Newton's method.
# Use with caution! The function is meant to be used after model selection.
# Consequently, it assumes that x does not have many columns and attains full column rank.
#
# Arguments:
# -- x is the design matrix.
# -- y is the binary response. 
#
# Optional Arguments:
# -- n = length(y)
# -- p = size(x,2)
# -- b is the statistical model. Defaults to zeros, and fills in the intercept (first component) with logit(mean(y)).
# -- tol is the convergence tolerance. Defaults to 1e-8.
# -- max_iter is the maximum permissible number of Newton steps. Defaults to 50.
# -- quiet is a Boolean to activate output. Defaults to true (no output). Use this primarily for debugging Newton's method.
# -- various temporary arrays:
#	# preallocate temporary arrays
#	b0   = copy(b)		 old b, track for convergence purposes 
#	ntb  = zeros(p)		 Newton step for b
#	db   = zeros(p)		 gradient
#	d2b  = zeros(p,p)	 Hessian
#	x2   = copy(x)		 used in intermediate calculation for d2b
#	xb   = zeros(n)		 xb = x*b 
#	lxb  = zeros(n)		 logistic(xb), which we call pi 
#	l2xb = zeros(n)		 logistic(xb) [ 1 - logistic(xb) ], or pi(1 - pi) 
# 
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function fit_logistic(x::DenseArray{Float64,2}, y::DenseArray{Float64,1}; n::Int = length(y), p::Int = size(x,2), b::DenseArray{Float64,1} = zeros(p), tol::Float64 = 1e-8, max_iter::Int = 50, quiet::Bool = true, b0::DenseArray{Float64,1} = copy(b), ntb::DenseArray{Float64,1} = zeros(p), db::DenseArray{Float64,1} = zeros(p), d2b::DenseArray{Float64,2} = zeros(p,p), x2::DenseArray{Float64,2} = copy(x), xb::DenseArray{Float64,1} = zeros(n), lxb::DenseArray{Float64,1} = zeros(n), l2xb::DenseArray{Float64,1} = zeros(n))

	# if b is not warm-started, then ensure that it is not entirely zero
	if sum(b) == 0.0
		b[1] = logit(mean(y))
	end
	
	# initialize intermediate arrays for calculations
	BLAS.gemv!('N', 1.0, x, b, 0.0, xb)
	log2xb!(lxb, l2xb, xb, n=n) 

	# track objective
	old_obj = -Inf
	new_obj = compute_loglik(y,x,b, n=n, xb=xb) 

	# output progress to console
	quiet || println("Iter\tHalves\tObjective")

	# enter loop for Newton's method
	for i = 1:max_iter

		# db = x'*(y - logistic(xb))
		BLAS.axpy!(n, 1.0, sdata(y), 1, sdata(lxb), 1)
		BLAS.gemv!('T', 1.0, sdata(x), sdata(lxb), 0.0, sdata(db))

		# d2b = x'*diagm(logistic(xb))*diagm((1 - logistic(xb)))*x		
		copy!(x2,x)
		scale!(sdata(l2xb), sdata(x2))
		BLAS.gemm!('T', 'N', 1.0, sdata(x), sdata(x2), 0.0, sdata(d2b))
		
		# b = b0 - inv(d2b)*db = b0 + inv[ x' diag(pi) diag(1 - pi) x ] x' (y - pi)
		ntb = d2b\db
		b = b0 + ntb 

		# compute objective
		new_obj = compute_loglik(y,x,b, n=n, xb=xb) 

		# backtrack
		j = 0
		while (new_obj < old_obj - tol) && j < 50

			# b = b0 + 0.5*ntb
#			b = fma(sdata(b0), 0.5, sdata(ntb))
			for i = 1:p
				b[i] = b[i] + 0.5*ntb[i]
			end

			# recalculate objective
			new_obj = compute_loglik(y,x,b, n=n, xb=xb)

			# increment iterator
			j += 1
		end

		# track distance between iterates
		dist = euclidean(sdata(b),sdata(b0)) / (norm(sdata(b0),2) + 1.0)

		# track progress
		quiet || println(i, "\t", j, "\t", dist)

		# check for convergence
		# if converged, then return b
		dist < tol && return(b)

		# unconverged at this point, so update intermediate arrays
		BLAS.gemv!('N', 1.0, sdata(x), sdata(b), 0.0, sdata(xb))
		log2xb!(lxb, l2xb, xb, n=n) 

		# save previous beta
		copy!(b0, b)
		old_obj = new_obj

	end

	warn("fit_logistic failed to converge in $(max_iter) iterations, exiting...")
	return b
end

end	# end module RegressionTools
