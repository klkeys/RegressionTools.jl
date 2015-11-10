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
function update_residuals!(r::DenseVector{Float64}, x::DenseMatrix{Float64}, y::DenseVector{Float64}, b::DenseVector{Float64}, w::DenseVector{Float64}; xb::DenseVector{Float64} = BLAS.gemv('N', 1.0, x, b), n::Int = length(y), p::Int = length(b))
	(n,p) == size(x) || throw(DimensionMismatch("update_residuals!: nonconformable arguments!"))
	@inbounds @simd for i = 1:n
		r[i] = y[i] - 0.5 - w[i] * xb[i] 
	end
	return r
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
function update_weights!(w::DenseVector{Float64}, x::DenseMatrix{Float64}, b::DenseVector{Float64}; xb::DenseVector{Float64} = BLAS.gemv('N', 1.0, x, b), n::Int = length(w), p::Int = length(b))
	(n,p) == size(x) || throw(DimensionMismatch("update_weights!: nonconformable arguments!"))
	@inbounds @simd for i = 1:n
		w[i] = ifelse(xb[i] == 0.0, 0.25, one_weight(i, xb))
	end
	return w
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
function compute_loglik(y::DenseVector{Float64}, x::DenseMatrix{Float64}, b::DenseVector{Float64}; n::Int = length(y), xb::Array{Float64,1} = BLAS.gemv('N',1.0,x,b)) 
	n == length(xb) || throw(DimensionMismatch("compute_loglik: y and X*b must have same length!"))

	# initialize return value
	s = 0.0

	# each part accumulates sum s
	@inbounds @simd for i = 1:n
		s += y[i]*xb[i] - softplus(xb[i])
	end

	return s
end


# COMPUTE THE MEAN CLASSIFICATION ERROR
# This subroutine computes the average number of misclassified responses in logistic regression.
function mce(y::DenseVector{Float64}, x::DenseMatrix{Float64}, b::DenseVector{Float64}; xb::Array{Float64,1} = BLAS.gemv('N', 1.0, x, b))
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


# COMPUTE THE PARTIAL GRADIENT OF THE LOGISTIC LOGLIKELIHOOD 
#
# This routine computes an approximate gradient for the loglikelihood based on the first k elements of the vector indices.
# N.B. there is little evidence that this function is useful! Partial gradients seem to lead to instability in the algorithm.
function loggrad(x::DenseMatrix{Float64}, y::DenseVector{Float64}, b::DenseVector{Float64}, indices::DenseVector{Int}, k::Int; n::Int = length(y), p::Int = size(x,2), xb::DenseVector{Float64} = BLAS.gemv('N', 1.0, x, b))
	size(x,1) == n  || throw(DimensionMismatch("x and y must have same number of rows"))	
	length(xb) == n || throw(DimensionMismatch("xb and y must have same number of rows"))	

	lg = zeros(p)
	return loggrad!(lg, x, y, b, indices, k, n=n, p=p, xb=xb) 
end

# calculate gradient of loglikelihood based on indices over smaller subspace
function loggrad!(lg::DenseVector{Float64}, x::DenseMatrix{Float64}, y::DenseVector{Float64}, b::DenseVector{Float64}, indices::DenseVector{Int}, k::Int; n::Int = length(y), p::Int = size(x,2), xb::DenseVector{Float64} = update_xb(x,b,indices,k, n=n, p=p)) 
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
function logistic!(y::DenseVector{Float64}, x::DenseVector{Float64}; n::Int = length(y))
	length(x) == n || throw(DimensionMismatch("length(y) != length(x)"))
	@inbounds @simd for i = 1:n
		y[i] = logistic(x[i])
	end
	return y
end


# compute the logistic function over a vector for Newton's method, save to another vector
function logistic!(y::DenseVector{Float64}, xb::DenseVector{Float64}, xlg::DenseVector{Float64}, mu::Float64; n::Int = length(y))
	length(xb) == n  || throw(DimensionMismatch("length(y) != length(xb)"))
	length(xlg) == n || throw(DimensionMismatch("length(y) != length(xlg)"))
	@inbounds @simd for i = 1:n
		y[i] = logistic(xb[i] + mu*xlg[i])
	end
	return y
end

# compute y - logistic(xb), based on precomputed logistic(xb), and save to another vector
function update_y2!(y2::DenseVector{Float64}, y::DenseVector{Float64}, lxb::DenseVector{Float64}; n::Int = length(y2))
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
function fit_logistic(x::DenseMatrix{Float64}, y::DenseVector{Float64}; n::Int = length(y), p::Int = size(x,2), b::DenseVector{Float64} = zeros(p), tol::Float64 = 1e-8, max_iter::Int = 50, quiet::Bool = true, b0::DenseVector{Float64} = copy(b), ntb::DenseVector{Float64} = zeros(p), db::DenseVector{Float64} = zeros(p), d2b::DenseMatrix{Float64} = zeros(p,p), x2::DenseMatrix{Float64} = copy(x), xb::DenseVector{Float64} = zeros(n), lxb::DenseVector{Float64} = zeros(n), l2xb::DenseVector{Float64} = zeros(n))

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
