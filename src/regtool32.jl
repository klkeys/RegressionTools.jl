# DIFFERENCE OF TWO VECTORS
#
# Compute the difference x = y - z, overwriting x.
function difference!(
	x :: DenseArray{Float32,1}, 
	y :: DenseArray{Float32,1}, 
	z :: DenseArray{Float32,1}; 
	a :: Float32 = 1.0f0, 
	b :: Float32 = 1.0f0,
	n :: Integer = length(x)
)
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
function ypatzmw!(
	x :: DenseArray{Float32,1}, 
	y :: DenseArray{Float32,1}, 
	a :: Float32, 
	z :: DenseArray{Float32,1}, 
	w :: DenseArray{Float32,1}; 
	n :: Integer = length(x))
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
function project_k!(
	b    :: DenseArray{Float32,1}, 
	bk   :: DenseArray{Float32,1}, 
#	sortk:: DenseArray{Int,1}, 
	perm :: DenseArray{Int,1}, 
	k    :: Integer; 
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
function selectperm!(
	z :: DenseArray{Int,1}, 
	x :: DenseArray{Float32,1}, 
	k :: Integer; 
	p :: Integer = length(x)
) 
    k <= p                 || throw(ArgumentError("selectperm: k cannot exceed length of x!"))
    length(z) == length(x) || throw(DimensionMismatch("Arguments z and x do not have the same length")) 
	kk = k == 1 ? 1 : 1:k
    select!(z, kk, by = (i)->abs(x[i]), rev = true)
	return nothing
end 


function selectperm(
	x :: DenseArray{Float32,1}, 
	k :: Integer; 
	p :: Integer           = length(x), 
	z :: DenseArray{Int,1} = collect(1:p)
)
	k <= p || throw(ArgumentError("selectperm: k cannot exceed length of x!"))
	kk = k == 1 ? 1 : 1:k
	return select!(z, kk, by = (i)->abs(x[i]), rev = true)
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
function threshold!(
	x   :: DenseArray{Float32,1}, 
	tol :: Float32; 
	n   :: Integer = length(x)
)
    @inbounds for i = 1:n
        x[i] = ifelse(abs(x[i]) < tol, 0.0, x[i])
    end  
    return nothing 
end



# calculate residuals (Y - XB)  piecemeal
# first line does residuals = - X * x_mm
# next line does residuals = residuals + Y = Y - X*x_mm
function update_residuals!(
	r :: Array{Float32,1}, 
	x :: Array{Float32,2}, 
	y :: Array{Float32,1}, 
	b :: Array{Float32,1}
)

    # ensure conformable arguments
    length(r) == length(y) || DimensionMismatch("update_residuals!: output vector must have same length as Y")
    length(b) == size(x,2) || DimensionMismatch("update_residuals!: X and beta not conformable")

    copy!(r, y)
    BLAS.gemv!('N', -1.0, x, b, 1.0, r)
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
function update_residuals!(
	r :: DenseArray{Float32,1}, 
	x :: DenseArray{Float32,2}, 
	y :: DenseArray{Float32,1}, 
	b :: DenseArray{Float32,1}
)
	copy!(r, y)
	BLAS.gemv!('N', -1.0, x, b, 1.0, r)
end

function update_residuals!(
	r  :: DenseArray{Float32,1}, 
	x  :: DenseArray{Float32,2}, 
	y  :: DenseArray{Float32,1}, 
	b  :: DenseArray{Float32,1}; 
	xb :: DenseArray{Float32,1} = BLAS.gemv('N', 1.0, x, b), 
	n  :: Integer               = length(y)
)
	@inbounds @simd for i = 1:n
		r[i] = y[i] - xb[i]
	end

	return nothing 
end



# UPDATE PARTIAL RESIDUALS BASED ON PERMUTATION VECTOR
function update_partial_residuals!(
	r    :: DenseArray{Float32,1}, 
	y    :: DenseArray{Float32,1}, 
	x    :: DenseArray{Float32,2}, 
	perm :: DenseArray{Int,1}, 
	b    :: DenseArray{Float32,1}, 
	k    :: Integer; 
	n    :: Integer = length(r), 
	p    :: Integer = length(b)
)
	k <= p || throw(error("update_partial_residuals!: k cannot exceed the length of b!"))
	copy!(r, y)
	@inbounds for j = 1:k
		idx = perm[j]
		@inbounds @simd for i = 1:n
			r[i] += -b[idx]*x[i,idx]
		end
	end
	return nothing 
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
function update_indices!(
	idx :: BitArray{1}, 
	x   :: DenseArray{Float32,1}; 
	p   :: Integer = length(x)
)
	length(idx) == p || throw(DimensionMismatch("update_indices!: arguments idx and x must have same length!"))
	@inbounds @simd for i = 1:p
		idx[i] = ifelse(x[i] != 0.0, true, false)
	end 
	return nothing
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
function update_col!(
	z :: DenseArray{Float32,1}, 
	x :: DenseArray{Float32,2}, 
	j :: Integer; 
	n :: Integer = size(x,1), 
	p :: Integer = size(x,2), 
	a :: Float32 = 1.0f0
) 
	length(z) == n || throw(DimensionMismatch("update_col!: arguments z and X must have same number of rows!"))	
	j <= p || throw(DimensionMismatch("update_col!: index j must not exceed number of columns p!")) 
	@inbounds @simd for i = 1:n
		z[i] = a*x[i,j]
	end
	return nothing 
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
function update_col!(
	x :: DenseArray{Float32,2}, 
	z :: DenseArray{Float32,1}, 
	j :: Integer; 
	n :: Integer = size(x,1), 
	p :: Integer = size(x,2), 
	a :: Float32 = 1.0f0
) 
	length(z) == n || throw(DimensionMismatch("update_col!: arguments z and X must have same number of rows!"))	
	j <= p || throw(DimensionMismatch("update_col!: index j must not exceed number of columns p!")) 
	@inbounds @simd for i = 1:n
		x[i,j] = a*z[i]
	end

	return nothing 
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
function update_col!(
	z :: DenseArray{Float32,2}, 
	x :: DenseArray{Float32,2}, 
	j :: Integer, 
	q :: Integer; 
	n :: Integer = size(x,1), 
	p :: Integer = size(x,2), 
	a :: Float32 = 1.0f0
) 
	size(z,1) == n || throw(DimensionMismatch("update_col!: arguments z and X must have same number of rows!"))	
	j <= p         || throw(DimensionMismatch("update_col!: index j must not exceed number of columns p!")) 
	@inbounds @simd for i = 1:n
		z[i,q] = a*x[i,j]
	end
	return nothing 
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
function update_xk!(
	xk     :: DenseArray{Float32,2}, 
	x      :: DenseArray{Float32,2}, 
	idxvec :: BitArray{1}; 
	k      :: Integer = size(xk,2), 
	p      :: Integer = length(idxvec), 
	n      :: Integer = size(x,1)
)
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
	return nothing 
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
function fill_perm!(
	x   :: DenseArray{Float32,1}, 
	y   :: DenseArray{Float32,1}, 
	idx :: BitArray{1}; 
	k   :: Integer = length(x), 
	p   :: Integer = length(idx)
)
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

	return nothing 
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
function fill_perm!(
	x   :: DenseArray{Float32,1}, 
	y   :: DenseArray{Float32,1}, 
	idx :: DenseArray{Int,1}; 
	k   :: Integer = length(x)
)
	k <= length(idx) || throw(DimensionMismatch("fill_perm!: length(x) != length(idx)")) 
	@inbounds for i = 1:k
			x[i] = y[idx[i]]
	end
	return nothing 
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
function update_xb!(
	xb      :: DenseArray{Float32,1}, 
	x       :: DenseArray{Float32,2}, 
	b       :: DenseArray{Float32,1}, 
	indices :: DenseArray{Int,1}, 
	k       :: Integer; 
	p       :: Integer = length(b), 
	n       :: Integer = size(x,1)
)
	fill!(xb, 0.0)
	for i = 1:k
		idx = indices[i]
		@inbounds for j = 1:n
			xb[j] += b[idx]*x[j,idx] 
		end
	end

	return nothing 
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
function update_xb(
	x       :: DenseArray{Float32,2}, 
	b       :: DenseArray{Float32,1}, 
	indices :: DenseArray{Int,1}, 
	k       :: Integer; 
	p       :: Integer = length(b), 
	n       :: Integer = size(x,1)
)
	xb = zeros(n)
	update_xb!(xb, x, b, indices, k, p=p, n=n)
	return xb
end

# COUNT NONZEROES ON A SUBVECTOR
# This subroutine counts nonzeroes starting from index k0 and ending at k (inclusive) 
function count_partialnz(x::DenseArray{Float32,1}, k0::Integer, k::Integer)
	k0 <= k || throw(ArgumentError("Start index must not exceed end index"))
	s = 0
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
function fill_partial!(x::DenseArray{Float32,1}, a::Float32, k0::Integer, k::Integer)
	1 <= k0 <= k || throw(ArgumentError("fill_partial!: Start index must lie between 1 and end index"))
	@inbounds @simd for i = k0:k
		x[i] = a
	end
	return x
end



# COMPUTE THE MEAN SQUARED ERROR
# This subroutine computes the average of the residuals in least squares regression.
function mse(
	y  :: DenseArray{Float32,1}, 
	x  :: DenseArray{Float32,2}, 
	b  :: DenseArray{Float32,1}; 
	xb :: DenseArray{Float32,1} = BLAS.gemv('N', 1.0, x, b), 
	n  :: Integer               = length(y)
)
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
function mse(
	y  :: DenseArray{Float32,1}, 
	x  :: DenseArray{Float32,2}, 
	b  :: DenseArray{Float32,1}, 
	j  :: Integer; 
	xb :: DenseArray{Float32,2} = BLAS.gemm('N', 'N', 1.0, x, b), 
	n  :: Integer = length(y), 
	k  :: Integer = size(b,2)
)
	n == size(xb,1) || throw(DimensionMismatch("y and xb must have same length"))
	j <= k          || throw(DimensionMismatch("Column index exceeds number of columns in B"))
	s = zero(Float32) 
	t = zero(Float32) 
	@inbounds @simd for i = 1:n
		t  = (y[i] - xb[i,j])
		t *= t
		s += t 
	end
	return s / n
end


# COMPUTE THE MEAN SQUARED ERROR
# This subroutine computes the average of the residuals in least squares regression.
# This variant works on a matrix B of betas. 
# It computes the MSE from ONE column of B.
function mse(
	y  :: DenseArray{Float32,1}, 
	x  :: DenseArray{Float32,2}, 
	b  :: SparseMatrixCSC{Float32,Int}, 
	j  :: Integer; 
	xb :: DenseArray{Float32,2} = x*b, 
	n  :: Integer               = length(y), 
	k  :: Integer               = size(b,2)
)
	n == size(xb,1) || throw(DimensionMismatch("y and xb must have same length"))
	j <= k          || throw(DimensionMismatch("Column index exceeds number of columns in B"))
	s = zero(Float32) 
	t = zero(Float32) 
	@inbounds @simd for i = 1:n
		t  = (y[i] - xb[i,j])
		t *= t
		s += t 
	end
	return s / n
end


# COMPUTE THE MEAN SQUARED ERROR
# This subroutine computes the average of the residuals in least squares regression.
# This variant works on a matrix of betas. 
function mses(
	y  :: DenseArray{Float32,1}, 
	x  :: DenseArray{Float32,2}, 
	b  :: DenseArray{Float32,1}; 
	xb :: DenseArray{Float32,2} = BLAS.gemm('N', 'N', 1.0, x, B), 
	n  :: Integer               = length(y), 
	k  :: Integer               = size(b,2)
)
	n == size(xb,1) || throw(DimensionMismatch("y and xb must have same length"))
	output = zeros(Float32,k)
	@inbounds @simd for i = 1:k
		output[i] = mse(y, x, B, i, xB=xB, n=n, k=k)
	end
	return output 
end


# COMPUTE THE MEAN SQUARED ERROR
# This subroutine computes the average of the residuals in least squares regression.
# This variant works on a *sparse* matrix of betas. 
function mses(
	y  :: DenseArray{Float32,1}, 
	x  :: DenseArray{Float32,2}, 
	b  :: SparseMatrixCSC{Float32,Int}; 
	xb :: DenseArray{Float32,2} = x*B, 
	n  :: Integer               = length(y), 
	k  :: Integer               = size(B,2)
)
	n == size(xB,1) || throw(DimensionMismatch("y and xb must have same length"))
	output = zeros(Float32,k)
	@inbounds @simd for i = 1:k
		output[i] = mse(y, x, B, i, xB=xB, n=n, k=k)
	end
	return output 
end
