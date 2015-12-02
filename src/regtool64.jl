"""
    mask!(x,b,val,mask_val,n)

A subroutine to mask entries of a vector `x`.

Arguments:

- `x` is the vector to mask.
- `v` is the `Int` vector used in masking.
- `val` is the value of `v` to use in masking.
- `mask_val` is the actual value of the mask, i.e. if `v[i] = val` then `x[i] = mask_val`.

Optional Arguments:

- `n = length(x)`.
"""
function mask!(
    x        :: DenseVector{Float64},
    v        :: DenseVector{Int},
    val      :: Int,
    mask_val :: Float64;
    n        :: Int = length(x)
)
    n == length(v) || throw(BoundsError("Vector x and its mask must have same length"))
    @inbounds for i = 1:n
        if v[i] == val
            x[i] = mask_val
        end
    end
    return nothing
end

"Can also be called with an `Int` argument `n` instead of the data vector `y`."
function cv_get_folds(n::Int, q::Int)
    m, r = divrem(n, q)
    shuffle!([repmat(1:q, m); 1:r])
end

"""
CREATE UNSTRATIFIED CROSSVALIDATION PARTITION

    cv_get_folds(y,q) -> Vector{Int}

This function will partition the `n` components of `y` into `q` disjoint sets for unstratified `q`-fold crossvalidation.

Arguments:

- `y` is the `n`-vector to partition.
- `q` is the number of disjoint sets in the partition.
"""
function cv_get_folds(y::DenseVector, q::Int)
    n, r = divrem(length(y), q)
    shuffle!([repmat(1:q, n); 1:r])
end


"""
    standardize!(j, x [, return_values=false, n=size(x,1), p=size(x,2)])

For the `j`-th column of an `n` x `p` matrix `x`,
compute both the mean `m` and standard deviation `s` of `x[:,j]`
and overwrite `x[:,j]` with its standardized variant.
This function assumes all finite values, so any `NA` or `NaN` will poison the computations!

Arguments:

- `j` is the index of the column of `x` to use in calculations
- `x` is the matrix to use in calculations

Optional Arguments:

- `return_values` is a `Bool` to indicate whether or not to return `m` and `s`. Defaults to `false`.
- `n` and `p` are the dimensions of `x`.

Output:

- if `return_values` is true, then `standardize!` returns the mean `m` and standard deviation `s` of `x[:,j]`.
"""
function standardize!(
    j :: Int,
    x :: DenseMatrix{Float64};
    return_values :: Bool = false,
    n :: Int = size(x,1),
    p :: Int = size(x,2)
)

    j <= p || throw(ArgumentError("Column index $i exceeds number of columns $p in x"))

    # accumulation variables
    m = 0.0
    s = 0.0

    # temporary float
    t = 0.0

    # accumulate mean
    @inbounds for i = 1:n
        m += x[i,j]
    end

    # normalize mean
    m /= n

    # calculate variance
    @inbounds for i = 1:n
        t = x[i,j] - m
        t *= t
        x[i,j] = t
        s += t
    end

    # get (unbiased) std
    s = sqrt(s / (n-1))

    # normalize column
    @inbounds for i = 1:n
        x[i,j] /= s
    end

    return_values && return m, s
    return nothing
end


"""
    difference!(x, y, z [, a=1.0, b=1.0, n=length(x)])

Compute the difference `x = a*y - b*z`, overwriting `x`.
"""
function difference!(
    x :: DenseVector{Float64},
    y :: DenseVector{Float64},
    z :: DenseVector{Float64};
    a :: Float64 = 1.0,
    b :: Float64 = 1.0,
    n :: Int = length(x)
)
    @inbounds for i = 1:n
        x[i] = a*y[i] - b*z[i]
    end
    return nothing
end

"""
    ypatzmw!(x, y, a, z, w[, n=length(x)])

Compute x = y + a(z - w) in one pass, overwriting x. Equivalent to

    copy!(x,w)
    axpy!(n, 1.0, z, 1, -1.0, x)
    axpy!(n, 1.0, y, 1, a, x)

The unusual name represents *Y* *P*lus *A* *T*imes *Z* *m*inus *W*.
Note that this function discards bounds checks for speed,
so the onous is on the user to supply vectors of the same dimension!
"""
function ypatzmw!(
    x :: DenseVector{Float64},
    y :: DenseVector{Float64},
    a :: Float64,
    z :: DenseVector{Float64},
    w :: DenseVector{Float64};
    n :: Int = length(x))
    @inbounds for i = 1:n
        x[i] = y[i] + a*(z[i] - w[i])
    end
    return nothing
end


"""
    project_k!(b, bk, perm, k)

This function projects a vector `b` onto the set S_k = { x in R^p : || x ||_0 <= k }.
It does so by first finding the indices of the `k` largest components of `x` in magnitude.
Those components are saved in the array `bk`, and then `b` is filled with zeros.
The nonzero components from `bk` are then returned to their proper indices in `b`.

Arguments:

- `b` is the vector to project.
- `bk` is a vector to store the values of the largest `k` components of `b` in magnitude.
- `perm` is an `Int` array that indexes `b`.
- `k` is the number of components of `b` to preserve.
"""
function project_k!(
    b    :: DenseVector{Float64},
    bk   :: DenseVector{Float64},
    perm :: DenseVector{Int},
    k    :: Int;
)
    kk = k == 1 ? 1 : 1:k
    select!(perm, kk, by = (i)->abs(b[i]), rev = true)
    fill_perm!(bk, b, perm, k=k)    # bk = b[sortk]
    fill!(b,zero(Float64))
    @inbounds for i = 1:k
        b[perm[i]] = bk[i]
    end
    return nothing
end



"""
    selectpermk!(z, x, k [, p=length(x)])

This subroutine extends `sortperm` to get the top `k` components of a vector in magnitude.
By performing only a partial sort, it saves in compute time and memory.
This function is superceded in Julia v0.4 with `selectperm!`, so it is included only for backward compatibility with Julia v0.3-.

Arguments:

- `z` is the vector of indices to use in the actual sorting process.
- `x` is the vector to use for sorting. Note that `x` itself will not be sorted.
- `k` is the number of elements to choose from the partial sort.

Optional Arguments:

- `p` is the number of elements to sort. Defaults to `length(x)`.
"""
function selectpermk!(
    z :: DenseVector{Int},
    x :: DenseVector{Float64},
    k :: Int;
    p :: Int = length(x)
)
    k <= p                 || throw(ArgumentError("selectperm: k cannot exceed length of x!"))
    length(z) == length(x) || throw(DimensionMismatch("Arguments z and x do not have the same length"))
    kk = k == 1 ? 1 : 1:k
    select!(z, kk, by = (i)->abs(x[i]), rev = true)
    return nothing
end

"""
    selectpermk(x, k [, p=length(x), z=collect(1:p(])

This function produces a `p`-vector `z` whose top `k` entries partially sort `x` by magnitude.
"""
function selectpermk(
    x :: DenseVector{Float64},
    k :: Int;
    p :: Int           = length(x),
    z :: DenseVector{Int} = collect(1:p)
)
    k <= p || throw(ArgumentError("selectperm: k cannot exceed length of x!"))
    kk = k == 1 ? 1 : 1:k
    return select!(z, kk, by = (i)->abs(x[i]), rev = true)
end



"""
    threshold!(x, tol [, n=length(x)])

This subroutine compares the absolute values of the components of a vector `x`
against a thresholding tolerance `tol`. All elements below `tol` are sent to zero.

Arguments:

- `x` is the vector to threshold.
- `tol` is the thresholding tolerance

Optional Arguments:

- `n` is the length of `x`.
"""
function threshold!(
    x   :: DenseVector{Float64},
    tol :: Float64;
    n   :: Int = length(x)
)
    @inbounds for i = 1:n
        x[i] = ifelse(abs(x[i]) < tol, 0.0, x[i])
    end
    return nothing
end


"""
    update_indices!(idx, x [, p = length(x)])

Calculate the vector of booleans indicating nonzero status of betas.

Arguments:

- `idx` is a `BitArray` of `p` `Bool`s. It contains `true` for each nonzero component of `x` and `false` otherwise.
- `x` is the `p`-vector to index.

Optional Arguments:

- `p` is the number of elements in both `x` and `idx`. Defaults to `length(x)`.
"""
function update_indices!(
    idx :: BitArray{1},
    x   :: DenseVector{Float64};
    p   :: Int = length(x)
)
    length(idx) == p || throw(DimensionMismatch("update_indices!: arguments idx and x must have same length!"))
    @inbounds @simd for i = 1:p
        idx[i] = ifelse(x[i] != 0.0, true, false)
    end
    return nothing
end




"""
    update_col!(z, x, j [, n=size(x,1), p=size(x,2), a=1.0])

This subroutine overwrites an `n`-vector `z` with the `j`th column of a matrix `x` scaled by `a`.
It is more efficient than `z = a*x[:,j]`.

Arguments:

- `z` is the `n`-vector to fill with `x[:,j]`.
- `x` is the `n` x `p` matrix to use in filling `z`.
- `j` indexes the column of `x` to use in filling `z`.

Optional Arguments:

- `n` is the leading dimension of `x`. Defaults to `size(x,1)`.
- `p` is the trailing dimension of `x`. Defaults to `size(x,2)`.
- `a` scales the entries of `z`. Defaults to `1.0` (no scaling).
"""
function update_col!(
    z :: DenseVector{Float64},
    x :: DenseMatrix{Float64},
    j :: Int;
    n :: Int = size(x,1),
    p :: Int = size(x,2),
    a :: Float64 = 1.0
)
    length(z) == n || throw(DimensionMismatch("update_col!: arguments z and X must have same number of rows!"))
    j <= p || throw(DimensionMismatch("update_col!: index j must not exceed number of columns p!"))
    @inbounds @simd for i = 1:n
        z[i] = a*x[i,j]
    end
    return nothing
end

"""
    update_col!(x, z, j [, n=size(x,1), p=size(x,2), a=1.0])

Fills the `j`th column of an `n` x `p` matrix `x` with the entries in the `n`-vector `z`.
"""
function update_col!(
    x :: DenseMatrix{Float64},
    z :: DenseVector{Float64},
    j :: Int;
    n :: Int = size(x,1),
    p :: Int = size(x,2),
    a :: Float64 = 1.0
)
    length(z) == n || throw(DimensionMismatch("update_col!: arguments z and X must have same number of rows!"))
    j <= p || throw(DimensionMismatch("update_col!: index j must not exceed number of columns p!"))
    @inbounds for i = 1:n
        x[i,j] = a*z[i]
    end

    return nothing
end


"""
    update_col!(z, x, j, q [, n=size(x,1), p=size(x,2), a=1.0])

Updates the `q`th column of a matrix `z` with the `j`th column of a matrix `x` scaled by `a`. Both `x` and `z` must have the same leading dimension `n`. Both `j` and `q` must not exceed the trailing dimension `p` of `x`.
"""
function update_col!(
    z :: DenseMatrix{Float64},
    x :: DenseMatrix{Float64},
    j :: Int,
    q :: Int;
    n :: Int = size(x,1),
    p :: Int = size(x,2),
    a :: Float64 = 1.0
)
    size(z,1) == n || throw(DimensionMismatch("update_col!: arguments z and X must have same number of rows!"))
    j <= p         || throw(DimensionMismatch("update_col!: index j must not exceed number of columns p!"))
    @inbounds @simd for i = 1:n
        z[i,q] = a*x[i,j]
    end
    return nothing
end


"""
    update_xk!(xk, x, idx [, k=size(xk,2), p=length(idx), n=size(x,1)])

This subroutine updates a submatrix `xk` of `k` columns of `x` as selected by `idx`.
It efficiently performs `xk = x[:,idx]`.

Arguments:

- `xk` is the `n` x `k` array to fill from `x`.
- `x` is the `n` x `p` array to use in filling `xk`.
- `idx` is a BitArray with a `true` for every column of `x` to write into `xk` and `false` otherwise.

Optional Arguments:

- `k` is the trailing dimension of `xk`. Defaults to `size(xk,2)`.
- `p` is the trailing dimension of `x`. Defaults to `length(idx)` and is used to test conformability with `x`.
- `n` is the leading dimension of `x` and `xk`. Defaults to `size(x,1)`.
"""
function update_xk!(
    xk  :: DenseMatrix{Float64},
    x   :: DenseMatrix{Float64},
    idx :: BitArray{1};
    k   :: Int = size(xk,2),
    p   :: Int = length(idx),
    n   :: Int = size(x,1)
)
    k >= sum(idx) || throw(DimensionMismatch("update_xk!: number of trues in idx must not exceed the number of columns in xk!"))
    p == size(x,2)   || throw(DimensionMismatch("update_xk!: number of components in idx must equal the number of columns in x!"))

    # counter j is used to track the number of trues in idxvec
    j = 0

    # loop over entire vector idxvec
    @inbounds for i = 1:p

        # if current component of idxvec is a true, then increment j and fill column
        if idx[i]
            j += 1
            update_col!(xk, x, i, j, n=n, p=p)
        end

        j == k && return nothing
    end
    return nothing
end

"""
    fill_perm!(x, y, idx [, k=length(x), p=length(idx)])

This subroutine fills a `k`-vector `x` from a `p`-vector `y` via an index vector `idx`.
This variant admits BitArray index vectors.

Arguments:

- `x` is the `k`-vector to fill.
- `y` is the `p`-vector to use in filling `x`.
- `idx` is either a `BitArray` or `Int` vector` that indexes the components of `y` to put into `x`. If `idx` contains `Int`s, then only the first `k` indices are used. Otherwise, `fill_perm!()` traverses `idx` until it encounters `k` `true`s.

Optional Arguments:

- `k = length(x)`.
- `p = length(idx)`.
"""
function fill_perm!(
    x   :: DenseVector{Float64},
    y   :: DenseVector{Float64},
    idx :: BitArray{1};
    k   :: Int = length(x),
    p   :: Int = length(idx)
)
#   k <= sum(idx) || throw(DimensionMismatch("fill_perm!: length(x) != sum(idx)"))

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
        j == k && return nothing
    end

    return nothing
end

function fill_perm!(
    x   :: DenseVector{Float64},
    y   :: DenseVector{Float64},
    idx :: DenseVector{Int};
    k   :: Int = length(x)
)
    k <= length(idx) || throw(DimensionMismatch("fill_perm!: length(x) != length(idx)"))
    @inbounds for i = 1:k
            x[i] = y[idx[i]]
    end
    return nothing
end



"""
    update_xb!(Xb, x, b, indices, k [, p=length(b), n=size(x,1)])

This function efficiently performs the "sparse" matrix-vector product `x*b`, of an `n` x `p` matrix `x` and a `p`-vector `b` with `k` nonzeroes.
The nonzeroes are encoded in the first `k` elements of the `Int vector `indices`.

Arguments:

- `Xb` is the array to overwrite with `x*b`.
- `x` is the `n` x `p` design matrix.
- `b` is the `p`-dimensional parameter vector.
- `indices` is a vector of integers that indexes `b`. The first `k` elements of `indices` should correspond to the `k` nonzeroes of `b`.
- `k` is the number of nonzeroes in `b`.

Optional Arguments:

- `p` is the trailing dimension of `x` and the dimension of `b`. Defaults to `length(b)`
- `n` is the leading dimension of `x` and the dimension of `Xb`. Defaults to `size(x,1)`
"""
function update_xb!(
    Xb      :: DenseVector{Float64},
    x       :: DenseMatrix{Float64},
    b       :: DenseVector{Float64},
    indices :: DenseVector{Int},
    k       :: Int;
    p       :: Int = length(b),
    n       :: Int = size(x,1)
)
    fill!(Xb, 0.0)
    @inbounds for i = 1:k
        idx = indices[i]
        @inbounds for j = 1:n
            Xb[j] += b[idx]*x[j,idx]
        end
    end

    return nothing
end

"""
    update_xb(x, b, indices, k [, p=length(b), n=size(x,1)])

This function efficiently performs the "sparse" matrix-vector product `x*b`, of an `n` x `p` matrix `x` and a `p`-vector `b` with `k` nonzeroes.
The nonzeroes are encoded in the first `k` elements of the `Int vector `indices`.
"""
function update_xb(
    x       :: Matrix{Float64},
    b       :: Vector{Float64},
    indices :: DenseVector{Int},
    k       :: Int;
    p       :: Int = length(b),
    n       :: Int = size(x,1)
)
    Xb = zeros(Float64, n)
    update_xb!(Xb, x, b, indices, k, p=p, n=n)
    return Xb
end


function update_xb(
    x       :: SharedMatrix{Float64},
    b       :: SharedVector{Float64},
    indices :: DenseVector{Int},
    k       :: Int;
    pids    :: DenseVector = procs(),
    p       :: Int = length(b),
    n       :: Int = size(x,1)
)
    Xb = SharedArray(Float64, n, init = S -> S[localindexes(S)] = 0.0, pids=procs())
    update_xb!(Xb, x, b, indices, k, p=p, n=n)
    return Xb
end

"""
    count_partialnz(x, k0, k)

This subroutine counts nonzeroes in `x` from `k0` to `k` (inclusive).
It is more efficient than countnz(x[k0,k]).
"""
function count_partialnz(x::DenseVector{Float64}, k0::Int, k::Int)
    k0 <= k || throw(ArgumentError("Start index must not exceed end index"))
    s = 0
    for i = k0:k
        if x[i] != 0.0
            s += 1
        end
    end
    return s
end

"""
    fill_partial!(x, a, k0, k)

This subroutine efficiently performs `x[k0:k] = a`.
For more complicated partial fills, consider `fill_perm!()`.
"""
function fill_partial!(x::DenseVector{Float64}, a::Float64, k0::Int, k::Int)
    1 <= k0 <= k || throw(ArgumentError("fill_partial!: Start index must lie between 1 and end index"))
    @inbounds @simd for i = k0:k
        x[i] = a
    end
    return x
end
