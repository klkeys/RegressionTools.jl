"""
    mce(lxb, y, mask_n [, n=length(y), mn=length(mask_n)])

For statistical design matrix `x`, model `b`, and binary response `y`, compute the mean classification error between a vector `lxb = logistic(x*b)` and `y` with potentially masked entries indicated by the `0`s of the `Int` vector `mask_n`.
"""
function mce{T <: Float}(
    lxb    :: DenseVector{T},
    y      :: DenseVector{T},
    mask_n :: DenseVector{Int};
    n      :: Int = length(y),
    mn     :: Int = sum(mask_n)
)
    s = zero(T)
    @inbounds for i = 1:n
        if mask_n[i] == 1
            s += abs(round(lxb[i]) - round(y[i]))
        end
    end
    return s / mn
end

"""
    update_x!(x, x0, y, idxs, a [, k=length(idxs)])

If fed an `Int` vector `idxs` of length `p` that sorts `x` by `abs()`, then `x` is updated as `x[idxs[1:k]] = x0[idxs[1:k]] - a*y[idxs[1:k]]`.
The output is comparable to

    copy!(x,x0)
    BLAS.axpy!(p,-a,y,1,x,1)

but `update_x!()` exploits the implied sparsity of `x` by only updating the top `k` components in magnitude.
"""
function update_x!{T <: Float}(
    x     :: DenseVector{T},
    x0    :: DenseVector{T},
    y     :: DenseVector{T},
    idxs  :: DenseVector{Int},
    a     :: T;
    k     :: Int = length(idxs)
)
    fill!(x, zero(T))
    @inbounds for i = 1:k
        idx = idxs[i]
        x[idx] = x0[idx] - a*y[idx]
    end
    return nothing
end


"""
    df_norm(df, idxs, k0, k)

This subroutine efficiently computes `norm(df[idxs[k0:k]])`.
"""
function df_norm{T <: Float}(
    df   :: DenseVector{T},
    idxs :: DenseVector{Int},
    k0   :: Int,
    k    :: Int
)
    s = zero(T)
    @inbounds for i = k0:k
        idx = idxs[i]
        s += df[idx]*df[idx]
    end
    return sqrt(s)
end

"""
    logistic!(y, x [, n=length(y)])

This subroutine computes `y = logistic(x)` in-place, modifying `y`."
"""
function logistic!{T <: Float}(
	y :: DenseVector{T},
	x :: DenseVector{T};
	n :: Int = length(y)
)
	n == length(x) || throw(ArgumentError("length(y) != length(x)"))
	@inbounds @simd for i = 1:n
		y[i] = one(T) / (one(T) + exp(-x[i]))
	end
	return nothing
end

"""
    logistic_loglik(Xb, y, b, sortidx, lambda, k [, n=length(Xb)])

If fed a vector `idxs` of indices that sort `b`, then `logistic_loglik` computes the loglikelihood over the `k` active components of `b`.
"""
function logistic_loglik{T <: Float}(
	Xb      :: DenseVector{T},
    y       :: DenseVector{T},
	b       :: DenseVector{T},
	sortidx :: DenseVector{Int},
	lambda  :: T,
	k       :: Int;
	n       :: Int = length(Xb)
)
	s = zero(T)
	@inbounds @simd for i = 1:n
		s += log(one(T) + exp(Xb[i])) - Xb[i]*y[i]
	end
    s /= n
	@inbounds for i = 1:k
		idx = sortidx[i]
        s  += lambda*b[idx]*b[idx] / 2
	end
	return s
end

"""
    logistic_loglik(Xb, y, b, lambda, k [, n=length(Xb)])

Compute the regularized negative logistic loglikelihood

   ( sum(log(1 + exp(x*b))) - y'*x*b) / n + 0.5*lambda*sumabs2(b)

The unregularized loglikelihood is computed by setting `lambda = 0.0`.

Arguments:

- `Xb = x*b`. Note that this must be precomputed; the function does not compute `x*b`.
- `y` is the response vector of `0`s and `1`s.
- `b` is the statistical model.
- `lambda` is the regularization parameter.
- `k` is the number of nonzeroes in b.

Optional Arguments:

- `n = length(Xb)`. Note that `logistic_loglik` assumes that `length(Xb) = length(y)` and performs no inbounds checks.
"""
function logistic_loglik{T <: Float}(
	Xb      :: DenseVector{T},
    y       :: DenseVector{T},
	b       :: DenseVector{T},
	lambda  :: T,
	k       :: Int;
	n       :: Int = length(xb)
)
	s = zero(T)
	@inbounds @simd for i = 1:n
		s += log(one(T) + exp(Xb[i])) - Xb[i]*y[i]
	end
    s /= n
	@inbounds for i = 1:k
        s += lambda*b[i]*b[i] / 2
	end
	return s
end


function logistic_loglik{T <: Float}(
	Xb      :: DenseVector{T},
    y       :: DenseVector{T},
	b       :: DenseVector{T},
    mask_n  :: DenseVector{Int},
    m_val   :: Int,
	lambda  :: T,
	k       :: Int;
	n       :: Int = length(xb),
    mn      :: Int = sum(mask_n),
)
	s = zero(T)
	@inbounds for i = 1:n
        if mask_n[i] != m_val
            s += log(one(T) + exp(Xb[i])) - Xb[i]*y[i]
        end
	end
#    s /= n
    s /= mn
	@inbounds for i = 1:k
        s += lambda*b[i]*b[i] / 2
	end
	return s
end


function logistic_loglik{T <: Float}(
	Xb      :: DenseVector{T},
    y       :: DenseVector{T},
	b       :: DenseVector{T},
    idxs    :: DenseVector{Int},
    mask_n  :: DenseVector{Int},
    m_val   :: Int,
	lambda  :: T,
	k       :: Int;
	n       :: Int = length(xb),
    mn      :: Int = sum(mask_n),
)
	s = zero(T)
	@inbounds for i = 1:n
        if mask_n[i] != m_val
            s += log(one(T) + exp(Xb[i])) - Xb[i]*y[i]
        end
	end
#    s /= n
    s /= mn
	@inbounds for i = 1:k
        idx = idxs[i]
        s += lambda*b[idx]*b[idx] / 2
	end
	return s
end

function logistic_loglik{T <: Float}(
	Xb      :: DenseVector{T},
    y       :: DenseVector{T},
	b       :: DenseVector{T},
    idxs    :: BitArray{1},
    mask_n  :: DenseVector{Int},
    m_val   :: Int,
	lambda  :: T,
	k       :: Int;
	n       :: Int = length(xb),
    mn      :: Int = sum(mask_n),
)
	s = zero(T)
	@inbounds for i = 1:n
        if mask_n[i] != m_val
            s += log(one(T) + exp(Xb[i])) - Xb[i]*y[i]
        end
	end
#    s /= n
    s /= mn
    nk = 0
	@inbounds for i = 1:length(idxs)
        idx = idxs[i]
        if idx
            s += lambda*b[idx]*b[idx] / 2
            nk += 1
        end
        nk >= k && break
	end
	return s
end

"""
    logistic_grad!(df, lxb, x, y, b, Xb, idxs, k, lambda [, n=length(Xb)])

Compute the *partial* gradient of the regularized negative logistic loglikelihood as indexed by the `Int` vector `idxs`.
"""
function logistic_grad!{T <: Float}(
    df     :: DenseVector{T},
    lxb    :: DenseVector{T},
    x      :: DenseMatrix{T},
    y      :: DenseVector{T},
    b      :: DenseVector{T},
    Xb     :: DenseVector{T},
    idxs   :: DenseVector{Int},
    k      :: Int,
    lambda :: T;
    n      :: Int = length(Xb)
)
    logistic!(lxb,Xb,n=n)
    fill!(df, zero(T))
    @inbounds for i = 1:k
        idx = idxs[i]
#        df[idx] = zero(T)
        @inbounds @simd for j = 1:n
            df[idx] += x[j,idx] * (lxb[j] - y[j])
        end
        df[idx] /= n
        df[idx] += lambda*b[idx]
    end
    return nothing
end

"""
    logistic_grad!(df, lxb, x::BEDFile, y, b, Xb, means, invstds, idxs, mask_n, k, lambda [, n=length(Xb)])

Compute the *partial* gradient of the regularized negative logistic loglikelihood, as indexed by the `Int` vector `idxs`, with a `BEDFile` object `x`.
"""
function logistic_grad!{T <: Float}(
    df      :: DenseVector{T},
    lxb     :: DenseVector{T},
    x       :: BEDFile,
    y       :: DenseVector{T},
    b       :: DenseVector{T},
    Xb      :: DenseVector{T},
    means   :: DenseVector{T},
    invstds :: DenseVector{T},
    idxs    :: DenseVector{Int},
    mask_n  :: DenseVector{Int},
    k       :: Int,
    lambda  :: T;
    n       :: Int = length(Xb),
    mn      :: Int = sum(mask_n),
)
    logistic!(lxb,Xb,n=n)
    fill!(df, zero(T))
    @inbounds for i = 1:k
        idx = idxs[i]
        m   = means[idx]
        d   = invstds[idx]
#        df[idx] = zero(T)
        @inbounds for j = 1:n
            if mask_n[i] == 1
                df[idx] += (x[j,idx] - m) * d * (lxb[j] - y[j])
            end
        end
#        df[idx] /= n
        df[idx] /= mn
        df[idx] += lambda*b[idx]
    end
    return nothing
end


"""
    logistic_grad!(df, lxb, x, y, b, Xb, lambda [, n=length(Xb), p=length(df)])

Computes the full gradient of the negative logistic loglikelihood.
"""
function logistic_grad!{T <: Float}(
    df     :: DenseVector{T},
    lxb    :: DenseVector{T},
    x      :: DenseMatrix{T},
    y      :: DenseVector{T},
    b      :: DenseVector{T},
    Xb     :: DenseVector{T},
    lambda :: T;
    n      :: Int = length(Xb),
    p      :: Int = length(df),
)
    logistic!(lxb,Xb,n=n)
    BLAS.axpy!(n,-one(T),y,1,lxb,1)
    BLAS.gemv!('T',one(T),x,lxb,zero(T),df)
    BLAS.scal!(p,1/n,df,1)
    BLAS.axpy!(p,lambda,b,1,df,1)
    return nothing
end

"""
    logistic_grad!(df, lxb, x::BEDFile, y, b, Xb, means, invstds, lambda, [, n=length(Xb), p=length(df)])

Computes the full gradient of the negative logistic loglikelihood with a `BEDFile` object `x`.
"""
function logistic_grad!{T <: Float}(
    df      :: SharedVector{T},
    lxb     :: SharedVector{T},
    x       :: BEDFile,
    y       :: SharedVector{T},
    b       :: SharedVector{T},
    Xb      :: SharedVector{T},
    means   :: SharedVector{T},
    invstds :: SharedVector{T},
    mask_n  :: DenseVector{Int},
    lambda  :: T;
    pids    :: DenseVector{Int} = procs(),
    n       :: Int = length(Xb),
    p       :: Int = length(df),
    mn      :: Int = sum(mask_n),
)
    logistic!(lxb,Xb,n=n)
    BLAS.axpy!(n,-one(T),sdata(y),1,sdata(lxb),1)
    xty!(df,x,lxb,mask_n, means=means, invstds=invstds, pids=pids)
#    BLAS.scal!(p,1/n,sdata(df),1)
    BLAS.scal!(p,1/mn,sdata(df),1)
    BLAS.axpy!(p,lambda,sdata(b),1,sdata(df),1)
    return nothing
end


"""
    log2xb!(lxb, l2xb, Xb [, n=length(Xb)])

Compute `lxb = logistic(Xb)` and `l2xb = lxb*(1 - lxb)` in-place and in one pass, overwriting `lxb` and `l2xb`.
"""
function log2xb!{T <: Float}(
    lxb  :: DenseVector{T},
    l2xb :: DenseVector{T},
    Xb   :: DenseVector{T};
    n    :: Int = length(Xb)
)
    @inbounds for i = 1:n
        lxb[i]  = one(T) / (one(T) + exp(-Xb[i]))
        l2xb[i] = lxb[i]*(one(T) - lxb[i]) / n
    end
    return nothing
end

"""
    fit_logistic(x, y, lambda)

Refit a regularized logistic model using Newton's method.

Arguments:

- `x` is the `n` x `p` statistical model.
- `y` is the `n`-vector of binary responses.
- `lambda` is the regularization parameter

Optional Arguments:

- `n` is the number of cases. Defaults to `length(y)`.
- `p` is the number of predictors. Defaults to `size(x,2)`.
- `tol` is the convergence tolerance. Defaults to `1e-8`.
- `max_iter` sets the maximum number of backtracking steps. Defaults to `50`.
- `quiet` controls output. Defaults to `true` (no output).
- Several temporary arrays:

    d2b  = zeros(T,p,p)  # Hessian of b
    x2   = zeros(T,n,p)  # copy of x used for updating Hessian
    b    = zeros(T,p)    # copy of bk used in refitting
    b0   = zeros(T,p)    # copy of bk used in refitting
    ntb  = zeros(T,p)    # Newton step for bk used in refitting
    db   = zeros(T,p)    # gradient of bk used in refitting
    Xb   = zeros(T,n)    # x*b
    lxb  = zeros(T,n)    # logistic(x*b) = 1 ./ (1 + exp(-x*b))
    l2xb = zeros(T,n)    # lxb * (1 - lxb)

Output:

- a vector of refit coefficients for `b`
- the number of backtracking steps taken.
"""
function fit_logistic{T <: Float}(
    x        :: DenseMatrix{T},
    y        :: DenseVector{T},
    lambda   :: T;
    n        :: Int    = length(y),
    p        :: Int    = size(x,2),
    d2b      :: DenseMatrix{T} = zeros(T, p,p),
    x2       :: DenseMatrix{T} = zeros(T, n,p),
    b        :: DenseVector{T} = zeros(T, p),
    b0       :: DenseVector{T} = zeros(T, p),
    ntb      :: DenseVector{T} = zeros(T, p),
    db       :: DenseVector{T} = zeros(T, p),
    Xb       :: DenseVector{T} = zeros(T, n),
    lxb      :: DenseVector{T} = zeros(T, n),
    l2xb     :: DenseVector{T} = zeros(T, n),
    tol      :: Float = convert(T, 1e-8),
    max_iter :: Int   = 50,
    quiet    :: Bool  = true,
)

    # if b is not warm-started, then ensure that it is not entirely zero
    if all(b .== 0) 
        b[1] = logit(abs(mean(y)))
    end

    # initialize intermediate arrays for calculations
    BLAS.gemv!('N', one(T), x, b, zero(T), Xb)
    log2xb!(lxb, l2xb, Xb, n=n)
    copy!(b0,b)
    fill!(db, zero(T))

    # track objective
    old_obj = oftype(tol, Inf)
#    new_obj = logistic_loglik(Xb,y,b,lambda,p, n=n)
    new_obj = (sum(log(1 + exp(Xb))) - dot(y,Xb)) / n + lambda*sumabs2(b) / 2

    # output progress to console
    quiet || println("Iter\tHalves\tObjective")

    i = 0
    bktrk = 0
    # enter loop for Newton's method
    for i = 1:max_iter

        # db = (x'*(lxb - y)) / n + lambda*b
        BLAS.axpy!(n, -one(T), sdata(y), 1, sdata(lxb), 1)
        BLAS.gemv!('T', one(T), sdata(x), sdata(lxb), zero(T), sdata(db))
        BLAS.scal!(p, 1/n, sdata(db), 1)
#        scale!(sdata(db), 1/n)
        BLAS.axpy!(p, lambda, sdata(b), 1, sdata(db), 1)

        # d2b = (x'*diagm(l2xb)*x)/n + lambda*I
        # note that log2xb!() already performs division by n on l2xb
        copy!(x2,x)
        scale!(sdata(l2xb), sdata(x2))
        BLAS.gemm!('T', 'N', one(T), sdata(x), sdata(x2), zero(T), sdata(d2b))
#        At_mul_B!(sdata(d2b), sdata(x), sdata(x2))
        d2b += lambda*I

        # b = b0 - ntb = b0 - inv(d2b)*db
        #   = b0 - inv[ x' diagm(pi) diagm(1 - pi) x + lambda*I] [x' (pi - y) + lambda*b]
        try
            ntb = d2b\db
        catch e
            warn("in fit_logistic, aborting after caught error: ", e)
            return b, i, div(bktrk,max_iter)
        end
        copy!(b,b0)
        BLAS.axpy!(p,-one(T),ntb,1,b,1)

        # compute objective
#        new_obj = logistic_loglik(Xb,y,b,lambda,p, n=n)
        BLAS.gemv!('N', one(T), x, b, zero(T), Xb)
        new_obj = (sum(log(1 + exp(Xb))) - dot(y,Xb)) / n + lambda*sumabs2(b) / 2

        # control against nonfinite objective
        !isfinite(new_obj) && throw(error("in fit_logistic, objective is no longer finite"))

        # backtrack
        j = 0
        while (new_obj > old_obj + tol) && (j < 50)

            # increment iterator
            j += 1

            # b = b0 - 0.5*ntb
            copy!(b,b0)
            BLAS.axpy!(p,-one(T) / (2^j),ntb,1,b,1)

            # recalculate objective
#            new_obj = logistic_loglik(Xb,y,b,lambda,p, n=n)
            BLAS.gemv!('N', one(T), x, b, zero(T), Xb)
            new_obj = (sum(log(1 + exp(Xb))) - dot(y,Xb)) / n + lambda*sumabs2(b) / 2
            !isfinite(new_obj) && throw(error("in fit_logistic, objective is no longer finite"))
        end

        # accumulate total backtracking steps
        bktrk += j

        # track distance between iterates
        dist = euclidean(sdata(b),sdata(b0)) / (norm(sdata(b0),2) + one(T))

        # track progress
        quiet || println(i, "\t", j, "\t", dist)

        # check for convergence
        # if converged, then return b
        dist < tol && return b, i, div(bktrk,i)

        # unconverged at this point, so update intermediate arrays
        BLAS.gemv!('N', one(T), sdata(x), sdata(b), zero(T), sdata(Xb))
        log2xb!(lxb, l2xb, Xb, n=n)

        # save previous beta
        copy!(b0, b)
        old_obj = new_obj
    end

    warn("fit_logistic failed to converge in $(max_iter) iterations, exiting...")
    return b, i, div(bktrk,max_iter)
end

"""
    fit_logistic(x, y, mask_n, lambda)

If called with an `Int` vector `mask_n`, then `fit_logistic()` will refit logistic effect sizes while masking components `y[i]` where `mask_n[i] = 0`.
"""
function fit_logistic{T <: Float}(
    x        :: DenseMatrix{T},
    y        :: DenseVector{T},
    mask_n   :: DenseVector{Int},
    lambda   :: T;
    n        :: Int    = length(y),
    p        :: Int    = size(x,2),
    mn       :: Int    = sum(mask_n),
    d2b      :: DenseMatrix{T} = zeros(T, p,p),
    x2       :: DenseMatrix{T} = zeros(T, n,p),
    b        :: DenseVector{T} = zeros(T, p),
    b0       :: DenseVector{T} = zeros(T, p),
    ntb      :: DenseVector{T} = zeros(T, p),
    db       :: DenseVector{T} = zeros(T, p),
    Xb       :: DenseVector{T} = zeros(T, n),
    lxb      :: DenseVector{T} = zeros(T, n),
    l2xb     :: DenseVector{T} = zeros(T, n),
    tol      :: Float = convert(T, 1e-8),
    max_iter :: Int   = 50,
    quiet    :: Bool  = true,
)

    # if b is not warm-started, then ensure that it is not entirely zero
    if all(b .== 0) 
        b[1] = logit(mean(y[mask_n .== 1]))
    end

    # initialize intermediate arrays for calculations
    BLAS.gemv!('N', one(T), x, b, zero(T), Xb)
    log2xb!(lxb, l2xb, Xb, n=n)
    mask!(lxb, mask_n, 0, zero(T), n=n)
#    lxb[mask_n .== 0] = zero(T)
    copy!(b0,b)
    fill!(db, zero(T))

    # track objective
    old_obj = Inf
    new_obj = logistic_loglik(Xb,y,b,mask_n,0,lambda,p, n=n)

    # output progress to console
    quiet || println("Iter\tHalves\tObjective")

    i = 0
    bktrk = 0
    # enter loop for Newton's method
    for i = 1:max_iter

        # db = (x'*(lxb - y)) / n + lambda*b
        BLAS.axpy!(n, -one(T), sdata(y), 1, sdata(lxb), 1)
        mask!(lxb, mask_n, 0, zero(T), n=n)
#        lxb[mask_n .== 0] = zero(T)
        BLAS.gemv!('T', one(T), sdata(x), sdata(lxb), zero(T), sdata(db))
#        BLAS.scal!(p, 1/n, sdata(db), 1)
        BLAS.scal!(p, 1/mn, sdata(db), 1)
        BLAS.axpy!(p, lambda, sdata(b), 1, sdata(db), 1)

        # d2b = (x'*diagm(l2xb)*x)/n + lambda*I
        # note that log2xb!() already performs division by n on l2xb
        copy!(x2,x)
        mask!(l2xb, mask_n, 0, zero(T), n=n)
#        l2xb[mask_n .== 0] = zero(T)
        BLAS.scal!(p, n/mn, sdata(l2xb), 1) # rescale to number of unmasked samples
        scale!(sdata(l2xb), sdata(x2))
        BLAS.gemm!('T', 'N', one(T), sdata(x), sdata(x2), zero(T), sdata(d2b))
        d2b += lambda*I

        # b = b0 - ntb = b0 - inv(d2b)*db
        #   = b0 - inv[ x' diagm(pi) diagm(1 - pi) x + lambda*I] [x' (pi - y) + lambda*b]
        ntb = d2b\db
        copy!(b,b0)
        BLAS.axpy!(p,-one(T),ntb,1,b,1)

        # compute objective
        new_obj = logistic_loglik(Xb,y,b,mask_n,0,lambda,p, n=n)

        # backtrack
        j = 0
        while (new_obj > old_obj + tol) && (j < 50)

            # increment iterator
            j += 1

            # b = b0 - 0.5*ntb
            copy!(b,b0)
#            BLAS.axpy!(p,-(0.5^j),ntb,1,b,1)
            BLAS.axpy!(p,-one(T) / (2^j),ntb,1,b,1)

            # recalculate objective
            new_obj = logistic_loglik(Xb,y,b,mask_n,0,lambda,p, n=n)

        end

        # accumulate total backtracking steps
        bktrk += j

        # track distance between iterates
        dist = euclidean(sdata(b),sdata(b0)) / (norm(sdata(b0),2) + one(T))

        # track progress
        quiet || println(i, "\t", j, "\t", dist)

        # check for convergence
        # if converged, then return b
        dist < tol && return b, i, div(bktrk,i)

        # unconverged at this point, so update intermediate arrays
        BLAS.gemv!('N', one(T), sdata(x), sdata(b), zero(T), sdata(Xb))
        log2xb!(lxb, l2xb, Xb, n=n)

        # save previous beta
        copy!(b0, b)
        old_obj = new_obj
    end

    warn("fit_logistic failed to converge in $(max_iter) iterations, exiting...")
    return b, i, div(bktrk,max_iter)
end
