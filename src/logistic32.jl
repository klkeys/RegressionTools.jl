function mce(
    lxb    :: DenseVector{Float32},
    y      :: DenseVector{Float32},
    mask_n :: DenseVector{Int}; 
    n      :: Int = length(y), 
    mn     :: Int = sum(mask_n)
)
    s = zero(Float32)
    @inbounds for i = 1:n
        if mask_n[i] == 1
            s += abs(round(lxb[i]) - round(y[i]))
        end
    end
    return s / mn
end

function update_x!(
    x     :: DenseVector{Float32},
    x0    :: DenseVector{Float32},
    y     :: DenseVector{Float32},
    idxs  :: DenseVector{Int},
    a     :: Float32;
    k     :: Int = length(idxs)
)
    fill!(x, zero(Float32)) 
    @inbounds for i = 1:k
        idx = idxs[i]
        x[idx] = x0[idx] - a*y[idx]
    end
    return nothing
end


function df_norm(
    df   :: DenseVector{Float32},
    idxs :: DenseVector{Int},
    k0   :: Int,
    k    :: Int
)
    s = zero(Float32)
    @inbounds for i = k0:k
        idx = idxs[i]
        s += df[idx]*df[idx]
    end
    return sqrt(s)
end

function logistic!(
	y :: DenseVector{Float32},
	x :: DenseVector{Float32};
	n :: Int = length(y)
)
	n == length(x) || throw(ArgumentError("length(y) != length(x)"))
	@inbounds for i = 1:n
		y[i] = one(Float32) / (one(Float32) + exp(-x[i]))
	end
	return nothing
end

function logistic_loglik(
	Xb      :: DenseVector{Float32},
    y       :: DenseVector{Float32},
	b       :: DenseVector{Float32},
	sortidx :: DenseVector{Int},
	lambda  :: Float32,
	k       :: Int;
	n       :: Int = length(Xb)
)
	s = zero(Float32)
	@inbounds for i = 1:n
		s += log(one(Float32) + exp(Xb[i])) - Xb[i]*y[i]
	end
    s /= n
	@inbounds for i = 1:k
		idx = sortidx[i]
        s  += 0.5f0*lambda*b[idx]*b[idx]
	end
	return s
end

function logistic_loglik(
	Xb      :: DenseVector{Float32},
    y       :: DenseVector{Float32},
	b       :: DenseVector{Float32},
	lambda  :: Float32,
	k       :: Int;
	n       :: Int = length(xb)
)
	s = zero(Float32)
	@inbounds for i = 1:n
		s += log(one(Float32) + exp(Xb[i])) - Xb[i]*y[i]
	end
    s /= n
	@inbounds for i = 1:k
        s += 0.5f0*lambda*b[i]*b[i]
	end
	return s
end


function logistic_loglik(
	Xb      :: DenseVector{Float32},
    y       :: DenseVector{Float32},
	b       :: DenseVector{Float32},
    mask_n  :: DenseVector{Int},
    m_val   :: Int,
	lambda  :: Float32,
	k       :: Int;
	n       :: Int = length(xb),
    mn      :: Int = sum(mask_n),
)
	s = zero(Float32)
	@inbounds for i = 1:n
        if mask_n[i] != m_val 
            s += log(one(Float32) + exp(Xb[i])) - Xb[i]*y[i]
        end
	end
#    s /= n
    s /= mn
	@inbounds for i = 1:k
        s += 0.5f0*lambda*b[i]*b[i]
	end
	return s
end


function logistic_loglik(
	Xb      :: DenseVector{Float32},
    y       :: DenseVector{Float32},
	b       :: DenseVector{Float32},
    idxs    :: DenseVector{Int},
    mask_n  :: DenseVector{Int},
    m_val   :: Int,
	lambda  :: Float32,
	k       :: Int;
	n       :: Int = length(xb),
    mn      :: Int = sum(mask_n),
)
	s = zero(Float32)
	@inbounds for i = 1:n
        if mask_n[i] != m_val 
            s += log(one(Float32) + exp(Xb[i])) - Xb[i]*y[i]
        end
	end
#    s /= n
    s /= mn
	@inbounds for i = 1:k
        idx = idxs[i]
        s += 0.5f0*lambda*b[idx]*b[idx]
	end
	return s
end

function logistic_grad!(
    df     :: DenseVector{Float32}, 
    lxb    :: DenseVector{Float32}, 
    x      :: DenseMatrix{Float32},
    y      :: DenseVector{Float32},
    b      :: DenseVector{Float32},
    Xb     :: DenseVector{Float32},
    idxs   :: DenseVector{Int},
    k      :: Int,
    lambda :: Float32;
    n      :: Int = length(Xb)
)
    logistic!(lxb,Xb,n=n)
    @inbounds for i = 1:k
        idx = idxs[i]
        df[idx] = zero(Float32)
        @inbounds for j = 1:n
            df[idx] += x[j,idx] * (lxb[j] - y[j])
        end
        df[idx] /= n
        df[idx] += lambda*b[idx]
    end
    return nothing
end

function logistic_grad!(
    df      :: DenseVector{Float32}, 
    lxb     :: DenseVector{Float32}, 
    x       :: BEDFile, 
    y       :: DenseVector{Float32},
    b       :: DenseVector{Float32},
    Xb      :: DenseVector{Float32},
    means   :: DenseVector{Float32},
    invstds :: DenseVector{Float32},
    idxs    :: DenseVector{Int},
    mask_n  :: DenseVector{Int},
    k       :: Int,
    lambda  :: Float32;
    n       :: Int = length(Xb),
    mn      :: Int = sum(mask_n),
)
    logistic!(lxb,Xb,n=n)
    @inbounds for i = 1:k
        idx = idxs[i]
        m   = means[idx]
        d   = invstds[idx]
        df[idx] = zero(Float32)
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


function logistic_grad!(
    df     :: DenseVector{Float32}, 
    lxb    :: DenseVector{Float32}, 
    x      :: DenseMatrix{Float32},
    y      :: DenseVector{Float32},
    b      :: DenseVector{Float32},
    Xb     :: DenseVector{Float32},
    lambda :: Float32;
    n      :: Int = length(Xb),
    p      :: Int = length(df),
)
    logistic!(lxb,Xb,n=n)
    BLAS.axpy!(n,-one(Float32),y,1,lxb,1)
    BLAS.gemv!('T',one(Float32),x,lxb,zero(Float32),df)
    BLAS.scal!(p,1/n,df,1)
    BLAS.axpy!(p,lambda,b,1,df,1)
    return nothing
end

function logistic_grad!(
    df      :: SharedVector{Float32}, 
    lxb     :: SharedVector{Float32}, 
    x       :: BEDFile, 
    y       :: SharedVector{Float32},
    b       :: SharedVector{Float32},
    Xb      :: SharedVector{Float32},
    means   :: SharedVector{Float32},
    invstds :: SharedVector{Float32},
    mask_n  :: DenseVector{Int},
    lambda  :: Float32;
    pids    :: DenseVector{Int} = procs(),
    n       :: Int = length(Xb),
    p       :: Int = length(df),
    mn      :: Int = sum(mask_n),
)
    logistic!(lxb,Xb,n=n)
    BLAS.axpy!(n,-one(Float32),sdata(y),1,sdata(lxb),1)
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
function log2xb!(
    lxb  :: DenseVector{Float32}, 
    l2xb :: DenseVector{Float32}, 
    Xb   :: DenseVector{Float32}; 
    n    :: Int = length(Xb)
)
    @inbounds for i = 1:n 
        lxb[i]  = one(Float32) / (one(Float32) + exp(-Xb[i])) 
        l2xb[i] = lxb[i]*(one(Float32) - lxb[i]) / n
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
- `tol` is the convergence tolerance. Defaults to `1f-8`.
- `max_iter` sets the maximum number of backtracking steps. Defaults to `50`.
- `quiet` controls output. Defaults to `true` (no output).
- Several temporary arrays:

    d2b  = zeros(Float32,p,p)  # Hessian of b 
    x2   = zeros(Float32,n,p)  # copy of x used for updating Hessian 
    b    = zeros(Float32,p)    # copy of bk used in refitting 
    b0   = zeros(Float32,p)    # copy of bk used in refitting 
    ntb  = zeros(Float32,p)    # Newton step for bk used in refitting 
    db   = zeros(Float32,p)    # gradient of bk used in refitting 
    Xb   = zeros(Float32,n)    # x*b 
    lxb  = zeros(Float32,n)    # logistic(x*b) = 1 ./ (1 + exp(-x*b)) 
    l2xb = zeros(Float32,n)    # lxb * (1 - lxb) 

Output: 

- a vector of refit coefficients for `b`
- the number of backtracking steps taken.
"""
function fit_logistic(
    x        :: DenseMatrix{Float32},
    y        :: DenseVector{Float32},
    lambda   :: Float32;
    n        :: Int    = length(y),
    p        :: Int    = size(x,2),
    d2b      :: DenseMatrix{Float32} = zeros(Float32, p,p),
    x2       :: DenseMatrix{Float32} = zeros(Float32, n,p),
    b        :: DenseVector{Float32} = zeros(Float32, p),
    b0       :: DenseVector{Float32} = zeros(Float32, p),
    ntb      :: DenseVector{Float32} = zeros(Float32, p),
    db       :: DenseVector{Float32} = zeros(Float32, p),
    Xb       :: DenseVector{Float32} = zeros(Float32, n),
    lxb      :: DenseVector{Float32} = zeros(Float32, n),
    l2xb     :: DenseVector{Float32} = zeros(Float32, n),
    tol      :: Float32 = 1f-8,
    max_iter :: Int     = 50,
    quiet    :: Bool    = true,
)

    # if b is not warm-started, then ensure that it is not entirely zero
    if sum(b) == zero(Float32)
        b[1] = logit(mean(y))
    end

    # initialize intermediate arrays for calculations
    BLAS.gemv!('N', one(Float32), x, b, zero(Float32), Xb)
    log2xb!(lxb, l2xb, Xb, n=n)
    copy!(b0,b)
    fill!(db, zero(Float32))

    # track objective
    old_obj = Inf32 
    new_obj = logistic_loglik(Xb,y,b,lambda,p, n=n)

    # output progress to console
    quiet || println("Iter\tHalves\tObjective")

    i = 0
    bktrk = 0
    # enter loop for Newton's method
    for i = 1:max_iter

        # db = (x'*(lxb - y)) / n + lambda*b
        BLAS.axpy!(n, -one(Float32), sdata(y), 1, sdata(lxb), 1)
        BLAS.gemv!('T', one(Float32), sdata(x), sdata(lxb), zero(Float32), sdata(db))
        BLAS.scal!(p, 1/n, sdata(db), 1)
        BLAS.axpy!(p, lambda, sdata(b), 1, sdata(db), 1)

        # d2b = (x'*diagm(l2xb)*x)/n + lambda*I 
        # note that log2xb!() already performs division by n on l2xb 
        copy!(x2,x)
        scale!(sdata(l2xb), sdata(x2))
        BLAS.gemm!('T', 'N', one(Float32), sdata(x), sdata(x2), zero(Float32), sdata(d2b))
        d2b += lambda*I

        # b = b0 - ntb = b0 - inv(d2b)*db 
        #   = b0 - inv[ x' diagm(pi) diagm(1 - pi) x + lambda*I] [x' (pi - y) + lambda*b]
        ntb = d2b\db
        copy!(b,b0)
        BLAS.axpy!(p,-one(Float32),ntb,1,b,1)

        # compute objective
        new_obj = logistic_loglik(Xb,y,b,lambda,p, n=n)

        # backtrack
        j = 0
        while (new_obj > old_obj + tol) && (j < 50)

            # increment iterator
            j += 1

            # b = b0 - 0.5f0*ntb
            copy!(b,b0)
            BLAS.axpy!(p,-(0.5f0^j),ntb,1,b,1)

            # recalculate objective
            new_obj = logistic_loglik(Xb,y,b,lambda,p, n=n)
        end

        # accumulate total backtracking steps
        bktrk += j

        # track distance between iterates
        dist = euclidean(sdata(b),sdata(b0)) / (norm(sdata(b0),2) + one(Float32))

        # track progress
        quiet || println(i, "\t", j, "\t", dist)

        # check for convergence
        # if converged, then return b
        dist < tol && return b, i, div(bktrk,i)

        # unconverged at this point, so update intermediate arrays
        BLAS.gemv!('N', one(Float32), sdata(x), sdata(b), zero(Float32), sdata(Xb))
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
function fit_logistic(
    x        :: DenseMatrix{Float32},
    y        :: DenseVector{Float32},
    mask_n   :: DenseVector{Int},
    lambda   :: Float32;
    n        :: Int    = length(y),
    p        :: Int    = size(x,2),
    mn       :: Int    = sum(mask_n),
    d2b      :: DenseMatrix{Float32} = zeros(Float32, p,p),
    x2       :: DenseMatrix{Float32} = zeros(Float32, n,p),
    b        :: DenseVector{Float32} = zeros(Float32, p),
    b0       :: DenseVector{Float32} = zeros(Float32, p),
    ntb      :: DenseVector{Float32} = zeros(Float32, p),
    db       :: DenseVector{Float32} = zeros(Float32, p),
    Xb       :: DenseVector{Float32} = zeros(Float32, n),
    lxb      :: DenseVector{Float32} = zeros(Float32, n),
    l2xb     :: DenseVector{Float32} = zeros(Float32, n),
    tol      :: Float32 = 1f-8,
    max_iter :: Int     = 50,
    quiet    :: Bool    = true,
)

    # if b is not warm-started, then ensure that it is not entirely zero
    if sum(b) == zero(Float32)
        b[1] = logit(mean(y[mask_n .== 1]))
    end

    # initialize intermediate arrays for calculations
    BLAS.gemv!('N', one(Float32), x, b, zero(Float32), Xb)
    log2xb!(lxb, l2xb, Xb, n=n)
#    mask!(lxb, mask_n, 0, zero(Float32), n=n)
    lxb[mask_n .== 0] = zero(Float32) 
    copy!(b0,b)
    fill!(db, zero(Float32))

    # track objective
    old_obj = Inf32 
    new_obj = logistic_loglik(Xb,y,b,mask_n,0,lambda,p, n=n)

    # output progress to console
    quiet || println("Iter\tHalves\tObjective")

    i = 0
    bktrk = 0
    # enter loop for Newton's method
    for i = 1:max_iter

        # db = (x'*(lxb - y)) / n + lambda*b
        BLAS.axpy!(n, -one(Float32), sdata(y), 1, sdata(lxb), 1)
#        mask!(lxb, mask_n, 0, zero(Float32), n=n)
        lxb[mask_n .== 0] = zero(Float32) 
        BLAS.gemv!('T', one(Float32), sdata(x), sdata(lxb), zero(Float32), sdata(db))
#        BLAS.scal!(p, 1/n, sdata(db), 1)
        BLAS.scal!(p, 1/mn, sdata(db), 1)
        BLAS.axpy!(p, lambda, sdata(b), 1, sdata(db), 1)

        # d2b = (x'*diagm(l2xb)*x)/n + lambda*I 
        # note that log2xb!() already performs division by n on l2xb 
        copy!(x2,x)
#        mask!(l2xb, mask_n, 0, zero(Float32), n=n)
        l2xb[mask_n .== 0] = zero(Float32) 
        BLAS.scal!(p, n/mn, sdata(l2xb), 1) # rescale to number of unmasked samples
        scale!(sdata(l2xb), sdata(x2))
        BLAS.gemm!('T', 'N', one(Float32), sdata(x), sdata(x2), zero(Float32), sdata(d2b))
        d2b += lambda*I

        # b = b0 - ntb = b0 - inv(d2b)*db 
        #   = b0 - inv[ x' diagm(pi) diagm(1 - pi) x + lambda*I] [x' (pi - y) + lambda*b]
        ntb = d2b\db
        copy!(b,b0)
        BLAS.axpy!(p,-one(Float32),ntb,1,b,1)

        # compute objective
        new_obj = logistic_loglik(Xb,y,b,mask_n,0,lambda,p, n=n)

        # backtrack
        j = 0
        while (new_obj > old_obj + tol) && (j < 50)

            # increment iterator
            j += 1

            # b = b0 - 0.5*ntb
            copy!(b,b0)
            BLAS.axpy!(p,-(0.5f0^j),ntb,1,b,1)

            # recalculate objective
            new_obj = logistic_loglik(Xb,y,b,mask_n,0,lambda,p, n=n)

        end

        # accumulate total backtracking steps
        bktrk += j

        # track distance between iterates
        dist = euclidean(sdata(b),sdata(b0)) / (norm(sdata(b0),2) + one(Float32))

        # track progress
        quiet || println(i, "\t", j, "\t", dist)

        # check for convergence
        # if converged, then return b
        dist < tol && return b, i, div(bktrk,i)

        # unconverged at this point, so update intermediate arrays
        BLAS.gemv!('N', one(Float32), sdata(x), sdata(b), zero(Float32), sdata(Xb))
        log2xb!(lxb, l2xb, Xb, n=n)

        # save previous beta
        copy!(b0, b)
        old_obj = new_obj
    end

    warn("fit_logistic failed to converge in $(max_iter) iterations, exiting...")
    return b, i, div(bktrk,max_iter)
end
