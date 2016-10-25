"""
    mce(lxβ, y, mask_n [, n=length(y), mn=length(mask_n)])

For statistical design matrix `x`, model `b`, and binary response `y`, compute the mean classification error between a vector `lxβ = logistic(x*b)` and `y` with potentially masked entries indicated by the `0`s of the `Int` vector `mask_n`.
"""
function mce{T <: Float}(
    lxβ    :: DenseVector{T},
    y      :: DenseVector{T},
    mask_n :: DenseVector{Int},
    mn     :: Int = sum(mask_n)
)
    s = zero(T)
    @inbounds for i in eachindex(y)
        if mask_n[i] == 1
            s += abs(round(lxβ[i]) - round(y[i]))
        end
    end
    return s / mn
end

"""
    update_x!(x, x0, y, idxs, a)

If fed an `Int` vector `idxs` that sorts `x` by `abs()`, then `x` is updated as `x[idxs[1:k]] = x0[idxs[1:k]] - a*y[idxs[1:k]]`.
The output is comparable to

    copy!(x,x0)
    BLAS.axpy!(p,-a,y,1,x,1)

but `update_x!()` exploits the implied sparsity of `x` by only updating the top `k` components in magnitude.
"""
function update_x!{T <: Float}(
    x    :: DenseVector{T},
    x0   :: DenseVector{T},
    y    :: DenseVector{T},
    idxs :: DenseVector{Int},
    a    :: T
)
    fill!(x, zero(T))
    @inbounds for i in eachindex(idxs)
        idx    = idxs[i]
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
    logistic!(y, x)

This subroutine computes `y = logistic(x)` in-place, modifying `y`."
"""
function logistic!{T <: Float}(
	y :: DenseVecOrMat{T},
	x :: DenseVecOrMat{T}
)
    @assert size(y) == size(x)
    @inbounds for i in eachindex(x)
		y[i] = one(T) / (one(T) + exp(-x[i]))
	end
	return nothing
end

"""
    logistic_loglik(xβ, y, b, idxs, λ, k)

If fed a vector `idxs` of indices that sort `b`, then `logistic_loglik` computes the loglikelihood over the `k` active components of `b`.
"""
function logistic_loglik{T <: Float}(
	xβ   :: DenseVector{T},
    y    :: DenseVector{T},
	β    :: DenseVector{T},
	idxs :: DenseVector{Int},
	λ    :: T,
    k    :: Int
)
    n = length(xβ)
	s = zero(T)
	@inbounds for i = 1:n
		s += log(one(T) + exp(xβ[i])) - xβ[i]*y[i]
	end
    s /= n
	@inbounds for i = 1:k
		idx = idxs[i]
        s  += λ*β[idx]*β[idx] / 2
	end
	return s
end

function logistic_loglik{T <: Float}(
	xβ   :: DenseVector{T},
    y    :: DenseVector{T},
	β    :: DenseVector{T},
	idxs :: BitArray{1},
	λ    :: T,
    k    :: Int
)
    n = length(xβ)
	s = zero(T)
	@inbounds for i = 1:n
		s += log(one(T) + exp(xβ[i])) - xβ[i]*y[i]
	end
    s /= n
    ntrue = 0
    @inbounds for i in eachindex(idxs) 
		idx = idxs[i]
        if idx
            ntrue += 1
            s  += λ*β[idx]*β[idx] / 2
        end
        ntrue >= k && break
	end
	return s
end

"""
    logistic_loglik(xβ, y, b, λ)

Compute the regularized negative logistic loglikelihood

   ( sum(log(1 + exp(x*β))) - y'*x*β) / n + 0.5*λ*sumabs2(β)

The unregularized loglikelihood is computed by setting `λ = 0.0`.

Arguments:

- `xβ = x*β`. Note that this must be precomputed; the function does not compute `x*b`.
- `y` is the response vector of `0`s and `1`s.
- `β` is the statistical model.
- `λ` is the regularization parameter.
"""
function logistic_loglik{T <: Float}(
	xβ :: DenseVector{T},
    y  :: DenseVector{T},
	β  :: DenseVector{T},
	λ  :: T
)
    n = length(xβ)
	s = zero(T)
	@inbounds for i = 1:n
		s += log(one(T) + exp(xβ[i])) - xβ[i]*y[i]
	end
    s /= n
    @inbounds for i in eachindex(β)
        s += λ*β[i]*β[i] / 2
	end
	return s
end


function logistic_loglik{T <: Float}(
	xβ     :: DenseVector{T},
    y      :: DenseVector{T},
	β      :: DenseVector{T},
    mask_n :: DenseVector{Int},
    m_val  :: Int,
	λ      :: T,
	k      :: Int;
    mn     :: Int = sum(mask_n),
)
	n = length(xβ),
	s = zero(T)
	@inbounds for i = 1:n
        if mask_n[i] != m_val
            s += log(one(T) + exp(xβ[i])) - xβ[i]*y[i]
        end
	end
#    s /= n
    s /= mn
	@inbounds for i = 1:k
        s += λ*β[i]*β[i] / 2
	end
	return s
end


function logistic_loglik{T <: Float}(
	xβ      :: DenseVector{T},
    y       :: DenseVector{T},
	b       :: DenseVector{T},
    idxs    :: DenseVector{Int},
    mask_n  :: DenseVector{Int},
    m_val   :: Int,
	λ  :: T,
	k       :: Int;
	n       :: Int = length(xb),
    mn      :: Int = sum(mask_n),
)
	s = zero(T)
	@inbounds for i = 1:n
        if mask_n[i] != m_val
            s += log(one(T) + exp(xβ[i])) - xβ[i]*y[i]
        end
	end
#    s /= n
    s /= mn
	@inbounds for i = 1:k
        idx = idxs[i]
        s += λ*b[idx]*b[idx] / 2
	end
	return s
end

function logistic_loglik{T <: Float}(
	xβ      :: DenseVector{T},
    y       :: DenseVector{T},
	b       :: DenseVector{T},
    idxs    :: BitArray{1},
    mask_n  :: DenseVector{Int},
    m_val   :: Int,
	λ  :: T,
	k       :: Int;
	n       :: Int = length(xb),
    mn      :: Int = sum(mask_n),
)
	s = zero(T)
	@inbounds for i = 1:n
        if mask_n[i] != m_val
            s += log(one(T) + exp(xβ[i])) - xβ[i]*y[i]
        end
	end
#    s /= n
    s /= mn
    nk = 0
	@inbounds for i = 1:length(idxs)
        idx = idxs[i]
        if idx
            s += λ*b[idx]*b[idx] / 2
            nk += 1
        end
        nk >= k && break
	end
	return s
end

"""
    logistic_grad!(df, lxβ, x, y, b, xβ, idxs, k, λ)

Compute the *partial* gradient of the regularized negative logistic loglikelihood as indexed by the `Int` vector `idxs`.
"""
function logistic_grad!{T <: Float}(
    df   :: DenseVector{T},
    lxβ  :: DenseVector{T},
    x    :: DenseMatrix{T},
    y    :: DenseVector{T},
    b    :: DenseVector{T},
    xβ   :: DenseVector{T},
    idxs :: DenseVector{Int},
    k    :: Int,
    λ    :: T;
)
    n = length(xβ)
    logistic!(lxβ,xβ)
    fill!(df, zero(T))
    @inbounds for i = 1:k
        idx = idxs[i]
#        df[idx] = zero(T)
        @inbounds for j = 1:n
            df[idx] += x[j,idx] * (lxβ[j] - y[j])
        end
        df[idx] /= n
        df[idx] += λ*b[idx]
    end
    return nothing
end

function logistic_grad!{T <: Float}(
    df   :: DenseVector{T},
    lxβ  :: DenseVector{T},
    x    :: DenseMatrix{T},
    y    :: DenseVector{T},
    b    :: DenseVector{T},
    xβ   :: DenseVector{T},
    idxs :: BitArray{1}, 
    k    :: Int,
    λ    :: T;
)
    n = length(xβ)
    logistic!(lxβ,xβ)
    fill!(df, zero(T))
    ntrue = 0
    @inbounds for i in eachindex(idxs) 
        idx = idxs[i]
#        df[idx] = zero(T)
        if idx
            ntrue += 1
            @inbounds for j = 1:n
                df[idx] += x[j,idx] * (lxβ[j] - y[j])
            end
            df[idx] /= n
            df[idx] += λ*b[idx]
        end
        ntrue >= k && break
    end
    return nothing
end

### 18 OCT 2016:
### must redo logistic_grad! for revamped BEDFiles

"""
    logistic_grad!(df, lxβ, x::BEDFile, y, b, xβ, means, invstds, idxs, mask_n, k, λ [, n=length(xβ)])

Compute the *partial* gradient of the regularized negative logistic loglikelihood, as indexed by the `Int` vector `idxs`, with a `BEDFile` object `x`.
"""
function logistic_grad!{T <: Float}(
    df      :: DenseVector{T},
    lxβ     :: DenseVector{T},
    x       :: BEDFile,
    y       :: DenseVector{T},
    b       :: DenseVector{T},
    xβ      :: DenseVector{T},
    means   :: DenseVector{T},
    invstds :: DenseVector{T},
    idxs    :: DenseVector{Int},
    mask_n  :: DenseVector{Int},
    k       :: Int,
    λ  :: T;
    n       :: Int = length(xβ),
    mn      :: Int = sum(mask_n),
)
    logistic!(lxβ,xβ,n=n)
    fill!(df, zero(T))
    @inbounds for i = 1:k
        idx = idxs[i]
        m   = means[idx]
        d   = invstds[idx]
#        df[idx] = zero(T)
        @inbounds for j = 1:n
            if mask_n[i] == 1
                df[idx] += (x[j,idx] - m) * d * (lxβ[j] - y[j])
            end
        end
#        df[idx] /= n
        df[idx] /= mn
        df[idx] += λ*b[idx]
    end
    return nothing
end


"""
    logistic_grad!(df, lxβ, x, y, β, xβ, λ)

Computes the full gradient of the negative logistic loglikelihood.
"""
function logistic_grad!{T <: Float}(
    df  :: DenseVector{T},
    lxβ :: DenseVector{T},
    x   :: DenseMatrix{T},
    y   :: DenseVector{T},
    β   :: DenseVector{T},
    xβ  :: DenseVector{T},
    λ   :: T;
)
    n = length(y)
    logistic!(lxβ, xβ)
    BLAS.axpy!(-one(T), y, lxβ)
    At_mul_B!(df, x, lxβ)
    scale!(df, 1/n)
    BLAS.axpy!(λ, β, df)
    return nothing
end

"""
    logistic_grad!(df, lxβ, x::BEDFile, y, b, xβ, means, invstds, λ, [, n=length(xβ), p=length(df)])

Computes the full gradient of the negative logistic loglikelihood with a `BEDFile` object `x`.
"""
function logistic_grad!{T <: Float}(
    df      :: SharedVector{T},
    lxβ     :: SharedVector{T},
    x       :: BEDFile,
    y       :: SharedVector{T},
    b       :: SharedVector{T},
    xβ      :: SharedVector{T},
    means   :: SharedVector{T},
    invstds :: SharedVector{T},
    mask_n  :: DenseVector{Int},
    λ  :: T;
    pids    :: DenseVector{Int} = procs(),
    n       :: Int = length(xβ),
    p       :: Int = length(df),
    mn      :: Int = sum(mask_n),
)
    logistic!(lxβ,xβ,n=n)
    BLAS.axpy!(n,-one(T),sdata(y),1,sdata(lxβ),1)
    xty!(df,x,lxβ,mask_n, means=means, invstds=invstds, pids=pids)
#    BLAS.scal!(p,1/n,sdata(df),1)
    BLAS.scal!(p,1/mn,sdata(df),1)
    BLAS.axpy!(p,λ,sdata(b),1,sdata(df),1)
    return nothing
end


"""
    log2xb!(lxβ, l2xb, xβ [, n=length(xβ)])

Compute `lxβ = logistic(xβ)` and `l2xb = lxβ*(1 - lxβ)` in-place and in one pass, overwriting `lxβ` and `l2xb`.
"""
function log2xb!{T <: Float}(
    lxβ  :: DenseVector{T},
    l2xb :: DenseVector{T},
    xβ   :: DenseVector{T};
    n    :: Int = length(xβ)
)
    @inbounds for i = 1:n
        lxβ[i]  = one(T) / (one(T) + exp(-xβ[i]))
        l2xb[i] = lxβ[i]*(one(T) - lxβ[i]) / n
    end
    return nothing
end
