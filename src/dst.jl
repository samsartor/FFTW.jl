# (This is part of the FFTW module.)

"""
    plan_dst!(A [, dims [, flags [, timelimit [, num_threads]]]])

Same as [`plan_dst`](@ref), but operates in-place on `A`.
"""
function plan_dst! end

"""
    plan_idst(A [, dims [, flags [, timelimit [, num_threads]]]])

Pre-plan an optimized inverse discrete sine transform (DST), similar to
[`plan_fft`](@ref) except producing a function that computes
[`idst`](@ref). The first two arguments have the same meaning as for
[`idst`](@ref).
"""
function plan_idst end

"""
    plan_dst(A [, dims [, flags [, timelimit [, num_threads]]]])

Pre-plan an optimized discrete sine transform (DST), similar to
[`plan_fft`](@ref) except producing a function that computes
[`dst`](@ref). The first two arguments have the same meaning as for
[`dst`](@ref).
"""
function plan_dst end

"""
    plan_idst!(A [, dims [, flags [, timelimit [, num_threads]]]])

Same as [`plan_idst`](@ref), but operates in-place on `A`.
"""
function plan_idst! end

"""
    dst(A [, dims])

Performs a multidimensional type-II discrete sine transform (DST) of the array `A`, using
the unitary normalization of the DST. The optional `dims` argument specifies an iterable
subset of dimensions (e.g. an integer, range, tuple, or array) to transform along.  Most
efficient if the size of `A` along the transformed dimensions is a product of small primes;
see [`nextprod`](@ref). See also [`plan_dst`](@ref) for even greater
efficiency.
"""
function dst end

"""
    idst(A [, dims])

Computes the multidimensional inverse discrete sine transform (DST) of the array `A`
(technically, a type-III DST with the unitary normalization). The optional `dims` argument
specifies an iterable subset of dimensions (e.g. an integer, range, tuple, or array) to
transform along.  Most efficient if the size of `A` along the transformed dimensions is a
product of small primes; see [`nextprod`](@ref).  See also
[`plan_idst`](@ref) for even greater efficiency.
"""
function idst end

"""
    dst!(A [, dims])

Same as [`dst`](@ref), except that it operates in-place on `A`, which must be an
array of real or complex floating-point values.
"""
function dst! end

"""
    idst!(A [, dims])

Same as [`idst`](@ref), but operates in-place on `A`.
"""
function idst! end

# Discrete sine transforms (type II/III) via FFTW's r2r transforms;
# we follow the Matlab convention and adopt a unitary normalization here.
# Unlike Matlab we compute the multidimensional transform by default,
# similar to the Julia fft functions.

mutable struct DSTPlan{T<:fftwNumber,K,inplace} <: Plan{T}
    plan::r2rFFTWPlan{T}
    r::Array{UnitRange{Int}} # array of indices for rescaling
    nrm::Float64 # normalization factor
    region::Dims # dimensions being transformed
    pinv::DSTPlan{T}
    DSTPlan{T,K,inplace}(plan,r,nrm,region) where {T<:fftwNumber,K,inplace} = new(plan,r,nrm,region)
end

size(p::DSTPlan) = size(p.plan)

function show(io::IO, p::DSTPlan{T,K,inplace}) where {T,K,inplace}
    print(io, inplace ? "FFTW in-place " : "FFTW ",
          K == RODFT10 ? "DST (DST-II)" : "IDST (DST-III)", " plan for ")
    showfftdims(io, p.plan.sz, p.plan.istride, eltype(p))
end

for (pf, pfr, K, inplace) in ((:plan_dst, :plan_r2r, RODFT10, false),
                              (:plan_dst!, :plan_r2r!, RODFT10, true),
                              (:plan_idst, :plan_r2r, RODFT01, false),
                              (:plan_idst!, :plan_r2r!, RODFT01, true))
    @eval function $pf(X::StridedArray{T}, region; kws...) where T<:fftwNumber
        r = [1:n for n in size(X)]
        nrm = sqrt(0.5^length(region) * normalization(X,region))
        DSTPlan{T,$K,$inplace}($pfr(X, $K, region; kws...), r, nrm,
                               ntuple(i -> Int(region[i]), length(region)))
    end
end

function plan_inv(p::DSTPlan{T,K,inplace}) where {T,K,inplace}
    X = Array{T}(undef, p.plan.sz)
    iK = inv_kind[K]
    DSTPlan{T,iK,inplace}(inplace ?
                          plan_r2r!(X, iK, p.region, flags=p.plan.flags) :
                          plan_r2r(X, iK, p.region, flags=p.plan.flags),
                          p.r, p.nrm, p.region)
end

for f in (:dst, :dst!, :idst, :idst!)
    pf = Symbol("plan_", f)
    @eval begin
        $f(x::AbstractArray{<:fftwNumber}) = $pf(x) * x
        $f(x::AbstractArray{<:fftwNumber}, region) = $pf(x, region) * x
        $pf(x::AbstractArray; kws...) = $pf(x, 1:ndims(x); kws...)
        $f(x::AbstractArray{<:Real}, region=1:ndims(x)) = $f(fftwfloat(x), region)
        $pf(x::AbstractArray{<:Real}, region; kws...) = $pf(fftwfloat(x), region; kws...)
        $pf(x::AbstractArray{<:Complex}, region; kws...) = $pf(fftwcomplex(x), region; kws...)
    end
end

const sqrthalf = sqrt(0.5)
const sqrt2 = sqrt(2.0)
const onerange = 1:1

function mul!(y::StridedArray{T}, p::DSTPlan{T,RODFT10}, x::StridedArray{T}) where T
    assert_applicable(p.plan, x, y)
    unsafe_execute!(p.plan, x, y)
    rmul!(y, p.nrm)
    r = p.r
    for d in p.region
        oldr = r[d]
        r[d] = onerange
        y[r...] *= sqrthalf
        r[d] = oldr
    end
    return y
end

# note: idst changes input data
function mul!(y::StridedArray{T}, p::DSTPlan{T,RODFT01}, x::StridedArray{T}) where T
    assert_applicable(p.plan, x, y)
    rmul!(x, p.nrm)
    r = p.r
    for d in p.region
        oldr = r[d]
        r[d] = onerange
        x[r...] *= sqrt2
        r[d] = oldr
    end
    unsafe_execute!(p.plan, x, y)
    return y
end

*(p::DSTPlan{T,RODFT10,false}, x::StridedArray{T}) where {T} =
    mul!(Array{T}(undef, p.plan.osz), p, x)

*(p::DSTPlan{T,RODFT01,false}, x::StridedArray{T}) where {T} =
    mul!(Array{T}(undef, p.plan.osz), p, copy(x)) # need copy to preserve input

*(p::DSTPlan{T,K,true}, x::StridedArray{T}) where {T,K} = mul!(x, p, x)

