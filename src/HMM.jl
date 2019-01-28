module HMM

using Distributions
using Random
using Combinatorics
using LinearAlgebra
using GCTools
using Printf

struct Digraph
    from::Array{Int}
    to::Array{Int}
end

Base.copy( g::Digraph ) = Digraph( copy( g.from ), copy( g.to ) )

function addedge( g::Digraph, from::Int, to::Int )
    push!( g.from, from )
    push!( g.to, to )
end

fullyconnected( n::Int ) = Digraph( vcat( [collect(1:n) for i in 1:n]... ), vcat( [fill(i,n) for i in 1:n]... ) )

mutable struct DirtyArray{T,N}
    data::Array{T,N}
    dirty::Bool
end

DirtyArray( a::Array{T,N} ) where {T,N} = DirtyArray{T,N}( a, true )

Base.copy( da::DirtyArray ) = DirtyArray( copy( da.data ), da.dirty )

const DirtyVector{T} = Union{DirtyArray{T,1},Nothing} where {T}
const DirtyMatrix{T} = Union{DirtyArray{T,2},Nothing} where {T}

struct Interval{T<:Number}
    lo::T
    hi::T
end

mutable struct GaussianHMM{Calc <: Real, Out <: Real}
    graph::Digraph
    initialprobabilities::Vector{Calc}
    transitionprobabilities::Matrix{Calc}
    means::Vector{Out}
    stds::Vector{Out}

    alpha::DirtyMatrix{Calc}
    beta::DirtyMatrix{Calc}
    gamma::DirtyMatrix{Calc}
    xi::Union{DirtyArray{Calc,3},Nothing}
    b::DirtyMatrix{Calc}
    likelihood::Calc
    y::Union{Vector{Out},Vector{Interval{Out}},Nothing}
    
    scratch::Dict{Symbol,Any}
end

GaussianHMM( g::Digraph, pi::Vector{Calc}, a::Matrix{Calc}, mu::Vector{Out}, sigma::Vector{Out};
                      scratch::Dict{Symbol,Any} = Dict{Symbol,Any}(), ) where {Calc, Out} =
                          GaussianHMM{Calc, Out}( g, pi, a, mu, sigma,
                                                  nothing, nothing, nothing, nothing, nothing, convert( Calc, NaN ), nothing,
                                                  scratch )

Base.copy( hmm::GaussianHMM ) =
    GaussianHMM(
        copy( hmm.graph ),
        copy( hmm.initialprobabilities ), copy( hmm.transitionprobabilities ),
        copy( hmm.means ), copy( hmm.stds ),
        copy( hmm.alpha ), copy( hmm.beta ), copy( hmm.gamma ), copy( hmm.xi ), copy( hmm.b ), hmm.likelihood, copy( hmm.y ),
        copy( hmm.scratch ),
    )

function randomhmm( g::Digraph; calc::DataType = Float64, out::DataType = Float64, seed::Int = 1 )
    Random.seed!( seed )
    
    numstates = max( maximum( g.from ), maximum( g.to ) )
    initialprobabilities = Vector{calc}(rand( numstates ))
    initialprobabilities ./= sum( initialprobabilities )
    transitionprobabilities = Matrix{calc}(rand( numstates, numstates ))
    transitionprobabilities ./= sum( transitionprobabilities, dims=2 )
    means = Vector{out}(randn( numstates ))
    stds = Vector{out}(randn( numstates ).^2)
    scratch = Dict{Symbol,Any}()
    scratch[:seed] = seed
    return GaussianHMM( g, initialprobabilities, transitionprobabilities, means, stds, scratch=scratch )
end

function Base.rand( hmm::GaussianHMM, n::Int )
    cdfs = [ cumsum( hmm.transitionprobabilities[i,:] ) for i in 1:length(hmm.initialprobabilities) ]
    
    state = searchsorted( cumsum( hmm.initialprobabilities ), rand() ).start
    observations = [rand( Normal(hmm.means[state], hmm.stds[state]) )]
    
    for t = 2:n
        state = searchsorted( cdfs[state], rand() ).start
        push!( observations, rand( Normal(hmm.means[state], hmm.stds[state]) ) )
    end
    return observations
end

function clearscratch( hmm::GaussianHMM{Calc} ) where {Calc}
    if hmm.alpha != nothing
        hmm.alpha.dirty = true
    end
    if hmm.beta != nothing
        hmm.beta.dirty = true
    end
    if hmm.gamma != nothing
        hmm.gamma.dirty = true
    end
    if hmm.xi != nothing
        hmm.xi.dirty = true
    end
    if hmm.b != nothing
        hmm.b.dirty = true
    end
    hmm.likelihood = convert( Calc, NaN )
end

function writearray( io::IO, v::Array{T,N} ) where {T,N}
    for i in size(v)
        Base.write( io, i )
    end
    for x in v
        Base.write( io, x )
    end
end

function readarray( io::IO, ::Type{Array{T,N}} ) where {T,N}
    size = Int[]
    for i in 1:N
        push!( size, Base.read( io, Int ) )
    end
    v = zeros( T, size... )
    for i = 1:prod(size)
        v[i] = Base.read( io, T )
    end
    return v
end

function Base.write( io::IO, hmm::GaussianHMM )
    writearray( io, hmm.graph.from )
    writearray( io, hmm.graph.to )
    writearray( io, hmm.initialprobabilities )
    writearray( io, hmm.transitionprobabilities )
    writearray( io, hmm.means )
    writearray( io, hmm.stds )
    writearray( io, hmm.y )
end

function Base.read( io::IO, ::Type{GaussianHMM{Calc,Out}} ) where {Calc,Out}
    from = readarray( io, Vector{Int} )
    to = readarray( io, Vector{Int} )
    graph = Digraph( from, to )
    initialprobabilities = readarray( io, Vector{Calc} )
    transitionprobabilities = readarray( io, Matrix{Calc} )
    means = readarray( io, Vector{Out} )
    stds = readarray( io, Vector{Out} )
    y = readarray( io, Vector{Out} )
    hmm = GaussianHMM( graph, initialprobabilities, transitionprobabilities,
                       means, stds,
                       nothing, nothing, nothing, nothing, nothing, convert( Calc, NaN ), nothing, Dict{Symbol,Any}() )
    setobservations( hmm, y )
    return hmm
end

function setobservations( hmm::GaussianHMM{Calc}, y::Union{Vector{U},Vector{Interval{U}}} ) where {Calc, U <: Real}
    T = length(y)
    m = length(hmm.initialprobabilities)
    
    hmm.alpha = DirtyArray( zeros( Calc, T, m ) )
    hmm.beta = DirtyArray( zeros( Calc, T, m ) )
    hmm.gamma = DirtyArray( zeros( Calc, T, m ) )
    hmm.xi = DirtyArray( zeros( Calc, T-1, m, m ) )
    hmm.b = DirtyArray( zeros( Calc, T, m ) )
    
    clearscratch( hmm )
    hmm.y = y
end

function observations( hmm::GaussianHMM )
    if hmm.y == nothing
        error( "Need to set observations in order to perform calculation" )
    end
    return hmm.y
end

probability( d::Distribution, x::N ) where {N <: Number} = pdf( d, x )
   
probability( d::Distribution, i::Interval ) = cdf( d, i.hi ) - cdf( d, i.lo )

function probability( hmm::GaussianHMM )
    GCTools.push!(:probability)
    if hmm.b.dirty
        y = observations( hmm )
        for i in 1:length(hmm.initialprobabilities)
            hmm.b.data[:,i] = pdf.( Normal( hmm.means[i], hmm.stds[i] ), y )
        end

        hmm.b.dirty = false
    end
    GCTools.pop!()
    return hmm.b.data
end

function stationary( hmm::GaussianHMM )
    P = Matrix{Float64}( hmm.transitionprobabilities )
    N = length(hmm.initialprobabilities)
    I = one(P)
    P -= I
    P[:,1] = ones(N)
    return I[1,:]'*pinv(P)
end

function forwardprobabilities( hmm::GaussianHMM )
    GCTools.push!(:forwardprobabilities)
    if hmm.alpha.dirty
        y = observations( hmm )
        N = length(hmm.initialprobabilities)
        b = probability( hmm )
                                   
        hmm.alpha.data[1,:] = hmm.initialprobabilities .* b[1,:]
        for i = 2:length(y)
            hmm.alpha.data[i,:] = hmm.transitionprobabilities' * hmm.alpha.data[i-1,:] .* b[i,:]
        end
        hmm.alpha.dirty = false
    end
    GCTools.pop!()
    return hmm.alpha.data
end

function backwardprobabilities( hmm::GaussianHMM{Calc, Out} ) where {Calc, Out}
    GCTools.push!(:backwardprobabilities)
    if hmm.beta.dirty
        y = observations( hmm )
        N = length(hmm.initialprobabilities)
        hmm.beta.data[end,:] = ones(Calc,length(hmm.initialprobabilities))
        b = probability( hmm )
        for i = length(y):-1:2
            hmm.beta.data[i-1,:] = hmm.transitionprobabilities * (hmm.beta.data[i,:] .* b[i,:])
        end
        hmm.beta.dirty = false
    end
    GCTools.pop!()
    return hmm.beta.data
end

function likelihood( hmm::GaussianHMM{Calc} ) where {Calc}
    GCTools.push!(:likelihood)
    if isnan(hmm.likelihood)
        alpha = forwardprobabilities( hmm )        
        hmm.likelihood = sum(alpha[end,:])
    end
    GCTools.pop!()
    return hmm.likelihood
end

function conditionaljointstateprobabilities( hmm::GaussianHMM )
    GCTools.push!(:conditionaljointstateprobabilities)
    if hmm.xi.dirty
        y = observations( hmm )
        T = length(y)
        alpha = forwardprobabilities( hmm )
        beta = backwardprobabilities( hmm )
        proby = likelihood( hmm )
        b = probability( hmm )
        GCTools.push!(:calculation)
        for i = 1:T-1
            hmm.xi.data[i,:,:] = hmm.transitionprobabilities.*(alpha[i,:]*(beta[i+1,:].*b[i+1,:])')/proby
        end
        GCTools.pop!()

        hmm.xi.dirty = false
    end
    GCTools.pop!()
    return hmm.xi.data
end

function conditionalstateprobabilities( hmm::GaussianHMM )
    GCTools.push!(:conditionalstateprobabilities)
    if hmm.gamma.dirty
        xi = conditionaljointstateprobabilities( hmm )
        hmm.gamma.data[1:end-1,:] = sum(xi, dims=3)
        hmm.gamma.data[end,:] = sum(xi[end,:,:], dims=2)

        hmm.gamma.dirty = false
    end
    GCTools.pop!()
    return hmm.gamma.data
end

function emstep( hmm::GaussianHMM{Calc,Out}, nexthmm::GaussianHMM; usestationary::Bool = false ) where {Calc,Out}
    GCTools.push!(:emstep)
    y = observations( hmm )
    T = length(y)
    
    gamma = conditionalstateprobabilities( hmm )
    occupation = sum(gamma[1:end-1,:],dims=1)
    
    xi = conditionaljointstateprobabilities( hmm )

    m = length(hmm.initialprobabilities)
    nexthmm.transitionprobabilities = reshape(sum(xi, dims=1), (m,m))./occupation'
    if usestationary
        nexthmm.initialprobabilities = stationary( nexthmm )
    else
        nexthmm.initialprobabilities = gamma[1,:]
    end
    
    nexthmm.means[:] = sum([gamma[i,:]*y[i] for i in 1:T])./vec(occupation)
    nexthmm.stds[:] = sqrt.(sum([gamma[i,:].*(y[i] .- hmm.means).^2 for i in 1:T])./vec(occupation))

    clearscratch( nexthmm )
    GCTools.pop!()
end

function em( hmm::GaussianHMM{Calc, Out};
             epsilon::Float64 = 0.0,
             debug::Int = 0,
             maxiterations::Iter = Inf,
             usestationary::Bool = false ) where {Calc, Out, Iter <: Number}
    t0 = Base.time()
    nexthmm = copy( hmm )
    hmms = [hmm, nexthmm]
    oldlikelihood = zero(Calc)
    newlikelihood = likelihood( hmm )
    done = false
    i = 1

    iterations = 1
    while !done
        if debug >= 2
            println( "Likelihood = $newlikelihood" )
        end
        emstep( hmms[i], hmms[3-i], usestationary=usestationary )
        oldlikelihood = newlikelihood
        done = any(isnan.(hmms[3-i].initialprobabilities)) || any(isnan.(hmms[3-i].transitionprobabilities)) ||
            any(isnan.(hmms[3-i].means)) || any(isnan.(hmms[3-i].stds)) || any(hmms[3-i].stds.<=0) ||
            iterations >= maxiterations
        if !done
            newlikelihood = likelihood( hmms[3-i] )
            done = newlikelihood / oldlikelihood - 1 <= epsilon
        end
        i = 3-i
        iterations += 1
    end

    hmm.initialprobabilities = hmms[3-i].initialprobabilities
    hmm.transitionprobabilities = hmms[3-i].transitionprobabilities
    hmm.means = hmms[3-i].means
    hmm.stds = hmms[3-i].stds
    hmm.scratch = hmms[3-i].scratch
    
    hmm.scratch[:iterations] = iterations
    hmm.scratch[:time] = Base.time() - t0
    
    if debug >= 1
        println( "Final likelihood = $oldlikelihood; iterations = $iterations, time = $(hmm.scratch[:time])" )
        flush(stdout)
    end
end

time( hmm::GaussianHMM ) = hmm.scratch[:time]

iterations( hmm::GaussianHMM ) = hmm.scratch[:iterations]

function permutederror( hmm1::GaussianHMM, hmm2::GaussianHMM )
    # use transition matrix to find best permutation; apply to means and stds
    n = length(hmm1.initialprobabilities)
    @assert( n == length(hmm2.initialprobabilities) )
    minerror = Inf
    minperm = nothing
    tp1 = convert( Matrix{Float64}, hmm1.transitionprobabilities )
    tp2 = convert( Matrix{Float64}, hmm2.transitionprobabilities )
    for perm in permutations(1:n)
        error = norm( tp1[perm, perm] - tp2, Inf )
        if error < minerror
            minerror = error
            minperm = perm
        end
    end
    meanerror = norm( hmm1.means[minperm] - hmm2.means, Inf )
    stderror = norm( hmm1.stds[minperm] - hmm2.stds, Inf )
    return (transitionprobabilities=minerror, means=meanerror, stds=stderror)
end

function reorder( hmm )
    perm = sortperm(hmm.means)
    hmm.means = hmm.means[perm]
    hmm.stds = hmm.stds[perm]
    hmm.initialprobabilities = hmm.initialprobabilities[perm]
    hmm.transitionprobabilities = hmm.transitionprobabilities[perm,perm]
    return hmm
end

ppv( io, s, v ) = println( io, rpad( s, 32 ), join([@sprintf("%8.2f", 100*convert(Float64,x)) for x in v]) )

function Base.show( io::IO, hmm::GaussianHMM )
    println( io )
    ppv( io, "initial probabilities:", hmm.initialprobabilities )
    println( io )
    ppv( io, "transition probabilities:", hmm.transitionprobabilities[1,:] )
    for i = 2:size(hmm.transitionprobabilities,1)
        ppv( io, "", hmm.transitionprobabilities[i,:] )
    end
    println( io )
    ppv( io, "means:", hmm.means )
    ppv( io, "stds:", hmm.stds )
end

end # module
