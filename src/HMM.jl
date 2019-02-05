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

function unzip( zipper::Vector{Tuple{Int,Int}} )
    n = length(zipper)
    from = fill( 0, n )
    to = fill( 0, n )
    for i = 1:length(zipper)
        from[i] = zipper[i][1]
        to[i] = zipper[i][2]
    end
    return (from, to)
end

Digraph( zipper::Vector{Tuple{Int,Int}} ) = Digraph( unzip( zipper )... )

Base.copy( g::Digraph ) = Digraph( copy( g.from ), copy( g.to ) )

function addedge( g::Digraph, from::Int, to::Int )
    push!( g.from, from )
    push!( g.to, to )
end

fullyconnected( n::Int ) = Digraph( vcat( [collect(1:n) for i in 1:n]... ), vcat( [fill(i,n) for i in 1:n]... ) )

mutable struct DirtyArray{A <: AbstractArray}
#    data::Array{T,N}
    data::A
    dirty::Bool
end

DirtyArray( a::A ) where {A} = DirtyArray{A}( a, true )
DirtyArray{A}() where {T,N,A <: AbstractArray{T,N}} =
    DirtyArray( convert( A, zeros( T, fill(0,N)... ) ), true )

Base.copy( da::DirtyArray ) = DirtyArray( copy( da.data ), da.dirty )

const DirtyVector{T} = DirtyArray{Vector{T}} where {T}
const DirtyMatrix{T} = DirtyArray{Matrix{T}} where {T}

mutable struct GaussianHMM{Calc <: Real, Out <: Real}
    graph::Digraph
    initialprobabilities::Vector{Calc}
    transitionprobabilities::Matrix{Calc}
    means::Vector{Out}
    stds::Vector{Out}

    b::DirtyMatrix{Calc}
    db::DirtyArray{Array{Calc,3}}
    
    alpha::DirtyMatrix{Calc}
    dalpha::DirtyArray{Array{Calc,3}}
    
    beta::DirtyMatrix{Calc}
    
    xi::DirtyArray{Array{Calc,3}}
    
    gamma::DirtyMatrix{Calc}
    
    likelihood::Calc
    dlikelihood::DirtyVector{Calc}

    y::Vector{Out}

    scratch::Dict{Symbol,Any}
end

GaussianHMM(
    g::Digraph,
    pi::Vector{Calc},
    a::Matrix{Calc},
    mu::Vector{Out},
    sigma::Vector{Out};
    scratch::Dict{Symbol,Any} = Dict{Symbol,Any}(),
) where {Calc,Out} = GaussianHMM(
    g, pi, a, mu, sigma,
        
    DirtyMatrix{Calc}(),
    DirtyArray{Array{Calc,3}}(),
        
    DirtyMatrix{Calc}(),
    DirtyArray{Array{Calc,3}}(),
        
    DirtyMatrix{Calc}(),
        
    DirtyArray{Array{Calc,3}}(),
        
    DirtyMatrix{Calc}(),
        
    convert( Calc, NaN ),
    DirtyVector{Calc}(),
        
    Vector{Out}(),
        
    scratch,
)

Base.copy( hmm::GaussianHMM ) =
    GaussianHMM(
        copy( hmm.graph ),
        copy( hmm.initialprobabilities ), copy( hmm.transitionprobabilities ),
        copy( hmm.means ), copy( hmm.stds ),
        
        copy( hmm.b ),
        copy( hmm.db ),
        
        copy( hmm.alpha ),
        copy( hmm.dalpha ),
        
        copy( hmm.beta ),
        copy( hmm.xi ),
        copy( hmm.gamma ),
        
        hmm.likelihood,
        copy( hmm.dlikelihood ),
        
        copy( hmm.y ),
        copy( hmm.scratch ),
    )

function randomhmm(
    g::Digraph;
    calc::DataType = Float64,
    out::DataType = Float64,
    seed::Int = 1,
)
    Random.seed!( seed )
    
    numstates = max( maximum( g.from ), maximum( g.to ) )
    initialprobabilities = Vector{calc}(rand( numstates ))
    initialprobabilities ./= sum( initialprobabilities )

    transitionprobabilities = zeros( calc, numstates, numstates )
    for i = 1:length(g.from)
        transitionprobabilities[g.from[i], g.to[i]] = rand()
    end
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

function clear( hmm::GaussianHMM{Calc} ) where {Calc}
    hmm.b.dirty = true
    hmm.db.dirty = true
    
    hmm.alpha.dirty = true
    hmm.dalpha.dirty = true
    
    hmm.beta.dirty = true
    hmm.xi.dirty = true
    hmm.gamma.dirty = true
    
    hmm.likelihood = convert( Calc, NaN )
    hmm.dlikelihood.dirty = true
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
    hmm = GaussianHMM( graph, initialprobabilities, transitionprobabilities, means, stds )
    setobservations( hmm, y )
    return hmm
end

function setobservations( hmm::GaussianHMM{Calc}, y::Vector{U} ) where {Calc, U <: Real}
    T = length(y)
    m = length(hmm.initialprobabilities)
    
    hmm.b = DirtyArray( zeros( Calc, T, m ) )
    hmm.db = DirtyArray( zeros( Calc, m*(m+2), m, T ) )
    
    hmm.alpha = DirtyArray( zeros( Calc, T, m ) )
    hmm.dalpha = DirtyArray( zeros( Calc, m*(m+2), m, T ) )
    
    hmm.beta = DirtyArray( zeros( Calc, T, m ) )
    hmm.xi = DirtyArray( zeros( Calc, T-1, m, m ) )
    hmm.gamma = DirtyArray( zeros( Calc, T, m ) )

    hmm.dlikelihood = DirtyArray( zeros( Calc, m*(m+2) ) )
    
    clear( hmm )
    hmm.y = y
end

observations( hmm::GaussianHMM ) = hmm.y

function probability( hmm::GaussianHMM )
    if hmm.b.dirty
        y = observations( hmm )
        for i in 1:length(hmm.initialprobabilities)
            hmm.b.data[:,i] = pdf.( Normal( hmm.means[i], hmm.stds[i] ), y )
        end

        hmm.b.dirty = false
    end
    return hmm.b.data
end

function dprobability( hmm::GaussianHMM )
    if hmm.db.dirty
        (trash,m,T) = size(hmm.db.data)
        b = probability( hmm )
        y = observations( hmm )
        for i = 1:m
            for j = 1:m
                for t = 1:T
                    if i == j
                        z = (y[t] - hmm.means[i])/hmm.stds[i]
                        hmm.db.data[m^2 + i,j,t] = z * b[t,i] / hmm.stds[i]
                        hmm.db.data[m*(m+1) + i,j,t] = (z^2 - 1)/hmm.stds[i] * b[t,i]
                    end
                end
            end
        end
    end
    return hmm.db.data
end

function stationary( hmm::GaussianHMM )
    P = Matrix{Float64}( hmm.transitionprobabilities )
    N = length(hmm.initialprobabilities)
    I = one(P)
    P -= I
    P[:,1] = ones(N)
    return I[1,:]'*pinv(P)
end

function forwardprobabilities( hmm::GaussianHMM{Calc,Out} ) where {Calc,Out}
    if hmm.alpha.dirty
        (T,m) = size(hmm.alpha.data)
        b = probability( hmm )
                                   
        hmm.alpha.data[1,:] = hmm.initialprobabilities .* b[1,:]
        hmm.alpha.data[2:T,:] = zeros( Calc, (T-1,m) )
        for i = 2:T
            for j = 1:length(hmm.graph.from)
                from = hmm.graph.from[j]
                to = hmm.graph.to[j]
                hmm.alpha.data[i,to] += hmm.transitionprobabilities[from, to] * hmm.alpha.data[i-1,from] * b[i,to]
            end
        end
        hmm.alpha.dirty = false
    end
    return hmm.alpha.data
end

function dforwardprobabilities( hmm::GaussianHMM{Calc,Out} ) where {Calc,Out}
    if hmm.dalpha.dirty
        (T,m) = size(hmm.alpha.data)
        b = probability( hmm )
        db = dprobability( hmm )

        hmm.dalpha.data[:,:,1] = hmm.initialprobabilities' .* db[:,:,1]
        hmm.dalpha.data[:,:,2:T] = zeros( m*(m+2), m, T-1 )
        for i = 2:T
            for j = 1:length(hmm.graph.from)
                from = hmm.graph.from[j]
                to = hmm.graph.to[j]
                
                paramindex = (from-1)*m + to
                hmm.dalpha.data[paramindex,to,i] += hmm.alpha.data[i-1,from] * b[i,to]
                
                hmm.dalpha.data[:,to,i] += hmm.transitionprobabilities[from, to] * hmm.dalpha.data[:,from,i-1] * b[i,to]
                
                hmm.dalpha.data[:,to,i] += hmm.transitionprobabilities[from, to] * hmm.alpha.data[i-1,from] * db[:,to,i]
            end
        end
        hmm.alpha.dirty = false
    end
    return hmm.dalpha.data
end

function backwardprobabilities( hmm::GaussianHMM{Calc, Out} ) where {Calc, Out}
    if hmm.beta.dirty
        (T,m) = size(hmm.beta.data)
        b = probability( hmm )
        
        hmm.beta.data[end,:] = ones(Calc,length(hmm.initialprobabilities))
        hmm.beta.data[1:T-1,:] = zeros( Calc, (T-1,m) )
        for i = T:-1:2
            for j = 1:length(hmm.graph.from)
                from = hmm.graph.from[j]
                to = hmm.graph.to[j]
                hmm.beta.data[i-1,from] += hmm.transitionprobabilities[from,to] * hmm.beta.data[i,to] * b[i,to]
            end
        end
        hmm.beta.dirty = false
    end
    return hmm.beta.data
end

function likelihood( hmm::GaussianHMM{Calc} ) where {Calc}
    if isnan(hmm.likelihood)
        alpha = forwardprobabilities( hmm )
        hmm.likelihood = sum(alpha[end,:])
    end
    return hmm.likelihood
end

function dlikelihood( hmm::GaussianHMM{Calc} ) where {Calc}
    if hmm.dlikelihood.dirty
        dalpha = dforwardprobabilities( hmm )
        (p,m,T) = size(dalpha)
        hmm.dlikelihood.data[:] = reshape(sum(dalpha[:,:,end],dims=2), (p,))
    end
    return hmm.dlikelihood.data
end

function conditionaljointstateprobabilities( hmm::GaussianHMM{Calc,Out} ) where {Calc,Out}
    if hmm.xi.dirty
        alpha = forwardprobabilities( hmm )
        beta = backwardprobabilities( hmm )
        proby = likelihood( hmm )
        b = probability( hmm )
        (T,m) = size(alpha)
        
        hmm.xi.data[1:T-1,:,:] = zeros( Calc, (T-1,m,m) )
        for i = 1:T-1
            for j = 1:length(hmm.graph.from)
                from = hmm.graph.from[j]
                to = hmm.graph.to[j]
                hmm.xi.data[i,from,to] += hmm.transitionprobabilities[from,to] * alpha[i,from] * beta[i+1,to] * b[i+1,to]
            end
        end
        hmm.xi.data[:,:,:] /= proby

        hmm.xi.dirty = false
    end
    return hmm.xi.data
end

function conditionalstateprobabilities( hmm::GaussianHMM )
    if hmm.gamma.dirty
        xi = conditionaljointstateprobabilities( hmm )
        hmm.gamma.data[1:end-1,:] = sum(xi, dims=3)
        hmm.gamma.data[end,:] = sum(xi[end,:,:], dims=1)

        hmm.gamma.dirty = false
    end
    return hmm.gamma.data
end

function emstep( hmm::GaussianHMM{Calc,Out}, nexthmm::GaussianHMM ) where {Calc,Out}
    y = observations( hmm )
    T = length(y)
    
    gamma = conditionalstateprobabilities( hmm )
    occupation = sum(gamma[1:end-1,:],dims=1)
    
    xi = conditionaljointstateprobabilities( hmm )

    m = length(hmm.initialprobabilities)
    nexthmm.transitionprobabilities = reshape(sum(xi, dims=1), (m,m))./occupation'
    nexthmm.initialprobabilities = gamma[1,:]
    
    nexthmm.means[:] = gamma' * y./vec(occupation)
    nexthmm.stds[:] = sqrt.(diag(gamma' * (y .- hmm.means').^2)./vec(occupation))

    clear( nexthmm )
end

function em(
    hmm::GaussianHMM{Calc, Out};
    epsilon::Float64 = 0.0,
    debug::Int = 0,
    maxiterations::Iter = Inf,
) where {Calc, Out, Iter <: Number}
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
        emstep( hmms[i], hmms[3-i] )
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
