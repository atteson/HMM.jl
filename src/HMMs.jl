module HMMs

using Distributions
using Random
using Combinatorics
using LinearAlgebra
using GCTools
using Printf
using SpecialFunctions

include("GenTDist.jl")

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

fullyconnected( n::Int ) =
    Digraph( vcat( [collect(1:n) for i in 1:n]... ), vcat( [fill(i,n) for i in 1:n]... ) )

mutable struct DirtyArray{A <: AbstractArray}
    data::A
    dirty::Bool
end

DirtyArray( a::A ) where {A} = DirtyArray{A}( a, true )
DirtyArray{A}() where {T,N,A <: AbstractArray{T,N}} =
    DirtyArray( convert( A, zeros( T, fill(0,N)... ) ), true )

Base.copy( da::DirtyArray ) = DirtyArray( copy( da.data ), da.dirty )

const DirtyVector{T} = DirtyArray{Vector{T}} where {T}
const DirtyMatrix{T} = DirtyArray{Matrix{T}} where {T}

mutable struct HMM{Dist <: Distribution, Calc <: Real, Out <: Real}
    graph::Digraph
    initialprobabilities::Vector{Out}
    transitionprobabilities::Matrix{Out}
    stateparameters::Matrix{Out}

    b::DirtyMatrix{Calc}
    dlogb::DirtyArray{Array{Calc,3}}
    d2logb::DirtyArray{Array{Calc,4}}
    d2b::DirtyArray{Array{Calc,4}}
    
    alpha::DirtyMatrix{Calc}
    dalpha::DirtyArray{Array{Calc,3}}
    d2alpha::DirtyArray{Array{Calc,4}}
    
    beta::DirtyMatrix{Calc}
    
    xi::DirtyArray{Array{Calc,3}}
    
    gamma::DirtyMatrix{Calc}
    
    likelihood::DirtyVector{Calc}
    dlikelihood::DirtyMatrix{Calc}
    d2likelihood::DirtyArray{Array{Calc,3}}
    d2loglikelihood::DirtyArray{Array{Calc,3}}

    y::Vector{Out}

    constraintmatrix::Matrix{Out}
    constraintvector::Vector{Out}
    basis::Matrix{Out}

    scratch::Dict{Symbol,Any}
end

const GaussianHMM{Calc, Out} = HMM{Normal,Calc,Out}
const LaplaceHMM{Calc, Out} = HMM{Laplace,Calc,Out}

function HMM{Dist,Calc,Out}(
    g::Digraph,
    pi::Vector{Out},
    a::Matrix{Out},
    stateparameters::Matrix{Out};
    scratch::Dict{Symbol,Any} = Dict{Symbol,Any}(),
) where {Dist,Calc,Out}
    (p,m) = size(stateparameters)
    A = [0.0 + (m!=1 && div(j-1,m)==i-1) for i in 1:m, j in 1:m^2]
    A = [A zeros(m,m*p)]
    result = HMM{Dist,Calc,Out}(
        g, pi, a, stateparameters,
        
        DirtyMatrix{Calc}(),
        DirtyArray{Array{Calc,3}}(),
        DirtyArray{Array{Calc,4}}(),
        DirtyArray{Array{Calc,4}}(),
        
        DirtyMatrix{Calc}(),
        DirtyArray{Array{Calc,3}}(),
        DirtyArray{Array{Calc,4}}(),
        
        DirtyMatrix{Calc}(),
        
        DirtyArray{Array{Calc,3}}(),
        
        DirtyMatrix{Calc}(),
        
        DirtyVector{Calc}(),
        DirtyMatrix{Calc}(),
        DirtyArray{Array{Calc,3}}(),
        DirtyArray{Array{Calc,3}}(),
        
        Vector{Out}(),

        A,
        ones(m),
        basis(A),
        
        scratch,
    )
end

Base.copy( hmm::HMM{Dist,Calc,Out} ) where {Dist,Calc,Out} =
    HMM{Dist,Calc,Out}(
        copy( hmm.graph ),
        copy( hmm.initialprobabilities ),
        copy( hmm.transitionprobabilities ),
        copy( hmm.stateparameters ),
        
        copy( hmm.b ),
        copy( hmm.dlogb ),
        copy( hmm.d2logb ),
        copy( hmm.d2b ),
        
        copy( hmm.alpha ),
        copy( hmm.dalpha ),
        copy( hmm.d2alpha ),
        
        copy( hmm.beta ),
        copy( hmm.xi ),
        copy( hmm.gamma ),
        
        copy( hmm.likelihood ),
        copy( hmm.dlikelihood ),
        copy( hmm.d2likelihood ),
        copy( hmm.d2loglikelihood ),
        
        copy( hmm.y ),

        copy( hmm.constraintmatrix ),
        copy( hmm.constraintvector ),
        copy( hmm.basis ),
        
        copy( hmm.scratch ),
    )

randomparameters( ::Union{Type{Normal},Type{Laplace}}, numstates::Int ) =
    [randn( 1, numstates ); randn( 1, numstates ).^2]

function randomhmm(
    g::Digraph;
    dist::Type{T} = Normal,
    calc::DataType = Float64,
    out::DataType = Float64,
    seed::Int = 1,
) where {T <: Distribution}
    Random.seed!( seed )
    
    numstates = max( maximum( g.from ), maximum( g.to ) )
    initialprobabilities = Vector{out}(rand( numstates ))
    initialprobabilities ./= sum( initialprobabilities )

    transitionprobabilities = zeros( out, numstates, numstates )
    for i = 1:length(g.from)
        transitionprobabilities[g.from[i], g.to[i]] = rand()
    end
    transitionprobabilities ./= sum( transitionprobabilities, dims=2 )
    
    parameters = Matrix{out}( randomparameters( dist, numstates ) )
    scratch = Dict{Symbol,Any}()
    scratch[:seed] = seed

    return HMM{dist,calc,out}( g, initialprobabilities, transitionprobabilities, parameters, scratch=scratch )
end

function Base.rand( hmm::HMM{Dist}, n::Int ) where {Dist}
    cdfs = [ cumsum( hmm.transitionprobabilities[i,:] ) for i in 1:length(hmm.initialprobabilities) ]
    
    state = searchsorted( cumsum( hmm.initialprobabilities ), rand() ).start
    statedists = [Dist(hmm.stateparameters[:,state]...) for state in 1:length(hmm.initialprobabilities)]
    observations = [rand( statedists[state] )]
    
    for t = 2:n
        state = searchsorted( cdfs[state], rand() ).start
        push!( observations, rand( statedists[state] ) )
    end
    return observations
end

function clear( hmm::HMM )
    hmm.b.dirty = true
    hmm.dlogb.dirty = true
    hmm.d2logb.dirty = true
    hmm.d2b.dirty = true
    
    hmm.alpha.dirty = true
    hmm.dalpha.dirty = true
    hmm.d2alpha.dirty = true
    
    hmm.beta.dirty = true
    hmm.xi.dirty = true
    hmm.gamma.dirty = true
    
    hmm.likelihood.dirty = true
    hmm.dlikelihood.dirty = true
    hmm.d2likelihood.dirty = true
    hmm.d2loglikelihood.dirty = true
end

function free( hmm::HMM{Dist,Calc} ) where {Dist,Calc}
    hmm.b = DirtyMatrix{Calc}()
    hmm.dlogb = DirtyArray{Array{Calc,3}}()
    hmm.d2logb = DirtyArray{Array{Calc,4}}()
    hmm.d2b = DirtyArray{Array{Calc,4}}()
    
    hmm.alpha = DirtyMatrix{Calc}()
    hmm.dalpha = DirtyArray{Array{Calc,3}}()
    hmm.d2alpha = DirtyArray{Array{Calc,4}}()
    
    hmm.beta = DirtyMatrix{Calc}()
    hmm.xi = DirtyArray{Array{Calc,3}}()
    hmm.gamma = DirtyMatrix{Calc}()
    
    hmm.likelihood = DirtyVector{Calc}()
    hmm.dlikelihood = DirtyMatrix{Calc}()
    hmm.d2likelihood = DirtyArray{Array{Calc,3}}()
    hmm.d2loglikelihood = DirtyArray{Array{Calc,3}}()
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

function Base.write( io::IO, hmm::HMM )
    writearray( io, hmm.graph.from )
    writearray( io, hmm.graph.to )
    writearray( io, hmm.initialprobabilities )
    writearray( io, hmm.transitionprobabilities )
    writearray( io, hmm.stateparameters )
    writearray( io, hmm.y )
end

function Base.read( io::IO, ::Type{HMM{Dist,Calc,Out}} ) where {Dist,Calc,Out}
    from = readarray( io, Vector{Int} )
    to = readarray( io, Vector{Int} )
    graph = Digraph( from, to )
    initialprobabilities = readarray( io, Vector{Out} )
    transitionprobabilities = readarray( io, Matrix{Out} )
    parameters = readarray( io, Matrix{Out} )
    y = readarray( io, Vector{Out} )
    hmm = HMM{Dist,Calc,Out}( graph, initialprobabilities, transitionprobabilities, parameters )
    setobservations( hmm, y )
    return hmm
end

function getparameters!( hmm::HMM{Dist,Calc,Out}, parameters::Vector{Out} ) where {Dist,Calc,Out}
    parameters[:] = [hmm.transitionprobabilities'[:]; hmm.stateparameters'[:]]
end

getparameters( hmm::HMM{Dist,Calc,Out} ) where {Dist,Calc,Out} =
    [hmm.transitionprobabilities'[:]; ; hmm.stateparameters'[:]]

function setparameters!( hmm::HMM{Dist,Calc,Out}, parameters::AbstractVector{Out} ) where {Dist,Calc,Out}
    (p,m) = size(hmm.stateparameters)
    hmm.transitionprobabilities'[:] = parameters[1:m*m]
    hmm.stateparameters'[:] = parameters[m*m+1:m*(m+p)]
#    hmm.stateparameters[2,:] = parameters[m*(m+1)+1:m*(m+2)]
    clear( hmm )
end

function setobservations( hmm::HMM{Dist,Calc}, y::Vector{U} ) where {Dist,Calc, U <: Real}
    T = length(y)
    m = length(hmm.initialprobabilities)
    
    free( hmm )
    hmm.y = y
end

observations( hmm::HMM ) = hmm.y

function probabilities( hmm::HMM{Dist,Calc} ) where {Dist,Calc}
    # makes the assumption that the first parameter is location and second is scale
    if hmm.b.dirty
        y = observations( hmm )
        T = length(y)
        m = length(hmm.initialprobabilities)
        if isempty(hmm.b.data)
            hmm.b.data = zeros( Calc, (T,m) )
        end

        for i in 1:length(hmm.initialprobabilities)
            if isnan(hmm.stateparameters[1,i]) || isnan(hmm.stateparameters[2,i])
                for j = 1:size(hmm.b.data[:,1],1)
                    hmm.b.data[j,i] = NaN
                end
            elseif hmm.stateparameters[2,i] == 0.0
                for j = 1:size(hmm.b.data[:,1],1)
                    hmm.b.data[j,i] = y == hmm.stateparameters[1,i] ? Inf : 0
                end
            else
                dist = Dist( hmm.stateparameters[:,i]... )
                hmm.b.data[:,i] = pdf.( dist, y )
            end
        end

        hmm.b.dirty = false
    end
    return hmm.b.data
end

function dlogprobabilities( hmm::GaussianHMM{Calc} ) where {Calc}
    if hmm.dlogb.dirty
        b = probabilities( hmm )
        (T,m) = size(b)
        if isempty(hmm.dlogb.data)
            hmm.dlogb.data = zeros( Calc, m*(m+2), m, T )
        end
        y = observations( hmm )
        for i = 1:m
            for t = 1:T
                z = (y[t] - hmm.stateparameters[1,i])/hmm.stateparameters[2,i]
                hmm.dlogb.data[m^2 + i,i,t] = z / hmm.stateparameters[2,i]
                hmm.dlogb.data[m*(m+1) + i,i,t] = (z^2 - 1)/hmm.stateparameters[2,i]
            end
        end
        hmm.dlogb.dirty = false
    end
    return hmm.dlogb.data
end

function dlogprobabilities( hmm::HMM{GenTDist,Calc} ) where {Calc}
    if hmm.dlogb.dirty
        b = probabilities( hmm )
        (T,m) = size(b)
        if isempty(hmm.dlogb.data)
            hmm.dlogb.data = zeros( Calc, m*(m+3), m, T )
        end
        
        y = observations( hmm )
        
        for i = 1:m
            (mu,sigma,nu) = hmm.stateparameters[:,i]
            for t = 1:T
                centeredy = y[t] - mu
                normalysq = (centeredy/sigma)^2
                hmm.dlogb.data[m^2 + i, i, t] = (nu+1)*centeredy/(nu*sigma^2 + centeredy^2)
                hmm.dlogb.data[m*(m+1) + i, i, t] = -1/sigma + (nu+1)* normalysq / (sigma*(nu + normalysq))
                
                hmm.dlogb.data[m*(m+2) + i, i, t] = digamma((nu+1)/2)/2 - 1/(2*nu) - digamma(nu/2)/2
                hmm.dlogb.data[m*(m+2) + i, i, t] += -log(1 + normalysq/nu)/2 + (nu+1)/(2*nu) * normalysq/(nu + normalysq)
            end
        end
        hmm.dlogb.dirty = false
    end
    return hmm.dlogb.data
end

function d2logprobabilities( hmm::GaussianHMM{Calc} ) where {Calc}
    if hmm.d2logb.dirty
        dlogb = dlogprobabilities( hmm )
        (p,m,T) = size(dlogb)
        if isempty(hmm.d2logb.data)
            hmm.d2logb.data = zeros( Calc, p, p, m, T )
        end
        y = observations( hmm )
        for i = 1:m
            for t = 1:T
                sigma = hmm.stateparameters[2,i]
                z = (y[t] - hmm.stateparameters[1,i])/sigma

                mui = m^2+i
                hmm.d2logb.data[mui,mui,i,t] = -1/sigma^2

                sigmai = m*(m+1)+i
                dmusigma = -2*z/sigma^2
                hmm.d2logb.data[mui,sigmai,i,t] = dmusigma
                hmm.d2logb.data[sigmai,mui,i,t] = dmusigma

                hmm.d2logb.data[sigmai,sigmai,i,t] = (1 - 3z^2)/sigma^2
            end
        end
        hmm.d2logb.dirty = false
    end
    return hmm.d2logb.data
end

function d2logprobabilities( hmm::HMM{GenTDist,Calc} ) where {Calc}
    if hmm.d2logb.dirty
        dlogb = dlogprobabilities( hmm )
        (p,m,T) = size(dlogb)
        if isempty(hmm.d2logb.data)
            hmm.d2logb.data = zeros( Calc, p, p, m, T )
        end
        y = observations( hmm )
        for i = 1:m
            (mu,sigma,nu) = hmm.stateparameters[:,i]
            centeredy = y .- mu
            centeredysq = centeredy .^ 2
            normalysq = (centeredy./sigma).^2
            denom = (nu * sigma^2 .+ centeredysq).^2

            mui = m^2 + i
            sigmai = m*(m+1) + i
            nui = m*(m+2) + i
            hmm.d2logb.data[mui,mui,i,:] = (nu+1) .* (centeredysq .- nu*sigma^2) ./ denom
            hmm.d2logb.data[mui,sigmai,i,:] = hmm.d2logb.data[sigmai,mui,i,:] = -2*(nu+1)*nu*sigma .* centeredy ./ denom
            hmm.d2logb.data[mui,nui,i,:] = hmm.d2logb.data[nui,mui,i,:] = (centeredy.^3 .- sigma^2 .* centeredy) ./ denom
            
            hmm.d2logb.data[sigmai,sigmai,i,:] =
                1/sigma^2 .- (nu+1) .* centeredysq .* (3*nu*sigma^2 .+ centeredysq) ./ (sigma^2 .* denom)
            hmm.d2logb.data[sigmai,nui,i,:] = hmm.d2logb.data[nui,sigmai,i,:] =
                sigma .* centeredysq .* (centeredysq .- sigma^2) ./ (sigma^2 .* denom)

            hmm.d2logb.data[nui,nui,i,:] =
                polygamma(1,(nu+1)/2)/4 + 1/(2*nu^2) - polygamma(1,nu/2)/4 .+
                ((nu-1) .* normalysq .^ 2 ./ 2 .- normalysq .* nu) ./ (nu .* (nu .+ normalysq)).^2
        end
    end
    return hmm.d2logb.data
end

function d2probabilities( hmm::HMM{Dist,Calc} ) where {Dist,Calc}
    if hmm.d2b.dirty
        b = probabilities( hmm )
        dlogb = dlogprobabilities( hmm )
        d2logb = d2logprobabilities( hmm )
        (p,m,T) = size(dlogb)
        if isempty(hmm.d2b.data)
            hmm.d2b.data = zeros( Calc, p, p, m, T )
        end
        for i = 1:m
            for t = 1:T
                hmm.d2b.data[:,:,i,t] = b[t,i] .* (d2logb[:,:,i,t] + dlogb[:,i,t] * dlogb[:,i,t]')
            end
        end

        hmm.d2b.dirty = false
    end
    return hmm.d2b.data
end

function stationary( hmm::HMM )
    P = Matrix{Float64}( hmm.transitionprobabilities )
    N = length(hmm.initialprobabilities)
    I = one(P)
    P -= I
    P[:,1] = ones(N)
    return I[1,:]'*pinv(P)
end

function variance( hmm )
    pi = stationary( hmm )
    mus = hmm.stateparameters[1,:]
    mu = dot( pi, mus )
    result = dot(pi, (hmm.stateparameters[2,:].^2 + (mus .- mu).^2))
    return result
end

function autocovariance( hmm, lag::Int )
    if lag == 0
        return variance( hmm )
    else
        pi = stationary( hmm )
        P = hmm.transitionprobabilities
        mus = hmm.stateparameters[1,:]
        mu = dot(pi, mus)
        jointprobabilities = pi' .* P^lag
        sum( jointprobabilities .* ((mus .- mu) * (mus .- mu)') )
    end
end

function forwardprobabilities( hmm::HMM{Dist,Calc,Out} ) where {Dist,Calc,Out}
    if hmm.alpha.dirty
        b = probabilities( hmm )
        (T,m) = size(b)
        if isempty( hmm.alpha.data )
            hmm.alpha.data = zeros( Calc, T, m )
        end
                                   
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

function dforwardprobabilities( hmm::HMM{Dist,Calc,Out} ) where {Dist,Calc,Out}
    if hmm.dalpha.dirty
        b = probabilities( hmm )
        dlogb = dlogprobabilities( hmm )
        alpha = forwardprobabilities( hmm )

        (p,m,T) = size(dlogb)

        if isempty( hmm.dalpha.data )
            hmm.dalpha.data = zeros( Calc, p, m, T )
        end
        
        hmm.dalpha.data[:,:,1] = hmm.initialprobabilities' .* (dlogb[:,:,1] .* b[1,:]')
        hmm.dalpha.data[:,:,2:T] = zeros( p, m, T-1 )
        for i = 2:T
            for j = 1:length(hmm.graph.from)
                from = hmm.graph.from[j]
                to = hmm.graph.to[j]
                
                paramindex = (from-1)*m + to
                hmm.dalpha.data[paramindex,to,i] += alpha[i-1,from] * b[i,to]
                
                hmm.dalpha.data[:,to,i] += hmm.transitionprobabilities[from, to] * hmm.dalpha.data[:,from,i-1] * b[i,to]
                
                hmm.dalpha.data[:,to,i] += hmm.transitionprobabilities[from, to] * alpha[i-1,from] * (dlogb[:,to,i] .* b[i,to])
            end
        end
        hmm.dalpha.dirty = false
    end
    return hmm.dalpha.data
end

function d2forwardprobabilities( hmm::HMM{Dist,Calc,Out} ) where {Dist,Calc,Out}
    if hmm.d2alpha.dirty
        b = probabilities( hmm )
        dlogb = dlogprobabilities( hmm )
        d2b = d2probabilities( hmm )
        
        alpha = forwardprobabilities( hmm )
        dalpha = dforwardprobabilities( hmm )

        (p,m,T) = size(dlogb)
        if isempty( hmm.d2alpha.data )
            hmm.d2alpha.data = zeros( Calc, p, p, m, T )
        end

        for i = 1:m
            hmm.d2alpha.data[:,:,i,1] = hmm.initialprobabilities[i] .* d2b[:,:,i,1]
        end
        hmm.d2alpha.data[:,:,:,2:T] = zeros( p, p, m, T-1 )
        for i = 2:T
            for j = 1:length(hmm.graph.from)
                from = hmm.graph.from[j]
                to = hmm.graph.to[j]
                paramindex = (from-1)*m + to
                
                hmm.d2alpha.data[:,:,to,i] +=
                    hmm.transitionprobabilities[from,to] * hmm.d2alpha.data[:,:,from,i-1] * b[i,to]

                hmm.d2alpha.data[paramindex,:,to,i] += dalpha[:,from,i-1] * b[i,to]
                hmm.d2alpha.data[:,paramindex,to,i] += dalpha[:,from,i-1] * b[i,to]

                hmm.d2alpha.data[paramindex,:,to,i] += alpha[i-1,from] * (dlogb[:,to,i] .* b[i,to])
                hmm.d2alpha.data[:,paramindex,to,i] += alpha[i-1,from] * (dlogb[:,to,i] .* b[i,to])

                # adding these together all at once preserves symmetry
                hmm.d2alpha.data[:,:,to,i] +=
                    dalpha[:,from,i-1] * (dlogb[:,to,i]' .* b[i,to]) * hmm.transitionprobabilities[from,to] + 
                    (dlogb[:,to,i] .* b[i,to]) * dalpha[:,from,i-1]' * hmm.transitionprobabilities[from,to]

                hmm.d2alpha.data[:,:,to,i] +=
                    alpha[i-1,from] * hmm.transitionprobabilities[from,to] * d2b[:,:,to,i]
            end
        end
        hmm.d2alpha.dirty = false
    end
    return hmm.d2alpha.data
end

function backwardprobabilities( hmm::HMM{Dist,Calc, Out} ) where {Dist,Calc, Out}
    if hmm.beta.dirty
        b = probabilities( hmm )
        (T,m) = size(b)
        if isempty( hmm.beta.data )
            hmm.beta.data = zeros( Calc, T, m )
        end
        
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

function likelihood( hmm::HMM{Dist,Calc} ) where {Dist,Calc}
    if hmm.likelihood.dirty
        alpha = forwardprobabilities( hmm )
        (T,m) = size(alpha)
        if isempty( hmm.likelihood.data )
            hmm.likelihood.data = zeros( Calc, T )
        end
        
        hmm.likelihood.data[:] = sum(alpha, dims=2)

        hmm.likelihood.dirty = false
    end
    return hmm.likelihood.data
end

function dlikelihood( hmm::HMM{Dist,Calc} ) where {Dist,Calc}
    if hmm.dlikelihood.dirty
        dalpha = dforwardprobabilities( hmm )
        (p,m,T) = size(dalpha)
        if isempty( hmm.dlikelihood.data )
            hmm.dlikelihood.data = zeros( Calc, p, T )
        end
        
        hmm.dlikelihood.data[:,:] = reshape(sum(dalpha[:,:,:],dims=2), (p,T))
        
        hmm.dlikelihood.dirty = false
    end
    return hmm.dlikelihood.data
end

function d2likelihood( hmm::HMM{Dist,Calc} ) where {Dist,Calc}
    if hmm.d2likelihood.dirty
        d2alpha = d2forwardprobabilities( hmm )
        (p,p,m,T) = size(d2alpha)
        if isempty( hmm.d2likelihood.data )
            hmm.d2likelihood.data = zeros( Calc, p, p, T )
        end
        
        hmm.d2likelihood.data[:,:,:] = reshape(sum(d2alpha[:,:,:,:],dims=3), (p,p,T))
        
        hmm.d2likelihood.dirty = false
    end
    return hmm.d2likelihood.data
end

function d2loglikelihood( hmm::HMM{Dist,Calc} ) where {Dist,Calc}
    if hmm.d2loglikelihood.dirty
        l = likelihood( hmm )
        dl = dlikelihood( hmm )
        d2l = d2likelihood( hmm )
        (p,T) = size(dl)

        if isempty( hmm.d2loglikelihood.data )
            hmm.d2loglikelihood.data = zeros(Calc,p,p,T)
        end

        for i = 1:T
            hmm.d2loglikelihood.data[:,:,i] = d2l[:,:,i] ./ l[i] - dl[:,i] * dl[:,i]' ./ l[i]^2
        end
        
        hmm.d2loglikelihood.dirty = false
    end
    return hmm.d2loglikelihood.data
end

function basis( A::Matrix{Out}; epsilon::Out = 1e-12 ) where {Out <: Number}
    (m, n) = size(A)
    (U,S,V) = svd( [A; zeros(n - m, n)] )
    indices = abs.(S) .< epsilon
    result = V[:,indices]
    return result
end

function addconstraints( hmm::HMM{Dist,Calc,Out}, A::AbstractMatrix{Out}, b::AbstractVector{Out} ) where {Dist,Calc,Out}
    hmm.constraintmatrix = [hmm.constraintmatrix; A]
    hmm.constraintvector = [hmm.constraintvector; b]
    hmm.basis = basis( hmm.constraintmatrix )
end

dcollapse( hmm::HMM ) = hmm.basis'

dexpand( hmm::HMM ) = hmm.basis

function sandwich( hmm::HMM{Dist,Calc} ) where {Dist,Calc}
    # for now, we're only going to put in the equality constraints for the simplex

    dc = dcollapse( hmm )

    l = likelihood( hmm )
    dl = dc * dlikelihood( hmm )

    (p,T) = size(dl)
    dlogl = [zeros(1,p); dl'./l]
    dlogln = diff( dlogl, dims=1 )
    V = dlogln' * dlogln

    J = dc * d2loglikelihood( hmm )[:,:,end] * dc'
    
    de = dexpand( hmm )
    result = de * inv(J) * V * inv(J) * de'
    return result
end

function conditionaljointstateprobabilities( hmm::HMM{Dist,Calc,Out} ) where {Dist,Calc,Out}
    if hmm.xi.dirty
        alpha = forwardprobabilities( hmm )
        beta = backwardprobabilities( hmm )
        proby = likelihood( hmm )[end]
        b = probabilities( hmm )
        (T,m) = size(alpha)

        if isempty( hmm.xi.data )
            hmm.xi.data = zeros( Calc, T-1, m, m )
        end
        
        hmm.xi.data[1:T-1,:,:] = zeros( Calc, (T-1,m,m) )
        for i = 1:T-1
            for j = 1:length(hmm.graph.from)
                from = hmm.graph.from[j]
                to = hmm.graph.to[j]
                hmm.xi.data[i,from,to] +=
                    hmm.transitionprobabilities[from,to] * alpha[i,from] * beta[i+1,to] * b[i+1,to]
            end
        end
        hmm.xi.data[:,:,:] /= proby

        hmm.xi.dirty = false
    end
    return hmm.xi.data
end

function conditionalstateprobabilities( hmm::HMM{Dist,Calc} ) where {Dist,Calc}
    if hmm.gamma.dirty
        xi = conditionaljointstateprobabilities( hmm )
        (Tm1, m, trash) = size(xi)
        if isempty( hmm.gamma.data )
            hmm.gamma.data = zeros( Calc, Tm1 + 1, m )
        end
        
        hmm.gamma.data[1:end-1,:] = sum(xi, dims=3)
        hmm.gamma.data[end,:] = sum(xi[end,:,:], dims=1)

        hmm.gamma.dirty = false
    end
    return hmm.gamma.data
end

function fit_mle!(
    ::Type{Normal},
    parameters::AbstractVector{Out},
    x::Vector{Out},
    w::Vector{Calc},
    scratch::Dict{Symbol,Any};
    kwargs...
) where {Calc,Out}
    mu = dot( w, x )
    sigma = sqrt( dot( w, (x .- mu).^2 ) )
    parameters[1] = mu
    parameters[2] = sigma
end

function fit_mle!(
    ::Type{Laplace},
    parameters::AbstractVector{Out},
    x::Vector{Out},
    w::Vector{Calc},
    scratch::Dict{Symbol,Any};
    kwargs...
) where {Calc,Out}
    if !haskey( scratch, :sortperm )
        scratch[:sortperm] = sortperm(x)
    end
    perm = scratch[:sortperm]
    
    cumw = cumsum( w[perm] )
    mu = x[perm[findfirst(cumw.>=0.5*cumw[end])]]
    sigma = dot( w, abs.(x .- mu) )
    parameters[1] = mu
    parameters[2] = sigma
end

function emstep(
    hmm::HMM{Dist,Calc,Out},
    nexthmm::HMM{Dist,Calc,Out};
    max_iter::Int = 0,
    print_level::Int = 0,
) where {Dist,Calc,Out}
    y = observations( hmm )
    T = length(y)
    
    gamma = conditionalstateprobabilities( hmm )
    occupation = sum(gamma[1:end-1,:],dims=1)
    
    xi = conditionaljointstateprobabilities( hmm )

    m = length(hmm.initialprobabilities)
    nexthmm.transitionprobabilities = reshape(sum(xi, dims=1), (m,m))./occupation'
    nexthmm.initialprobabilities = gamma[1,:]./sum(gamma[1,:])

    for i = 1:m
        nexthmm.stateparameters[:,i] = hmm.stateparameters[:,i]
        if max_iter > 0
            fit_mle!( Dist, view( nexthmm.stateparameters, :, i ), y, gamma[:,i]/occupation[i], hmm.scratch,
                      max_iter = max_iter, print_level=print_level )
        else
            fit_mle!( Dist, view( nexthmm.stateparameters, :, i ), y, gamma[:,i]/occupation[i], hmm.scratch,
                      print_level=print_level )
        end
    end

    clear( nexthmm )
end

function em(
    hmm::HMM{Dist,Calc, Out};
    epsilon::Float64 = 0.0,
    debug::Int = 0,
    maxiterations::Iter = Inf,
    keepintermediates = false,
    acceleration = Inf,
    accelerationlinestart = 500,
    accelerationmaxhalves = 5,
    max_iter::Int = 0,
    print_level::Int = 0,
    timefractiontowardszero = 0.5,
    observations = 10,
    finishbig = 0,
) where {Dist, Calc, Out, Iter <: Number}
    if acceleration < Inf
        @assert( keepintermediates )
    end
    
    t0 = Base.time()
    nexthmm = copy( hmm )
    hmms = [hmm, nexthmm]
    oldlikelihood = zero(Calc)
    newlikelihood = likelihood( hmm )[end]
    done = false
    i = 1
    nextacceleration = acceleration
    m = length(hmm.initialprobabilities)

    if keepintermediates
        intermediates = [getparameters( hmm )]
    end

    iterations = 1
    while !done
        if debug >= 2
            println( "Iteration $iterations, likelihood = $newlikelihood" )
        end
        emstep( hmms[i], hmms[3-i], max_iter=max_iter, print_level=print_level )
        oldlikelihood = newlikelihood
        done = any(isnan.(hmms[3-i].initialprobabilities)) ||
            any(isnan.(hmms[3-i].transitionprobabilities)) ||
            any(isnan.(hmms[3-i].stateparameters)) || any(hmms[3-i].stateparameters[2,:].<=0) ||
            iterations >= maxiterations
        if !done
            newlikelihood = likelihood( hmms[3-i] )[end]
            done = newlikelihood / oldlikelihood - 1 <= epsilon
            if !done
                i = 3-i
            end
        else
            i = 3-i
        end
        iterations += 1
        if keepintermediates
            push!( intermediates, getparameters( hmms[i] ) )
        end
        if iterations >= nextacceleration
            y = hcat( intermediates[end-observations+1:end]... )'
            t = -collect(reverse(0:observations-1))
            X = [ones(observations) t t.^2]
            beta = inv(X'*X)*X'*y
            
            t = accelerationlinestart
                
            # let's bound t to where the probabilities go <0 or >1
            m = length(hmms[3-i].initialprobabilities)
            for j = 1:m^2
                (c,b,a) = beta[:,j]

                for target=0:1
                    discriminant = b^2 - 4 * a * (c - target) * timefractiontowardszero
                    if discriminant >= 0
                        srd = sqrt(discriminant)
                        roots = (-b .+ [-srd,srd])/(2*a)
                        roots = roots[roots .> 0]
                        if !isempty(roots)
                            t = min( t, minimum(roots) )
                        end
                    end
                end
            end

            newerlikelihood = 0.0
            optimalparameters = nothing
            optimalt = 0
            for j = 1:accelerationmaxhalves
                x = [1.0, t, t^2]
                p = x' * beta
                p = reshape( p, length(p) )
                setparameters!( hmms[3-i], p )
                
                hmms[3-i].transitionprobabilities ./= sum( hmms[3-i].transitionprobabilities, dims=2 )

                if any(hmms[3-i].stateparameters[2,:] .< 0 )
                    continue
                end

                evennewerlikelihood = likelihood( hmms[3-i] )[end]
                if evennewerlikelihood > newerlikelihood
                    newerlikelihood = evennewerlikelihood
                    optimalparameters = getparameters( hmms[3-i] )
                    optimalt = t
                end
                t = t/2
            end
            if newerlikelihood > newlikelihood
                if debug >= 2
                    println( "Accepting acceleration with $(newerlikelihood) and t=$optimalt" )
                end
                newlikelihood = newerlikelihood
                i = 3-i
                setparameters!( hmms[i], optimalparameters )
                push!( intermediates, optimalparameters )
            else
                if debug >= 2
                    println( "Acceleration not accepted with $(newerlikelihood)" )
                end
            end
            nextacceleration += acceleration
        end
    end

    if finishbig > 0
        bighmm = HMM{Dist,BigFloat,Out}(
            hmm.graph,
            hmm.initialprobabilities,
            hmm.transitionprobabilities,
            hmm.stateparameters,
        )
        setobservations( bighmm, hmm.y )

        # no acceleration
        em( bighmm,
            epsilon=epsilon,
            debug=debug,
            maxiterations=finishbig,
            max_iter=max_iter,
            print_level=print_level,
            )

        setparameters!( hmm, getparameters( bighmm ) )
        clear( hmm )
    end

    hmm.initialprobabilities = hmms[i].initialprobabilities
    hmm.transitionprobabilities = hmms[i].transitionprobabilities
    hmm.stateparameters = hmms[i].stateparameters
    hmm.scratch = hmms[i].scratch
    
    hmm.scratch[:iterations] = iterations
    hmm.scratch[:time] = Base.time() - t0
    if keepintermediates
        hmm.scratch[:intermediates] = intermediates
    end
    
    if debug >= 1
        println( "Final likelihood = $(HMMs.likelihood(hmm)[end]); iterations = $iterations, time = $(hmm.scratch[:time])" )
        flush(stdout)
    end
end

time( hmm::HMM ) = hmm.scratch[:time]

iterations( hmm::HMM ) = hmm.scratch[:iterations]

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
    meanerror = norm( hmm1.stateparameters[1,minperm] - hmm2.stateparameters[1,:], Inf )
    stderror = norm( hmm1.stateparameters[2,minperm] - hmm2.stateparameters[2,:], Inf )
    return (transitionprobabilities=minerror, means=meanerror, stds=stderror)
end

function applyperm!( hmm, perm )
    for i = 1:size(hmm.stateparameters,1)
        hmm.stateparameters[i,:] = hmm.stateparameters[i,perm]
    end
    hmm.initialprobabilities = hmm.initialprobabilities[perm]
    hmm.transitionprobabilities = hmm.transitionprobabilities[perm,perm]
    return hmm
end

reorder!( hmm ) = applyperm!( hmm, sortperm(hmm.stateparameters[1,:]) )

ppv( io, s, v ) = println( io, rpad( s, 32 ), join([@sprintf("%8.2f", 100*convert(Float64,x)) for x in v]) )

function Base.show( io::IO, hmm::HMM{T} ) where {T <: Union{Normal,Laplace}}
    println( io )
    ppv( io, "initial probabilities:", hmm.initialprobabilities )
    println( io )
    ppv( io, "transition probabilities:", hmm.transitionprobabilities[1,:] )
    for i = 2:size(hmm.transitionprobabilities,1)
        ppv( io, "", hmm.transitionprobabilities[i,:] )
    end
    println( io )
    ppv( io, "locations:", hmm.stateparameters[1,:] )
    ppv( io, "scales:", hmm.stateparameters[2,:] )
end

function Base.show( io::IO, hmm::HMM{GenTDist} )
    println( io )
    ppv( io, "initial probabilities:", hmm.initialprobabilities )
    println( io )
    ppv( io, "transition probabilities:", hmm.transitionprobabilities[1,:] )
    for i = 2:size(hmm.transitionprobabilities,1)
        ppv( io, "", hmm.transitionprobabilities[i,:] )
    end
    println( io )
    ppv( io, "locations:", hmm.stateparameters[1,:] )
    ppv( io, "scales:", hmm.stateparameters[2,:] )
    ppv( io, "nus:", hmm.stateparameters[3,:] )
end

function draw( outputfile::String, hmm::HMM )
    inputfile = tempname()

    reorder!( hmm )
    
    open( inputfile, "w" ) do io
        println( io, "digraph G {" )
        for index in CartesianIndices(hmm.transitionprobabilities)
            (i,j) = Tuple(index)
            weight = sqrt(1 - hmm.transitionprobabilities[index])
            value = @sprintf( "%02x", Int(round(convert(Float64,255*weight))) )
            if value != "00"
                println( io, "$i -> $j [color=\"#$value$value$value\"];" )
            end
        end
        println( io, "}" )
    end
    run( `dot -Tpdf $inputfile -o $outputfile` )
    rm( inputfile )
end

end # module
