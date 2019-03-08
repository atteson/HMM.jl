using Ipopt
using MathProgBase
using Distributions
using SpecialFunctions

mutable struct GenTDist <: ContinuousUnivariateDistribution
    mu::Float64
    sigma::Float64
    t::TDist
end

GenTDist( mu::Float64, sigma::Float64, nu::Float64 ) = GenTDist( mu, sigma, TDist( nu ) )

Base.rand( t::GenTDist ) = t.sigma .* rand( t.t ) .+ t.mu

Base.rand( t::GenTDist, n::Int ) = t.sigma .* rand( t.t, n ) .+ t.mu

Distributions.pdf( t::GenTDist, x::Float64 ) = pdf( t.t, (x - t.mu)/t.sigma)/t.sigma

randomparameters( ::Type{GenTDist} ) = [randn(), rand( Exponential() ), 1 + rand(Exponential())]

randomparameters( ::Type{GenTDist}, n::Int ) =
    [randn( 1, n ); rand( Exponential(), 1, n ); 1 .+ rand( Exponential(), 1, n )]

mutable struct GenTDistOptimizer <: MathProgBase.AbstractNLPEvaluator
    y::Vector{Float64}
    w::Vector{Float64}
    debug::Int
end

GenTDistOptimizer( y::Vector, w::Vector ) = GenTDistOptimizer( y, w, 0 )

function MathProgBase.initialize( ::GenTDistOptimizer, requested_features::Vector{Symbol} )
    unimplemented = setdiff( requested_features, [:Grad] )
    if !isempty( unimplemented )
        error( "The following features aren't implemented: " * join( string.(unimplemented), ", " ) )
    end
end

MathProgBase.features_available( ::GenTDistOptimizer ) = [:Grad]

function MathProgBase.eval_f( t::GenTDistOptimizer, x; debug=0 )
    if t.debug > 0
        println( "eval_f called with $x" )
    end
    (mu, sigma, nu) = x

    sigma <= 0.0 && return -Inf
    
    y = t.y
    w = t.w
    n = sum(w)
    normalysq = ((y .- mu)/sigma).^2
    constant = n*(lgamma((nu+1)/2) - log(nu*pi)/2 - lgamma(nu/2) - log(sigma))
    return constant - (nu+1)/2*sum(w .* log.(1 .+ normalysq/nu))
end

function MathProgBase.eval_grad_f( t::GenTDistOptimizer, g, x )
    (mu, sigma, nu) = x
    y = t.y
    w = t.w
    n = sum(w)
    normaly = (y .- mu)/sigma
    normalysq = normaly.^2
    g[1] = (nu+1)*sum(w .* (y .- mu)./(nu*sigma^2 .+ (y .- mu).^2))
    g[2] = -n/sigma + (nu+1)*sum(w .* normalysq ./ (sigma*(nu .+ normalysq)))
    g[3] = n*(digamma((nu+1)/2)/2 - 1/(2*nu) - digamma(nu/2)/2)
    g[3] += -sum(w .* log.(1 .+ normalysq/nu))/2 + (nu+1)/(2*nu)*sum(w .* normalysq ./ (nu .+ normalysq))
    if t.debug > 0
        println( "grad_f called with $x, returning $g" )
    end
end

function fit_mle!( ::Type{GenTDist},
                   parameters::AbstractVector{Out},
                   x::Vector{Out},
                   w::Vector{Calc},
                   scratch::Dict{Symbol,Any} ) where {Calc,Out}
    t = GenTDistOptimizer( x, w )
    
    solver = IpoptSolver(print_level=0)

    model = MathProgBase.NonlinearModel(solver)
    MathProgBase.loadproblem!(model, 3, 0, [-Inf,0.0,1.0], fill(Inf,3), Float64[], Float64[], :Max, t)
    MathProgBase.setwarmstart!( model, parameters )
    MathProgBase.optimize!(model)

    status = MathProgBase.status(model)
    if status != :Optimal
        @warn( "Non-optimal optimization result: $status" )
    end
    parameters[:] = MathProgBase.getsolution(model)
end
              

