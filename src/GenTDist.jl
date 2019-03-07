using Ipopt
using MathProgBase
using Random
using Distributions
using SpecialFunctions

mutable struct GenTDistOptimizer <: MathProgBase.AbstractNLPEvaluator
    y::Vector{Float64}
    w::Vector{Float64}
end

function MathProgBase.initialize( ::GenTDistOptimizer, requested_features::Vector{Symbol} )
    unimplemented = setdiff( requested_features, [:Grad] )
    if !isempty( unimplemented )
        error( "The following features aren't implemented: " * join( string.(unimplemented), ", " ) )
    end
end

MathProgBase.features_available( ::GenTDistOptimizer ) = [:Grad]

function MathProgBase.eval_f( t::GenTDistOptimizer, x )
    println( "eval_f called with $x" )
    (mu, sigma, nu) = x
    y = t.y
    w = t.w
    n = sum(w)
    normalysq = ((y .- mu)/sigma).^2
    constant = n*(log(gamma((nu+1)/2)) - log(nu*pi)/2 - log(gamma(nu/2)) - log(sigma))
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
end

Random.seed!(1)
mu = randn()
sigma = rand( Exponential() )
nu = 1 + rand( Exponential() )
y = sigma*rand( TDist(nu), 1_000_000 ) .+ mu
w = ones(length(y))

mu1 = randn()
sigma1 = rand( Exponential() )
nu1 = 1 + rand( Exponential() )
t = GenTDistOptimizer( y, w )
MathProgBase.eval_f( t, [mu1, sigma1, nu1] )

g = zeros(3)
MathProgBase.eval_grad_f( t, g, [mu1, sigma1, nu1] )

delta = 1e-8
fp = MathProgBase.eval_f( t, [mu1 + delta, sigma1, nu1] )
fm = MathProgBase.eval_f( t, [mu1 - delta, sigma1, nu1] )
((fp - fm)/(2*delta) - g[1])/g[1]
fp = MathProgBase.eval_f( t, [mu1, sigma1 + delta, nu1] )
fm = MathProgBase.eval_f( t, [mu1, sigma1 - delta, nu1] )
((fp - fm)/(2*delta) - g[2])/g[2]
fp = MathProgBase.eval_f( t, [mu1, sigma1, nu1 + delta] )
fm = MathProgBase.eval_f( t, [mu1, sigma1, nu1 - delta] )
((fp - fm)/(2*delta) - g[3])/g[3]

solver = IpoptSolver()

model = MathProgBase.NonlinearModel(solver)
MathProgBase.loadproblem!(model, 3, 0, [-Inf,0.0,1.0], fill(Inf,3), Float64[], Float64[], :Max, t)
MathProgBase.setwarmstart!( model, [mu1, sigma1, nu1] )
MathProgBase.optimize!(model)
@assert( MathProgBase.status(model) == :Optimal )
x = MathProgBase.getsolution(model)
[x [mu,sigma,nu]]

n = 1_000_000
y = [sigma*rand( TDist(nu), n ) .+ mu; rand(n)]
w = [ones(n); zeros(n)]
perm = randperm(2*n)

t = GenTDistOptimizer( y[perm], w[perm] )

solver = IpoptSolver()

model = MathProgBase.NonlinearModel(solver)
MathProgBase.loadproblem!(model, 3, 0, [-Inf,0.0,1.0], fill(Inf,3), Float64[], Float64[], :Max, t)
MathProgBase.setwarmstart!( model, [mu1, sigma1, nu1] )
MathProgBase.optimize!(model)
@assert( MathProgBase.status(model) == :Optimal )
x = MathProgBase.getsolution(model)
[x [mu,sigma,nu]]

MathProgBase.eval_f( t, x )
g = zeros(3)
MathProgBase.eval_grad_f( t, g, x )

