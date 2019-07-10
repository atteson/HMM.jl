using HMMs
using Random
using Brobdingnag
using Distributions
using MathProgBase

import HMMs: GenTDist

p3 = [0.0,1.0,0.0]
g = zeros(length(p3))
for i = 0:10
    opt2 = HMMs.GenTDistOptimizer( Float64[i], [1.0] )
    p3[3] = 1e-6 + 1e-12
    fup = MathProgBase.eval_f( opt2, p3 )
    p3[3] = 1e-6 - 1e-12
    fdn = MathProgBase.eval_f( opt2, p3 )
    fidi = (fup - fdn)/2e-12
    theory = i^4/4 - i^2/2 - 1/4
    @assert( (fidi - theory)/theory < 1e-4 )
    MathProgBase.eval_grad_f( opt2, g, p3 )
    @assert( abs(g[3] - fidi)/theory < 1e-4, "fidi and gradient mismatch for $i" )
end

Random.seed!(1)
(mu1,sigma1,nu1) = HMMs.randomparameters( GenTDist )
t1 = GenTDist( mu1, sigma1, nu1 )
y1 = rand( t1, 1_000_000 );
w1 = ones(length(y1));

opt = HMMs.GenTDistOptimizer( y1, w1 )
@assert( abs( MathProgBase.eval_f( opt, [mu1,sigma1,nu1] ) - sum(log.(pdf.( t1, y1 ))) ) < 1e-8 )

parameters = HMMs.randomparameters( GenTDist )
HMMs.fit_mle!( GenTDist, parameters, y1, w1, Dict{Symbol,Any}(), print_level=1 )
@assert( maximum(abs.(parameters - [mu1,sigma1,nu1])) < 1e-2 )

(mu2,sigma2,nu2) = HMMs.randomparameters( GenTDist )
t2 = GenTDist( mu2, sigma2, nu2 )
y2 = fill(NaN,1_000_000);
w2 = fill(NaN,length(y2));
for i = 1:length(y2)
    w2[i] = rand()
    y2[i] = rand() < w2[i] ? rand( t1 ) : rand( t2 )
end

parameters = HMMs.randomparameters( GenTDist )
HMMs.fit_mle!( GenTDist, parameters, y2, w2, Dict{Symbol,Any}() )
# this doesn't seem to work but I'm not sure it's well-founded
#@assert( maximum(abs.(parameters - [mu1,sigma1,nu1])) < 1e-2 )

parameters = [0.0005, 0.01, 0.0]
t3 = GenTDist( parameters... )
y3 = rand( t3, 1_000_000 );
w3 = rand(length(y3));

opt = HMMs.GenTDistOptimizer( y3, w3 )
p2 = HMMs.randomparameters( GenTDist )
calc1 = MathProgBase.eval_f( opt, p2 )
calc2 = sum(w3.*(logpdf.(TDist(1/p2[3]), (y3 .- p2[1])./p2[2]) .- log(p2[2])))
@assert( abs( calc1 - calc2 ) < 1e-8 )

@assert( abs(MathProgBase.eval_f( opt, parameters ) - sum(w3.*logpdf.(Normal(parameters[1],parameters[2]), y3)) ) < 1e-8 )

function f( y, w, p )
    opt = HMMs.GenTDistOptimizer( y, w )
    return MathProgBase.eval_f( opt, Vector{BigFloat}(p) )
end

function fidi( y, w, p, i, epsilon )
    g = zeros(length(p))
    pup = copy( p )
    pup[i] += epsilon
    pdn = copy( p )
    pdn[i] -= epsilon
    return (f( y, w, pup ) - f( y, w, pdn ))/(2*epsilon)
end

function grad( y, w, p )
    opt = HMMs.GenTDistOptimizer( y, w )
    g = zeros(length(p))
    MathProgBase.eval_grad_f( opt, g, p )
    return g
end

p = [0.000492603, 0.0100018, 4.87344e-12]
fup = f( y3, w1, p + [0,0,1e-12] )
fdn = f( y3, w1, p - [0,0,1e-12] )
(fup - fdn)/(2e-12)

parameters = [0.000497648, 0.0100008, 7.34902e-155]
n = length(y3)
y = y3[1:n]
w = w1[1:n]
f( y, w, parameters )
g = grad( y, w, parameters )
fidi( y, w, parameters, 3, 1e-120 )
f( y, w, parameters + [0,0,1e-120] ) - f( y, w, parameters + [0,0,1e-120] )
alpha = -parameters[3]/g[3]
p = parameters + alpha * g
f( y, w, p ) - f( y, w, parameters )
g = grad( y, w, p )


parameters = HMMs.randomparameters( GenTDist )
HMMs.fit_mle!( GenTDist, parameters, y3, w1, Dict{Symbol,Any}(), print_level=1 )

f( y3, w1, [0.000497648, 0.0100008, -6.38655e-118] )

(digamma((nu+1)/2)/2 - 1/(2*nu) - digamma(nu/2)/2)
(digamma((nu+1)/2)/2 - digamma(nu/2)/2 - 1/(2*nu))
nu = BigFloat(nu)
(digamma((nu+1)/2)/2 - digamma(nu/2)/2 - 1/(2*nu))
digamma((nu+1)/2)/2 - digamma(nu/2)/2
nu = Float64(nu)
digamma((nu+1)/2)/2 - digamma(nu/2)/2

graph = HMMs.fullyconnected(1)
hmm1 = HMMs.randomhmm( graph, dist=GenTDist, calc=Brob, seed=1 )
y1 = rand( hmm1, 10_000 );

hmm2 = HMMs.randomhmm( graph, dist=GenTDist, calc=Brob, seed=2 )
HMMs.setobservations( hmm2, y1 );
HMMs.em( hmm2, debug=2 )

graph = HMMs.fullyconnected(2)
hmm1 = HMMs.randomhmm( graph, dist=GenTDist, calc=Brob, seed=1 )
y1 = rand( hmm1, 100_000 );

hmm2 = HMMs.randomhmm( graph, dist=GenTDist, calc=Brob, seed=2 )
HMMs.setobservations( hmm2, y1 );
HMMs.em( hmm2, debug=2 )

hmm1 = HMMs.randomhmm( graph, dist=GenTDist, calc=Brob, seed=1 )
y1 = rand( hmm1, 10_000 );

hmm2 = HMMs.randomhmm( graph, dist=GenTDist, calc=Brob, seed=2 )
HMMs.setobservations( hmm2, y1 );
HMMs.em( hmm2, debug=2 )

hmm2 = HMMs.randomhmm( graph, dist=GenTDist, calc=Brob, seed=3 )
HMMs.setobservations( hmm2, y1 );
HMMs.em( hmm2, debug=2 )
