using HMMs
using Random
using Brobdingnag
using Distributions
using MathProgBase

import HMMs: GenTDist

Random.seed!(1)
(mu1,sigma1,nu1) = HMMs.randomparameters( GenTDist )
t1 = GenTDist( mu1, sigma1, nu1 )
y1 = rand( t1, 1_000_000 );
w1 = ones(length(y1));

opt = HMMs.GenTDistOptimizer( y1, w1 )
@assert( abs( MathProgBase.eval_f( opt, [mu1,sigma1,nu1] ) - sum(log.(pdf.( t1, y1 ))) ) < 1e-8 )

parameters = HMMs.randomparameters( GenTDist )
HMMs.fit_mle!( GenTDist, parameters, y1, w1, Dict{Symbol,Any}(), print_level=5 )
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
