using HMM
using Brobdingnag
using PyPlot

m = 3
graph1 = HMM.fullyconnected( m )

hmm1 = HMM.randomhmm( graph1, seed=1 )
y1 = HMM.rand( hmm1, 100_000 );

hmm2 = HMM.randomhmm( graph1, calc=Brob, seed=2 )
HMM.setobservations( hmm2, y1 );
HMM.em( hmm2, debug=2, keepintermediates=true )
# 631 iterations, 232 seconds, exp(117683.72053268638) likelihood
hmm2a = HMM.randomhmm( graph1, calc=Brob, seed=2 )
HMM.setobservations( hmm2a, y1 );
HMM.em( hmm2a, debug=2, keepintermediates=true, acceleration=10 )
# 296 iterations, 113 seconds, exp(117681.20996505348) likelihood

graph3 = HMM.Digraph( [1,1,2,2,2,3,3], [1,2,1,2,3,2,3] )
hmm3 = HMM.randomhmm( graph3, seed=1 )
y3 = rand( hmm3, 100_000 );

hmm4 = HMM.randomhmm( graph1, calc=Brob, seed=2 )
HMM.setobservations( hmm4, y3 );
HMM.em( hmm4, debug=2, keepintermediates=true )
# 7406 iterations, 2792 seconds


hmm4a = HMM.randomhmm( graph1, calc=Brob, seed=2 )
HMM.setobservations( hmm4a, y3 );
HMM.em( hmm4a, debug=2, keepintermediates=true, acceleration=10 )
# 960 iterations, 394 seconds, exp(-17394.489088597584) likelihood
# old version: 4512 iterations, 1782 seconds, exp(-17394.488432664482) likelihood

C = convert( Matrix{Float64}, HMM.sandwich( hmm4a ) )
convert(Matrix{Float64}, [HMM.getparameters( hmm4a ) sqrt.(diag(C))] )

hmm4a3 = HMM.randomhmm( graph1, calc=Brob, seed=3 )
HMM.setobservations( hmm4a, y3 );
HMM.em( hmm4a, debug=2, keepintermediates=true, acceleration=10 )
