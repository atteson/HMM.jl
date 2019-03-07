using HMMs
using Brobdingnag
using PyPlot
using LinearAlgebra

m = 3
graph1 = HMMs.fullyconnected( m )

hmm1 = HMMs.randomhmm( graph1, seed=1 )
y1 = HMMs.rand( hmm1, 100_000 );

hmm2 = HMMs.randomhmm( graph1, calc=Brob, seed=2 )
HMMs.setobservations( hmm2, y1 );
HMMs.em( hmm2, debug=2, keepintermediates=true )
# 631 iterations, 232 seconds, exp(117683.72053268638) likelihood
hmm2a = HMMs.randomhmm( graph1, calc=Brob, seed=2 )
HMMs.setobservations( hmm2a, y1 );
HMMs.em( hmm2a, debug=2, keepintermediates=true, acceleration=10 )
# 296 iterations, 108 seconds, exp(117681.20996505348) likelihood

C = convert( Matrix{Float64}, HMMs.sandwich( hmm2a ) )
convert(Matrix{Float64}, [HMMs.getparameters( hmm2a ) sqrt.(diag(C))] )
convert(Vector{Float64}, HMMs.getparameters( hmm2a )./sqrt.(diag(C)) )

graph3 = HMMs.Digraph( [1,1,2,2,2,3,3], [1,2,1,2,3,2,3] )
hmm3 = HMMs.randomhmm( graph3, seed=1 )
y3 = rand( hmm3, 100_000 );

hmm4 = HMMs.randomhmm( graph1, calc=Brob, seed=2 )
HMMs.setobservations( hmm4, y3 );
HMMs.em( hmm4, debug=2, keepintermediates=true )
# 7406 iterations, 2792 seconds


hmm4a = HMMs.randomhmm( graph1, calc=Brob, seed=2 )
HMMs.setobservations( hmm4a, y3 );
HMMs.em( hmm4a, debug=2, keepintermediates=true, acceleration=10 )
# 960 iterations, 394 seconds, exp(-17394.489088597584) likelihood
# old version: 4512 iterations, 1782 seconds, exp(-17394.488432664482) likelihood

C = convert( Matrix{Float64}, HMMs.sandwich( hmm4a ) )
convert(Matrix{Float64}, [HMMs.getparameters( hmm4a ) sqrt.(diag(C))] )

hmm4a3 = HMMs.randomhmm( graph1, calc=Brob, seed=3 )
HMMs.setobservations( hmm4a3, y3 );
HMMs.em( hmm4a3, debug=2, keepintermediates=true, acceleration=10 )
# 551 iterations, 223 seconds, exp(-17394.490665902726) likelihood

C = convert( Matrix{Float64}, HMMs.sandwich( hmm4a3 ) )
convert(Vector{Float64}, [HMMs.getparameters( hmm4a3 ); sqrt.(diag(C))] )
