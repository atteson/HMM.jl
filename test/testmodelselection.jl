using HMMs
using Brobdingnag
using Distributions

m = 3
graph1 = HMMs.fullyconnected( m )
hmm1 = HMMs.randomhmm( graph1, seed=1 )
y1 = rand( hmm1, 100_000 );

hmm2 = HMMs.randomhmm( graph1, calc=Brob, seed=2 )
HMMs.setobservations( hmm2, y1 );
HMMs.em( hmm2, debug=2 )

hmm1 = HMMs.reorder!( hmm1 )
hmm2 = HMMs.reorder!( hmm2 )

C2 = convert( Matrix{Float64}, HMMs.sandwich( hmm2 ) )

indices = [(i-1)*m+j for i in 1:m, j in 1:m]
ijs = [divrem(index-1,m) .+ (1,1) for index in indices]

P2 = convert( Matrix{Float64}, hmm2.transitionprobabilities )
[cdf( Normal( P2[i,j], sqrt(C2[indices[i,j], indices[i,j]]) ), 0.0 ) for i in 1:m, j in 1:m]

graph3 = HMMs.Digraph( [1,1,2,2,2,3,3], [1,2,1,2,3,2,3] )
hmm3 = HMMs.randomhmm( graph3, seed=1 )
y3 = rand( hmm3, 100_000 )

hmm4 = HMMs.randomhmm( graph1, calc=Brob, seed=2 )
HMMs.setobservations( hmm4, y3 )
HMMs.em( hmm4, debug=2 )

hmm3 = HMMs.reorder!( hmm3 )
hmm4 = HMMs.reorder!( hmm4 )

C4 = convert( Matrix{Float64}, HMMs.sandwich( hmm4 ) )

P4 = convert( Matrix{Float64}, hmm4.transitionprobabilities )
[cdf( Normal( P4[i,j], sqrt(C4[indices[i,j], indices[i,j]]) ), 0.0 ) for i in 1:m, j in 1:m]

HMMs.setobservations( hmm3, y3 )
C3 = convert( Matrix{Float64}, HMMs.sandwich( hmm3 ) )

