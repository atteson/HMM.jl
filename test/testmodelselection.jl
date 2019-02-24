using HMM
using Brobdingnag
using Distributions

hmm = HMM.randomhmm( HMM.fullyconnected( 1 ), calc=Brob, seed=1 )
y = rand( hmm, 100_000 )
C = convert( Matrix{Float64}, HMM.sandwich( hmm ) )

m = 3
graph1 = HMM.fullyconnected( m )
hmm1 = HMM.randomhmm( graph1, seed=1 )
y1 = rand( hmm1, 100_000 );

hmm2 = HMM.randomhmm( graph1, calc=Brob, seed=2 )
HMM.setobservations( hmm2, y1 );
HMM.em( hmm2, debug=2 )

hmm1 = HMM.reorder( hmm1 )
hmm2 = HMM.reorder( hmm2 )

C2 = convert( Matrix{Float64}, HMM.sandwich( hmm2 ) )

indices = [(i-1)*m+j for i in 1:m, j in 1:m]
ijs = [divrem(index-1,m) .+ (1,1) for index in indices]

P2 = convert( Matrix{Float64}, hmm2.transitionprobabilities )
[cdf( Normal( P2[i,j], sqrt(C2[indices[i,j], indices[i,j]]) ), 0.0 ) for i in 1:m, j in 1:m]

graph3 = HMM.Digraph( [1,1,2,2,2,3,3], [1,2,1,2,3,2,3] )
hmm3 = HMM.randomhmm( graph3, seed=1 )
y3 = rand( hmm3, 100_000 )

hmm4 = HMM.randomhmm( graph1, calc=Brob, seed=2 )
HMM.setobservations( hmm4, y3 )
HMM.em( hmm4, debug=2 )

hmm3 = HMM.reorder( hmm3 )
hmm4 = HMM.reorder( hmm4 )

C4 = convert( Matrix{Float64}, HMM.sandwich( hmm4 ) )

P4 = convert( Matrix{Float64}, hmm4.transitionprobabilities )
[cdf( Normal( P4[i,j], sqrt(C4[indices[i,j], indices[i,j]]) ), 0.0 ) for i in 1:m, j in 1:m]

HMM.setobservations( hmm3, y3 )
C3 = convert( Matrix{Float64}, HMM.sandwich( hmm3 ) )

