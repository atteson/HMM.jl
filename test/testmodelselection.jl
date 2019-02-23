using HMM
using Brobdingnag
using Distributions

m = 3
graph = HMM.fullyconnected( m )
hmm1 = HMM.randomhmm( graph, seed=1 )
y = HMM.rand( hmm1, 100_000 );

hmm2 = HMM.randomhmm( graph, calc=Brob, seed=m )
HMM.setobservations( hmm2, y );
HMM.em( hmm2, debug=2 )

hmm1 = HMM.reorder( hmm1 )
hmm2 = HMM.reorder( hmm2 )

C = convert( Matrix{Float64}, HMM.sandwich( hmm2 ) )

indices = [(i-1)*m+j for i in 1:m, j in 1:m]
ijs = [divrem(index-1,m) .+ (1,1) for index in indices]

P = convert( Matrix{Float64}, hmm2.transitionprobabilities )

[cdf( Normal( P[i,j], sqrt(C[indices[i,j], indices[i,j]]) ), 0.0 ) for i in 1:m, j in 1:m]

 


