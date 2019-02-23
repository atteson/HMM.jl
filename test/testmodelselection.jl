using HMM
using Brobdingnag

m = 3
graph = HMM.fullyconnected( m )
hmm1 = HMM.randomhmm( graph, seed=1 )
y = HMM.rand( hmm1, 100_000 );

hmm2 = HMM.randomhmm( graph, calc=Brob, seed=m )
HMM.setobservations( hmm2, y );
HMM.em( hmm2, debug=2 )

hmm1 = HMM.reorder( hmm1 )
hmm2 = HMM.reorder( hmm2 )

C = HMM.sandwich( hmm2 )

i=1
for j = 2:m
    println( convert( Float64, C[i,i] - C[i,j] - C[j,i] + C[j,j] ) )
end
v = [1; -1; zeros(m-2)]
v' * C * v



