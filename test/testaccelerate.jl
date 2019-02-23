using HMM
using Brobdingnag
using PyPlot

m = 3
graph1 = HMM.fullyconnected( m )

graph3 = HMM.Digraph( [1,1,2,2,2,3,3], [1,2,1,2,3,2,3] )
hmm3 = HMM.randomhmm( graph3, seed=1 )
y3 = HMM.rand( hmm3, 100_000 );

hmm4 = HMM.randomhmm( graph1, calc=Brob, seed=2 )
HMM.setobservations( hmm4, y3 );
HMM.em( hmm4, debug=2, maxiterations=10, keepintermediates=true )

intermediates = hmm4.scratch[:intermediates];
for intermediate in intermediates
    hmm4.transitionprobabilities[:] = intermediate[1:m^2]
    hmm4.means = intermediate[m^2+1:m*(m+1)]
    hmm4.stds = intermediate[m*(m+1)+1:m*(m+2)]
    HMM.clear( hmm4 )
    println( HMM.likelihood( hmm4 )[end] )
end

