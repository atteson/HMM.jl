using HMM

graph = HMM.fullyconnected( 2 )
hmm1 = HMM.randomhmm( graph, seed=1 )
y = HMM.rand( hmm1, 100_000 );

hmm2 = HMM.randomhmm( graph, calc=Brob, seed=2 )
HMM.setobservations( hmm2, y );
HMM.em( hmm2, debug=2 )

hmm1 = HMM.reorder( hmm1 )
hmm2 = HMM.reorder( hmm2 )



