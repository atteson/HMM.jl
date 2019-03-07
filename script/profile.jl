using HMMs
using Brobdingnag
using Profile
using ProfileView
using GCTools

hmm1 = HMMs.randomhmm( HMMs.fullyconnected(2), calc=Brob )
y = HMMs.rand( hmm1, 100000 );

hmm2 = HMMs.randomhmm( HMMs.fullyconnected(2), calc=Brob, seed=2 )
HMMs.setobservations( hmm2, y );
@time HMMs.em( hmm2 )
hmm2 = HMMs.randomhmm( HMMs.fullyconnected(2), calc=Brob, seed=2 )
HMMs.setobservations( hmm2, y );
GCTools.reset()
@time HMMs.em( hmm2 )
GCTools.print()

Profile.clear()
hmm2 = HMMs.randomhmm( HMMs.fullyconnected(2), calc=Brob, seed=2 )
HMMs.setobservations( hmm2, y );
@profile HMMs.em( hmm2 )
ProfileView.view()

hmm2 = HMMs.randomhmm( HMMs.fullyconnected(2), calc=Brob, seed=2 )
HMMs.setobservations( hmm2, y );
@time HMMs.em( hmm2 )
# 2.02s
# 1.40s

