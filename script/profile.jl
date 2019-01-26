using HMM
using Brobdingnag
using Profile
using ProfileView

hmm1 = HMM.randomhmm( HMM.fullyconnected(2), calc=Brob )
y = HMM.rand( hmm1, 100000 );
hmm2 = HMM.randomhmm( HMM.fullyconnected(2), calc=Brob, seed=2 )
HMM.setobservations( hmm2, y );
HMM.em( hmm2 )

Profile.clear()
hmm2 = HMM.randomhmm( HMM.fullyconnected(2), calc=Brob, seed=2 )
HMM.setobservations( hmm2, y );
@profile HMM.em( hmm2 )
ProfileView.view()

hmm2 = HMM.randomhmm( HMM.fullyconnected(2), calc=Brob, seed=2 )
HMM.setobservations( hmm2, y );
@time HMM.em( hmm2 )
# 2.02s
