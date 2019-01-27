using HMM
using Brobdingnag
using Profile
using ProfileView
using GCTools

hmm1 = HMM.randomhmm( HMM.fullyconnected(2), calc=Brob )
y = HMM.rand( hmm1, 100000 );
hmm2 = HMM.randomhmm( HMM.fullyconnected(2), calc=Brob, seed=2 )
HMM.setobservations( hmm2, y );
HMM.em( hmm2 )

hmm2 = HMM.randomhmm( HMM.fullyconnected(2), calc=Brob, seed=2 )
HMM.setobservations( hmm2, y );
GCTools.reset()
@time HMM.em( hmm2 )
GCTools.print()

Profile.clear()
hmm2 = HMM.randomhmm( HMM.fullyconnected(2), calc=Brob, seed=2 )
HMM.setobservations( hmm2, y );
@profile HMM.em( hmm2 )
ProfileView.view()

hmm2 = HMM.randomhmm( HMM.fullyconnected(2), calc=Brob, seed=2 )
HMM.setobservations( hmm2, y );
@time HMM.em( hmm2 )
# 2.02s

x = rand( 100000 )
bx = convert( Vector{Brob}, x )
GCTools.reset()
GCTools.push!( :all )
for i = 2:length(bx)
    bx[i].positive = bx[i-1].positive
    bx[i].log = bx[i-1].log
end
GCTools.pop!()
GCTools.print()

t = [T1(rand(1:1_000_000), rand(1:1_000_000)) for i in 1:100_000]
