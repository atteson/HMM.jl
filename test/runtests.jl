using HMM
using Brobdingnag
using GCTools

# write your own tests here
N = 2
T = 1000
hmm1 = HMM.randomhmm( HMM.fullyconnected( N ), calc=BigFloat );
y1 = rand( hmm1, T );
HMM.setobservations( hmm1, y1 );
alpha = HMM.forwardprobabilities( hmm1 );
beta = HMM.backwardprobabilities( hmm1 );

@assert(maximum(abs.(diff([sum(alpha[i,:].*beta[i,:]) for i in 1:T]))) < 1e-8)

gamma = HMM.conditionalstateprobabilities( hmm1 );
xi = HMM.conditionaljointstateprobabilities( hmm1 );
@assert( maximum(abs.([maximum(abs.(sum(xi[i,:,:],dims=2) - gamma[i,:])) for i in 1:T-1])) < 1e-8 )
@assert( maximum(abs.([maximum(abs.(sum(xi[i,:,:],dims=1)' - gamma[i+1,:])) for i in 1:T-1])) < 1e-8 )

hmm2 = copy( hmm1 );
HMM.emstep( hmm1, hmm2 )
@assert( maximum(abs.( sum(hmm2.transitionprobabilities,dims=2) .- 1 )) < 1e-8 )

hmm3 = HMM.randomhmm( hmm1.graph, calc=BigFloat, seed=2 )
y2 = rand( hmm1, 100000 );
HMM.setobservations( hmm3, y2 );
@time HMM.em( hmm3, debug=2 )

error = HMM.permutederror( hmm1, hmm3 )
@assert( error.transitionprobabilities < 1e-2 )
@assert( error.means < 1e-2 )
@assert( error.stds < 1e-2 )

hmm4 = HMM.randomhmm( hmm1.graph, calc=Brob, seed=2 )
HMM.setobservations( hmm4, y2 );
GCTools.reset()
@time HMM.em( hmm4, debug=2 )
GCTools.print()

error = HMM.permutederror( hmm1, hmm4 )
@assert( error.transitionprobabilities < 1e-2 )
@assert( error.means < 1e-2 )
@assert( error.stds < 1e-2 )

hmm5 = HMM.randomhmm( HMM.fullyconnected(3), calc=Brob, seed=1 )
y3 = rand( hmm5, 100000 )
hmm6 = HMM.randomhmm( HMM.fullyconnected(3), calc=Brob, seed=2 )
HMM.setobservations( hmm6, y3 )
@time HMM.em( hmm6, debug=2, maxiterations=100 )

