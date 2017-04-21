using HMM
using Base.Test

# write your own tests here
srand(2)
N = 2
T = 1000
hmm1 = HMM.randomhmm( HMM.fullyconnected( N ), float=BigFloat );
y = rand( hmm1, T );
HMM.setobservations( hmm1, y );
alpha = HMM.forwardprobabilities( hmm1 );
beta = HMM.backwardprobabilities( hmm1 );

assert(maxabs(diff([sum(alpha[i].*beta[i]) for i in 1:T])) < 1e-8)

gamma = HMM.conditionalstateprobabilities( hmm1 );
xi = HMM.conditionaljointstateprobabilities( hmm1 );
assert( maxabs([maxabs(sum(xi[i],2) - gamma[i,:]) for i in 1:T-1]) < 1e-8 )
assert( maxabs([maxabs(sum(xi[i],1)' - gamma[i+1,:]) for i in 1:T-1]) < 1e-8 )

hmm2 = copy( hmm1 );
HMM.emstep( hmm1, hmm2 )
assert( maxabs( sum(hmm2.transitionprobabilities,2) - 1 ) < 1e-8 )
