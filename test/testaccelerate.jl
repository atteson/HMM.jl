using HMM
using Brobdingnag
using PyPlot

m = 3
graph1 = HMM.fullyconnected( m )

hmm1 = HMM.randomhmm( graph1, seed=1 )
y1 = HMM.rand( hmm1, 100_000 );

hmm2 = HMM.randomhmm( graph1, calc=Brob, seed=2 )
HMM.setobservations( hmm2, y1 );
HMM.em( hmm2, debug=2, keepintermediates=true )
# 631 iterations, 232 seconds
hmm2 = HMM.randomhmm( graph1, calc=Brob, seed=2 )
HMM.setobservations( hmm2, y1 );
HMM.em( hmm2, debug=2, keepintermediates=true, acceleration=10 )
# 41 iterations, 16 seconds

graph3 = HMM.Digraph( [1,1,2,2,2,3,3], [1,2,1,2,3,2,3] )
hmm3 = HMM.randomhmm( graph3, seed=1 )
y3 = rand( hmm3, 100_000 );

hmm4 = HMM.randomhmm( graph1, calc=Brob, seed=2 )
HMM.setobservations( hmm4, y3 );
HMM.em( hmm4, debug=2 )
# 7406 iterations, 2792 seconds
hmm4 = HMM.randomhmm( graph1, calc=Brob, seed=2 )
HMM.setobservations( hmm4, y3 );
HMM.em( hmm4, debug=2, keepintermediates=true, acceleration=10 )
# 4512 iterations, 1782 seconds, exp(-17394.488432664482) likelihood

C = convert( Matrix{Float64}, HMM.sandwich( hmm4 ) )

intermediates = hmm4.scratch[:intermediates];
for intermediate in intermediates
    hmm4.transitionprobabilities[:] = intermediate[1:m^2]
    hmm4.means = intermediate[m^2+1:m*(m+1)]
    hmm4.stds = intermediate[m*(m+1)+1:m*(m+2)]
    HMM.clear( hmm4 )
    println( HMM.likelihood( hmm4 )[end] )
end

hmm4 = HMM.randomhmm( graph1, calc=Brob, seed=2 )
HMM.setobservations( hmm4, y3 );
HMM.em( hmm4, debug=2, maxiterations=1000, keepintermediates=true )

intermediates = hmm4.scratch[:intermediates];

clf()
for i = 1:length(intermediates[1])
    plot( getindex.( intermediates, i )[200:end] )
end


x = getindex.(intermediates[200:299], 1)
y = diff(x)
gamma = log(y[1]./y[2])
beta = y[1]/(exp(-gamma[1])-1)
alpha = x[1] - beta

[x alpha .+ beta*exp.(-gamma[1] .* collect(0:99))]

estimates = Brob[]
for i = 1:length(intermediates[1])
    x = getindex.(intermediates[200:299], i)
    y = diff(x)
    gamma = log(y[1]./y[2])
    beta = y[1]/(exp(-gamma[1])-1)
    alpha = x[1] - beta
    push!( estimates, alpha )
end

hmm4.transitionprobabilities[:] = estimates[1:m^2]
hmm4.means = estimates[m^2+1:m*(m+1)]
hmm4.stds = estimates[m*(m+1)+1:m*(m+2)]
HMM.clear( hmm4 )
HMM.likelihood( hmm4 )[end]

hmm4.transitionprobabilities[:] = intermediates[200][1:m^2]
hmm4.means = intermediates[200][m^2+1:m*(m+1)]
hmm4.stds = intermediates[200][m*(m+1)+1:m*(m+2)]
HMM.clear( hmm4 )
HMM.likelihood( hmm4 )[end]




