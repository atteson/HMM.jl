using HMM
using Brobdingnag
using Random

Random.seed!(1)
x = rand( 10, 5 )
name = tempname()
file = open( name, "w" )
HMM.writearray( file, x )
close(file)
file = open( name, "r" )
y = HMM.readarray( file, Matrix{Float64} )
close(file)
rm(name)
@assert( x == y )

N = 2
T = 1000
hmm1 = HMM.randomhmm( HMM.fullyconnected( N ), calc=BigFloat );
y1 = rand( hmm1, T );
HMM.setobservations( hmm1, y1 );
alpha = HMM.forwardprobabilities( hmm1 );
beta = HMM.backwardprobabilities( hmm1 );

@assert(maximum(abs.(diff([sum(alpha[i,:].*beta[i,:]) for i in 1:T]))) < 1e-8)

pi1 = HMM.stationary( hmm1 )
@assert( maximum(abs.(pi1 * hmm1.transitionprobabilities - pi1)) < 1e-8 )

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
@time HMM.em( hmm4, debug=2 )

error = HMM.permutederror( hmm1, hmm4 )
@assert( error.transitionprobabilities < 1e-2 )
@assert( error.means < 1e-2 )
@assert( error.stds < 1e-2 )

# check probability derivatives
epsilon = 1e-6
b = copy( HMM.probability( hmm4 ) )
(T,m) = size(b)
db = copy( HMM.dprobability( hmm4 ) )

alpha = copy( HMM.forwardprobabilities( hmm4 ) )
dalpha = copy( HMM.dforwardprobabilities( hmm4 ) )

for i = 1:m
    # first versus transition probabilities
    for j = 1:m
        # note the constraint to add to 1 is handled elsewhere
        p = hmm4.transitionprobabilities[i,j]
        
        hmm4.transitionprobabilities[i,j] += epsilon
        HMM.clear( hmm4 )
        bp = copy( HMM.probability( hmm4 ) )
        alphap = copy( HMM.forwardprobabilities( hmm4 ) )
        hmm4.transitionprobabilities[i,j] = p
        
        hmm4.transitionprobabilities[i,j] -= epsilon
        HMM.clear( hmm4 )
        bm = copy( HMM.probability( hmm4 ) )
        alpham = copy( HMM.forwardprobabilities( hmm4 ) )
        hmm4.transitionprobabilities[i,j] = p
        
        fddb = (bp - bm)/(2*epsilon)
        @assert( maximum(abs.(convert(Matrix{Float64},fddb - db[(i-1)*m + j,:,:]'))) < 1e-8 )
        
        fddalpha = (alphap - alpham)/(2*epsilon)
        @assert( maximum(abs.(convert( Matrix{Float64}, (fddalpha - dalpha[(i-1)*m + j,:,:]')./(1 .+ fddalpha) ))) < 1e-2 )
    end
    
    # now mean
    mu = hmm4.means[i]
    
    hmm4.means[i] += epsilon
    HMM.clear( hmm4 )
    bp = copy( HMM.probability( hmm4 ) )
    alphap = copy( HMM.forwardprobabilities( hmm4 ) )
    hmm4.means[i] = mu
    
    hmm4.means[i] -= epsilon
    HMM.clear( hmm4 )
    bm = copy( HMM.probability( hmm4 ) )
    alpham = copy( HMM.forwardprobabilities( hmm4 ) )
    hmm4.means[i] = mu

    fddb = (bp - bm)/(2*epsilon)
    @assert( maximum(abs.(convert(Matrix{Float64},fddb - db[m^2 + i,:,:]'))) < 1e-4 )

    fddalpha = (alphap - alpham)/(2*epsilon)
    @assert( maximum(abs.(convert(Matrix{Float64},(fddalpha - dalpha[m^2 + i,:,:]')./(1 .+ fddalpha)))) < 1e-4 )
    
    # now standard deviation
    sigma = hmm4.stds[i]
    
    hmm4.stds[i] += epsilon
    HMM.clear( hmm4 )
    bp = copy( HMM.probability( hmm4 ) )
    alphap = copy( HMM.forwardprobabilities( hmm4 ) )
    hmm4.stds[i] = sigma
    
    hmm4.stds[i] -= epsilon
    HMM.clear( hmm4 )
    bm = copy( HMM.probability( hmm4 ) )
    alpham = copy( HMM.forwardprobabilities( hmm4 ) )
    hmm4.stds[i] = sigma
    
    fddb = (bp - bm)/(2*epsilon)
    @assert( maximum(abs.(convert(Matrix{Float64},fddb - db[m*(m+1) + i,:,:]'))) < 1e-4 )

    fddalpha = (alphap - alpham)/(2*epsilon)
    @assert( maximum(abs.(convert(Matrix{Float64},(fddalpha - dalpha[m*(m+1) + i,:,:]')./(1 .+ fddalpha)))) < 1e-3 )
end

file = open( name, "w" )
write( file, hmm4 )
close(file)
file = open( name, "r" )
hmm4a = read( file, typeof(hmm4) )
close(file)
rm(name)
fields = [:initialprobabilities, :transitionprobabilities, :means, :stds]
@assert( all([==( getfield.( [hmm4,hmm4a], field )... ) for field in fields]) )
@assert( all([==( getfield.( [hmm4.graph,hmm4a.graph], field )... ) for field in [:from,:to]]) )

hmm5 = HMM.randomhmm( HMM.fullyconnected(3), calc=Brob, seed=1 )
y3 = rand( hmm5, 100000 );
hmm6 = HMM.randomhmm( HMM.fullyconnected(3), calc=Brob, seed=2 )
HMM.setobservations( hmm6, y3 );
@time HMM.em( hmm6, debug=2, maxiterations=100 )

graph = HMM.Digraph( [(1,1),(1,2),(2,2),(2,3),(3,3),(3,1)] )
hmm6 = HMM.randomhmm( graph, calc=Brob, seed=1 )
y4 = rand( hmm6, 100_000 );
hmm7 = HMM.randomhmm( HMM.fullyconnected(3), calc=Brob, seed=4 )
HMM.setobservations( hmm7, y4 );
@time HMM.em( hmm7, debug=2 )
HMM.reorder(hmm7)
HMM.reorder(hmm6)

error = HMM.permutederror( hmm6, hmm7 )
@assert( error.transitionprobabilities < 1e-2 )
@assert( error.means < 1e-2 )
@assert( error.stds < 1e-2 )

hmm8 = HMM.randomhmm( graph, calc=Brob, seed=4 )
HMM.setobservations( hmm8, y4 );
@time HMM.em( hmm8, debug=2 )
HMM.reorder(hmm8)
HMM.reorder(hmm6)

error = HMM.permutederror( hmm6, hmm7 )
@assert( error.transitionprobabilities < 1e-2 )
@assert( error.means < 1e-2 )
@assert( error.stds < 1e-2 )

hmm9 = HMM.randomhmm( graph, calc=Brob, seed=4 )
hmm9.initialprobabilities = [1.0; zeros(2)]
HMM.setobservations( hmm9, y4 );
@time HMM.em( hmm9, debug=2 )
HMM.reorder(hmm9)
HMM.reorder(hmm6)

error = HMM.permutederror( hmm6, hmm7 )
@assert( error.transitionprobabilities < 1e-2 )
@assert( error.means < 1e-2 )
@assert( error.stds < 1e-2 )

