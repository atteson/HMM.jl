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
@time HMM.emstep( hmm1, hmm2 )
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

# check derivatives; need to do this before checking statistical convergence
# since MLE has 0 likelihood derivative
function ddffd( hmm, parameter, f, df; delta = 1e-6, relative=false )
    original = parameter[1]
    
    parameter[1] += delta
    HMM.clear( hmm )
    dpos = copy( f( hmm ) )
    parameter[1] = original
    
    parameter[1] -= delta
    HMM.clear( hmm )
    dneg = copy( f( hmm ) )
    parameter[1] = original

    fd = (dpos - dneg)/(2*delta)
    dims = length(size(fd))
    outtype = dims == 0 ? Float64 : Array{Float64,dims}
    result = convert(outtype,df .- fd)
    result = abs.(relative ? result : result ./ (1 .+ convert(outtype,df)))
    return result
end

function testfd( hmm, parameter, f, df; delta = 1e-6, epsilon = 1e-4, relative=false, string="" )
    diffs = ddffd( hmm, parameter, f, df, delta=delta, relative=relative )
    first = findfirst( abs.(diffs) .>= epsilon )
    if first != nothing
        location = join( Tuple(first), ", " )
        @assert( false, "Error in $string at $location (difference of $(abs.(diffs[first])))" )
    end
end

b = copy( HMM.probabilities( hmm4 ) );
(T,m) = size(b)
dlogb = copy( HMM.dlogprobabilities( hmm4 ) );
d2logb = copy( HMM.d2logprobabilities( hmm4 ) );
d2b = copy( HMM.d2probabilities( hmm4 ) );
alpha = copy( HMM.forwardprobabilities( hmm4 ) );
dalpha = copy( HMM.dforwardprobabilities( hmm4 ) );
d2alpha = copy( HMM.d2forwardprobabilities( hmm4 ) );
dl = HMM.dlikelihood( hmm4 )

f1 = m -> log.(HMM.probabilities(m))
f2 = m -> permutedims(HMM.probabilities(m) .* permutedims(HMM.dlogprobabilities(m),[3,2,1]), [3,2,1])
for i = 1:m
    # first versus transition probabilities
    for j = 1:m
        # note the constraint to add to 1 is handled elsewhere
        parameter = view( hmm4.transitionprobabilities, i, j )
        index1 = (i-1)*m + j
        s = "transition probability ($i,$j)"
        println( "Testing $s" )
        
        testfd( hmm4, parameter, f1, dlogb[index1,:,:]', string=s )
        testfd( hmm4, parameter, HMM.dlogprobabilities, d2logb[index1,:,:,:], string=s )
        testfd( hmm4, parameter, f2, d2b[index1,:,:,:], string=s )
        
        testfd( hmm4, parameter, HMM.forwardprobabilities, dalpha[index1,:,:]', epsilon=1e-2, relative=true, string=s )
        testfd( hmm4, parameter, HMM.dforwardprobabilities, d2alpha[index1,:,:,:], epsilon=1e-2, relative=true, string=s )
        testfd( hmm4, parameter, HMM.likelihood, dl[index1], epsilon=1e-2, string=s )
    end
    
    # now mean
    parameter = view( hmm4.means, i )
    index1 = m^2 + i
    s = "mean $i"
    println( "Testing $s" )
    
    testfd( hmm4, parameter, f1, dlogb[index1,:,:]', string=s )
    testfd( hmm4, parameter, HMM.dlogprobabilities, d2logb[index1,:,:,:], string=s )
    testfd( hmm4, parameter, f2, d2b[index1,:,:,:], string=s )
    
    testfd( hmm4, parameter, HMM.forwardprobabilities, dalpha[index1,:,:]', epsilon=1e-3, relative=true, string=s )
    testfd( hmm4, parameter, HMM.dforwardprobabilities, d2alpha[index1,:,:,:], epsilon=1e-2, relative=true, string=s )
    testfd( hmm4, parameter, HMM.likelihood, dl[index1], epsilon=1e-3, string=s )
    
    # now standard deviation
    parameter = view( hmm4.stds, i )
    index1 = m*(m+1) + i
    s = "std $i"
    println( "Testing $s" )
    
    testfd( hmm4, parameter, f1, dlogb[index1,:,:]', string=s )
    testfd( hmm4, parameter, HMM.dlogprobabilities, d2logb[index1,:,:,:], string=s )
    testfd( hmm4, parameter, f2, d2b[index1,:,:,:], string=s )
    
    testfd( hmm4, parameter, HMM.forwardprobabilities, dalpha[index1,:,:]', epsilon=1e-3, relative=true, string=s )
    testfd( hmm4, parameter, HMM.dforwardprobabilities, d2alpha[index1,:,:,:], epsilon=1e-2, relative=true, string=s )
    testfd( hmm4, parameter, HMM.likelihood, dl[index1], epsilon=1e-3, string=s )
end

@time HMM.em( hmm4, debug=2 )
error = HMM.permutederror( hmm1, hmm4 )
@assert( error.transitionprobabilities < 1e-2 )
@assert( error.means < 1e-2 )
@assert( error.stds < 1e-2 )

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

