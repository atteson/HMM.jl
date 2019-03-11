using HMMs
using Brobdingnag
using Distributions
using Random

Random.seed!(1)
x = rand( 10, 5 )
name = tempname()
file = open( name, "w" )
HMMs.writearray( file, x )
close(file)
file = open( name, "r" )
y = HMMs.readarray( file, Matrix{Float64} )
close(file)
rm(name)
@assert( x == y )

N = 2
T = 1000
hmm1 = HMMs.randomhmm( HMMs.fullyconnected( N ), calc=BigFloat );
y1 = rand( hmm1, T );
HMMs.setobservations( hmm1, y1 );
alpha = HMMs.forwardprobabilities( hmm1 );
beta = HMMs.backwardprobabilities( hmm1 );

@assert(maximum(abs.(diff([sum(alpha[i,:].*beta[i,:]) for i in 1:T]))) < 1e-8)

pi1 = HMMs.stationary( hmm1 )
@assert( maximum(abs.(pi1 * hmm1.transitionprobabilities - pi1)) < 1e-8 )

gamma = HMMs.conditionalstateprobabilities( hmm1 );
xi = HMMs.conditionaljointstateprobabilities( hmm1 );
@assert( maximum(abs.([maximum(abs.(sum(xi[i,:,:],dims=2) - gamma[i,:])) for i in 1:T-1])) < 1e-8 )
@assert( maximum(abs.([maximum(abs.(sum(xi[i,:,:],dims=1)' - gamma[i+1,:])) for i in 1:T-1])) < 1e-8 )

hmm2 = copy( hmm1 );
@time HMMs.emstep( hmm1, hmm2 )
@assert( maximum(abs.( sum(hmm2.transitionprobabilities,dims=2) .- 1 )) < 1e-8 )

hmm3 = HMMs.randomhmm( hmm1.graph, calc=BigFloat, seed=2 )
y2 = rand( hmm1, 100000 );
HMMs.setobservations( hmm3, y2 );
@time HMMs.em( hmm3, debug=2 )

error = HMMs.permutederror( hmm1, hmm3 )
@assert( error.transitionprobabilities < 1e-2 )
@assert( error.means < 1e-2 )
@assert( error.stds < 1e-2 )

# check derivatives; need to do this before checking statistical convergence
# since MLE has 0 likelihood derivative
function ddffd( hmm, parameter, f, df; delta = 1e-6, relative=false )
    original = parameter[1]
    
    parameter[1] += delta
    HMMs.clear( hmm )
    dpos = copy( f( hmm ) )
    parameter[1] = original
    
    parameter[1] -= delta
    HMMs.clear( hmm )
    dneg = copy( f( hmm ) )
    parameter[1] = original

    fd = (dpos - dneg)/(2*delta)
    dims = length(size(fd))
    outtype = dims == 0 ? Float64 : Array{Float64,dims}
    result = convert(outtype,df .- fd)
    result = abs.(relative ? result ./ (1 .+ abs.(convert(outtype,df))) : result)
    return result
end

function testfd( hmm, parameter, f, df; delta = 1e-6, epsilon = 1e-4, relative=false )
    diffs = ddffd( hmm, parameter, f, df, delta=delta, relative=relative )
    (max, index) = findmax( diffs )
    location = join( Tuple(index), ", " )
    println( "Max is $max (out of $epsilon) at ($location)" )
    @assert( max < epsilon )
end

hmm4 = HMMs.randomhmm( hmm1.graph, calc=Brob, seed=2 )
HMMs.setobservations( hmm4, y2 );

b = copy( HMMs.probabilities( hmm4 ) );
(T,m) = size(b)
dlogb = copy( HMMs.dlogprobabilities( hmm4 ) );
d2logb = copy( HMMs.d2logprobabilities( hmm4 ) );
d2b = copy( HMMs.d2probabilities( hmm4 ) );
alpha = copy( HMMs.forwardprobabilities( hmm4 ) );
dalpha = copy( HMMs.dforwardprobabilities( hmm4 ) );
d2alpha = copy( HMMs.d2forwardprobabilities( hmm4 ) );
l = copy( HMMs.likelihood( hmm4 ) );
dl = copy( HMMs.dlikelihood( hmm4 ) );
d2l = copy( HMMs.d2likelihood( hmm4 ) );
d2logl = copy( HMMs.d2loglikelihood( hmm4 ) );

f1 = m -> log.(HMMs.probabilities(m))
f2 = m -> permutedims(HMMs.probabilities(m) .* permutedims(HMMs.dlogprobabilities(m),[3,2,1]), [3,2,1])
f3 = m -> (HMMs.dlikelihood(m)'./HMMs.likelihood(m))'
# note the constraint to add to 1 is handled elsewhere
for index in 1:m*(m+2)
    if index <= m^2
        (i,j) = divrem(index - 1, m) .+ (1,1)

        parameter = view( hmm4.transitionprobabilities, i, j )
        println( "\n\nTesting transition probability ($i,$j)" )
    elseif index <= m*(m+1)
        i = index - m^2
        parameter = view( hmm4.stateparameters, 1, i )
        println( "\n\nTesting mean $i" )
    else
        i = index - m*(m+1)
        parameter = view( hmm4.stateparameters, 2, i )
        println( "\n\nTesting std $i" )
    end

    print( "dlogb: " )
    testfd( hmm4, parameter, f1, dlogb[index,:,:]' )
    print( "d2logb: " )
    testfd( hmm4, parameter, HMMs.dlogprobabilities, d2logb[index,:,:,:] )
    print( "d2b: " )
    testfd( hmm4, parameter, f2, d2b[index,:,:,:] )
        
    print( "dalpha: " )
    testfd( hmm4, parameter, HMMs.forwardprobabilities, dalpha[index,:,:]', epsilon=1e-2, relative=true )
    print( "d2alpha: " )
    testfd( hmm4, parameter, HMMs.dforwardprobabilities, d2alpha[index,:,:,:], epsilon=1e-2, relative=true )

    print( "dl: " )
    testfd( hmm4, parameter, HMMs.likelihood, dl[index,:], epsilon=1e-2 )
    print( "d2l: " )
    testfd( hmm4, parameter, HMMs.dlikelihood, d2l[index,:,:], epsilon=1e-2 )
    print( "d2logl: " )
    testfd( hmm4, parameter, f3, d2logl[index,:,:], epsilon=1e-2, delta=1e-4, relative=true )
end

@time HMMs.em( hmm4, debug=2 )
error = HMMs.permutederror( hmm1, hmm4 )
@assert( error.transitionprobabilities < 1e-2 )
@assert( error.means < 1e-2 )
@assert( error.stds < 1e-2 )

convert( Matrix{Float64}, HMMs.sandwich( hmm4 ) )

file = open( name, "w" )
write( file, hmm4 )
close(file)
file = open( name, "r" )
hmm4a = read( file, typeof(hmm4) )
close(file)
rm(name)
fields = [:initialprobabilities, :transitionprobabilities, :stateparameters]
@assert( all([==( getfield.( [hmm4,hmm4a], field )... ) for field in fields]) )
@assert( all([==( getfield.( [hmm4.graph,hmm4a.graph], field )... ) for field in [:from,:to]]) )

hmm5 = HMMs.randomhmm( HMMs.fullyconnected(3), calc=Brob, seed=1 )
y3 = rand( hmm5, 100000 );
hmm6 = HMMs.randomhmm( HMMs.fullyconnected(3), calc=Brob, seed=2 )
HMMs.setobservations( hmm6, y3 );
@time HMMs.em( hmm6, debug=2, maxiterations=100 )

graph = HMMs.Digraph( [(1,1),(1,2),(2,2),(2,3),(3,3),(3,1)] )
hmm6 = HMMs.randomhmm( graph, calc=Brob, seed=1 )
y4 = rand( hmm6, 100_000 );
hmm7 = HMMs.randomhmm( HMMs.fullyconnected(3), calc=Brob, seed=4 )
HMMs.setobservations( hmm7, y4 );
@time HMMs.em( hmm7, debug=2 )
HMMs.reorder!(hmm7)
HMMs.reorder!(hmm6)

error = HMMs.permutederror( hmm6, hmm7 )
@assert( error.transitionprobabilities < 1e-2 )
@assert( error.means < 1e-2 )
@assert( error.stds < 1e-2 )

hmm8 = HMMs.randomhmm( graph, calc=Brob, seed=4 )
HMMs.setobservations( hmm8, y4 );
@time HMMs.em( hmm8, debug=2 )
HMMs.reorder!(hmm8)
HMMs.reorder!(hmm6)

error = HMMs.permutederror( hmm6, hmm7 )
@assert( error.transitionprobabilities < 1e-2 )
@assert( error.means < 1e-2 )
@assert( error.stds < 1e-2 )

phmm9 = HMMs.randomhmm( graph, calc=Brob, seed=4 )
hmm9.initialprobabilities = [1.0; zeros(2)]
HMMs.setobservations( hmm9, y4 );
@time HMMs.em( hmm9, debug=2 )
HMMs.reorder!(hmm9)
HMMs.reorder!(hmm6)

error = HMMs.permutederror( hmm6, hmm7 )
@assert( error.transitionprobabilities < 1e-2 )
@assert( error.means < 1e-2 )
@assert( error.stds < 1e-2 )

graph = HMMs.fullyconnected(2)
hmm10 = HMMs.randomhmm( graph, dist=Laplace, calc=Brob, seed=1 )
y10 = rand( hmm10, 10_000 )

hmm11 = HMMs.randomhmm( graph, dist=Laplace, calc=Brob, seed=2 )
HMMs.setobservations( hmm11, y10 )
HMMs.em( hmm11, debug=2 )

HMMs.reorder!( hmm10 )
HMMs.reorder!( hmm11 )

hmm12 = HMMs.randomhmm( HMMs.fullyconnected(3), dist=Laplace, calc=Brob, seed=1 )


hmm13 = HMMs.randomhmm( HMMs.fullyconnected(2), dist=HMMs.GenTDist, calc=Brob, seed=1 )
y13 = rand( hmm13, 10_000 );

hmm14 = HMMs.randomhmm( HMMs.fullyconnected(2), dist=HMMs.GenTDist, calc=Brob, seed=2 )
HMMs.setobservations( hmm14, y13 );

b = copy( HMMs.probabilities( hmm14 ) );
(T,m) = size(b)
dlogb = copy( HMMs.dlogprobabilities( hmm14 ) );
d2logb = copy( HMMs.d2logprobabilities( hmm14 ) );
d2b = copy( HMMs.d2probabilities( hmm14 ) );
alpha = copy( HMMs.forwardprobabilities( hmm14 ) );
dalpha = copy( HMMs.dforwardprobabilities( hmm14 ) );
d2alpha = copy( HMMs.d2forwardprobabilities( hmm14 ) );
l = copy( HMMs.likelihood( hmm14 ) );
dl = copy( HMMs.dlikelihood( hmm14 ) );
d2l = copy( HMMs.d2likelihood( hmm14 ) );
d2logl = copy( HMMs.d2loglikelihood( hmm14 ) );

for index in 1:m*(m+2)
    if index <= m^2
        (i,j) = divrem(index - 1, m) .+ (1,1)

        parameter = view( hmm14.transitionprobabilities, i, j )
        println( "\n\nTesting transition probability ($i,$j)" )
    elseif index <= m*(m+1)
        i = index - m^2
        parameter = view( hmm14.stateparameters, 1, i )
        println( "\n\nTesting mean $i" )
    elseif index <= m*(m+2)
        i = index - m*(m+1)
        parameter = view( hmm14.stateparameters, 2, i )
        println( "\n\nTesting std $i" )
    else
        i = index - m*(m+2)
        parameter = view( hmm14.stateparameters, 2, i )
        println( "\n\nTesting nu $i" )
    end
    
    print( "dlogb: " )
    testfd( hmm14, parameter, f1, dlogb[index,:,:]' )
    print( "d2logb: " )
    testfd( hmm14, parameter, HMMs.dlogprobabilities, d2logb[index,:,:,:] )
    testfd( hmm14, parameter, HMMs.dlogprobabilities, d2logb[index,:,:,:] )
    print( "d2b: " )
    testfd( hmm14, parameter, f2, d2b[index,:,:,:] )
        
    print( "dalpha: " )
    testfd( hmm14, parameter, HMMs.forwardprobabilities, dalpha[index,:,:]', epsilon=1e-2, relative=true )
    print( "d2alpha: " )
    testfd( hmm14, parameter, HMMs.dforwardprobabilities, d2alpha[index,:,:,:], epsilon=1e-2, relative=true )

    print( "dl: " )
    testfd( hmm14, parameter, HMMs.likelihood, dl[index,:], epsilon=1e-2 )
    print( "d2l: " )
    testfd( hmm14, parameter, HMMs.dlikelihood, d2l[index,:,:], epsilon=1e-2 )
    print( "d2logl: " )
    testfd( hmm14, parameter, f3, d2logl[index,:,:], epsilon=1e-2, delta=1e-4, relative=true )
end
