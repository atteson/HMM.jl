using HMMs
using Brobdingnag
using Random

file = open( joinpath( dirname(dirname(pathof(HMMs))), "data", "thmm3_20040630" ), "r" )
hmm = read( file, HMMs.HMM{HMMs.GenTDist, Brob, Float64} )
close( file )
HMMs.reorder!( hmm )

Random.seed!(1)
y = rand( hmm, 1000 );
HMMs.setobservations( hmm, y );

initial = hmm.initialprobabilities
alpha = HMMs.forwardprobabilities( hmm );
penultimate = convert(Vector{Float64}, alpha[end,:]/sum(alpha[end,:]))

yp = rand( hmm, 1 )

HMMs.initialize( hmm )
@assert( hmm.initialprobabilities == convert( Vector{Float64}, alpha[end,:]/sum(alpha[end,:]) ) )

HMMs.update( hmm, yp[1] )
final = hmm.initialprobabilities

HMMs.clear( hmm )
HMMs.setobservations( hmm, hmm.y )
hmm.initialprobabilities = initial
alpha2 = HMMs.forwardprobabilities( hmm );

penultimate2 = convert(Vector{Float64}, alpha2[end-1,:]/sum(alpha2[end-1,:]))
@assert( penultimate == penultimate )

final2 = convert(Vector{Float64}, alpha2[end,:]/sum(alpha2[end,:]))
@assert( maximum(abs.(final .- final2)) < 1e-12 )

