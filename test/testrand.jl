using HMMs
using Brobdingnag
using Distributions
using GCTools

filename = joinpath( dirname(dirname(pathof(HMMs))), "data", "thmm3_20040630" )
hmm = open( filename, "r" ) do file
    read( file, HMMs.HMM{HMMs.GenTDist,Brob,Float64} )
end

n = 1000
path = zeros(n)
@time Distributions.rand!( hmm, path )

GCTools.reset()
@time Distributions.rand!( hmm, path )
GCTools.print()
