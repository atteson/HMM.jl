using Distributed

workers = addprocs(20)

@everywhere begin
    using HMMs
    using Brobdingnag
    using SpecialFunctions
    using Random
    using Distributed
    using Models
end

@everywhere function fithmm( n, nhmm, seed )
    y = rand( nhmm, 1:n, seed=seed  );
    hmm = deepcopy( nhmm )
    HMMs.setobservations( hmm, y )
    HMMs.em( hmm, debug=2 )
    p = Models.getcompressedparameters( hmm )
    C = Models.sandwich( hmm )
    return (p, C)
end

hmmtype = HMMs.HMM{1,HMMs.GenTDist,Brob,Float64,Int}
nhmm = rand( hmmtype )
n = 1_000_000

N = 100
data1 = pmap( fithmm, fill(n,N), fill(nhmm,N), 1:N )

using StatsBase

p = Models.getcompressedparameters( nhmm )
vs = Vector{Float64}[]
for i = 1:N
    (p1, C1) = data1[i]
    sC1 = real.(sqrt(C1))
    @assert( maximum(abs.(C1 - sC1 * sC1')) < 1e-10 )
    push!( vs, inv(sC1) * (p - p1) )
end

hmmtype = HMMs.HMM{2,HMMs.GenTDist,Brob,Float64,Int}
nhmm = rand( hmmtype )
n = 1_000_000

N = 100
data2 = pmap( fithmm, fill(n,N), fill(nhmm,N), 1:N )

p = Models.getcompressedparameters( nhmm )
vs = Vector{Float64}[]
for i = 1:N
    (p1, C1) = data1[i]
    sC1 = real.(sqrt(C1))
    @assert( maximum(abs.(C1 - sC1 * sC1')) < 1e-10 )
    push!( vs, inv(sC1) * (p - p1) )
end
