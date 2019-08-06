using Dependencies
using Serialization
using Models
using HMMs
using Brobdingnag
using LinearAlgebra
using Dates
using Distributions
using Random
using Distributed

criterion( model ) = HMMs.likelihood(model.model)[end]

dir = Dependencies.defaultdir
objects = Dict{String,Any}()
for root in readdir(dir)
    file = joinpath( dir, root )
    io = open( file )
    objects[file] = deserialize( io )
    close( io )
end

statecount( ::HMMs.HMM{N} ) where {N} = N
statecount( m::Models.AbstractModel ) = statecount( Models.rootmodel( m ) )

good = statecount.( values( objects ) ) .== 3
goodmodels = collect( values( objects ) )[good];

l = length.( getfield.( Models.rootmodel.( goodmodels ), :y ) )
l .- minimum(l)

hmm = Models.rootmodel( goodmodels[3] )
C = Models.sandwich( hmm )
(eva, eve) = eigen( C )

T = real.(sqrt(C))
newhmm = deepcopy( hmm )
n = size(T,1)
p = Models.getcompressedparameters( newhmm )
Models.setcompressedparameters!( newhmm, p + T * randn(n) )

n = length(hmm.y)
sample = fill( NaN, n )
Distributions.rand!( hmm, sample )

Random.seed!(1)
N = 100
hmms = typeof(hmm)[]
for i = 1:N
    push!( hmms, rand( typeof(hmm) ) )
    HMMs.setobservations( hmms[end], sample )
end

processes = 50
addprocs( processes )
modules = [:HMMs, :Brobdingnag, :Models]
futures = Future[]
for pid in workers()
    for moduletoeval in modules
        push!( futures, remotecall( Core.eval, pid, Main, Expr(:using,Expr(:.,moduletoeval)) ) )
    end
end
for future in futures
    wait(future)
end
@time hmms = pmap( HMMs.em, hmms );

likelihoods = [HMMs.likelihood(hmm)[end] for hmm in hmms]
perm = sortperm(likelihoods)
likelihoods[perm]
hmms[perm[end]]
hmms[perm[end-1]]
hmms[perm[90]]

C = Models.sandwich( hmms[perm[end]] )
(eva, eve) = eigen(C)
B = hmms[perm[end]].basis
B * C * B'

hmm2 = deepcopy( hmm )
HMMs.setobservations( hmm2, sample )
HMMs.em( hmm2, debug=2 )
HMMs.likelihood( hmm2 )[end]
sort(likelihoods)[end-10:end]

Random.seed!(1)
N = 100
ys = [zeros(n) for i in 1:N]
[rand!( hmm, ys[i] ) for i in 1:N]
hmms = [typeof(hmm)[] for i in 1:N]
for i = 1:N
    for j = 1:N
        push!( hmms[i], rand( typeof(hmm) ) )
        HMMs.setobservations( hmms[i][end], ys[i] )
    end
end
hmms = vcat( hmms... );

@everywhere function loudfit( i, hmm )
    println( "Fitting HMM $i" )
    HMMs.em( hmm )
    println( "Done fitting HMM $i" )
    return hmm
end

fithmms = pmap( loudfit, 1:N*N, hmms );
