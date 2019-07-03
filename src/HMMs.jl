module HMMs

using Models
using Distributions

mutable struct HMM{N, Dist <: Distribution, Calc <: Real, T, U} <: Models.AbstractModel{T,U}
    initialprobabilities::Vector{Calc}
    currentprobabilities::Vector{Calc}
    
    transitionprobabilities::Matrix{Calc}
    stateparameters::Vector{Calc}

    # some linear equality constraints including transition probability unit row sums
    constraintmatrix::Matrix{Calc}
    constraintvector::Vector{Calc}
end

function Models.rand( ::Type{HMM{N,D,C,T,U}}; seed::Int = -1, ) where {N,D,C,T,U}
    if seed >= 0
        Random.seed!( seed )
    end

    initialprobabilities = Vector{C}( rand(N) )
    initialprobabilities /= sum( initialprobabilities )
    currentprobabilities = copy( initialprobabilities )

    transitionprobabilities = Matrix{C}( rand(N,N) )
    transitionprobabilities ./= sum( transitionprobabilities, dims=2 )

    stateparameters = hcat( [randomparameters( D ) for i in 1:N]... )

    p = size( stateparameters, 1 )
    constraintmatrix = [0.0 + (N==1 || div(j-1,N)==i-1) for i in 1:N, j in 1:N^2]
    constraintmatrix = [constraintmatrix zeros(N,N*p)]
    constraintvector = ones(N)

    return HMM( initialprobabilities, currentprobabilities, transitionprobabilities, stateparameters,
                constraintmatrix, constraintvector )
end

end # module
