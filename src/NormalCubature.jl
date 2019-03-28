using HCubature
using Distributions
using LinearAlgebra

struct Fun
    f::Function
    n::Int
    y::Vector{Float64}
end

Fun( f::Function, n::Int ) = Fun( f, n, zeros(n) )

function (f::Fun)( x::AbstractVector{Float64} )
    factor = 1.0
    for i = 1:f.n
        y = x[i]
        denom = y*(1-y)
        f.y[i] = (2*y-1)/denom
        factor *= (2*y^2 - 2*y + 1)/denom^2
    end
    return f.f( f.y ) * factor
end

function invtransform( x::Float64 )
    if x == 0.0
        return 0.5
    elseif isinf(x)
        return x == -Inf ? 0.0 : 1.0
    else
        return (x-2+sqrt(x^2+4))/(2*x)
    end
end

function nintegrate( f::Fun, a::Vector{Float64}, b::Vector{Float64}; kwargs... )
    for i = 1:f.n
        a[i] = invtransform(a[i])
        b[i] = invtransform(b[i])
    end
    return hcubature( f, a, b; kwargs... )
end

function lenormalcubature( mean, cov, A, b; kwargs... )
    n = length(mean)
    mvn = MvNormal( mean, cov )
    f = Fun( x -> all(A*x .<= b) ? pdf( mvn, x ) : 0.0, n )
    return nintegrate( f, fill(-Inf,n), fill(Inf,n); kwargs... )
end

