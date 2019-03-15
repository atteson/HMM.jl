using HMMs
using Brobdingnag
using SpecialFunctions
using Random

nhmm1 = HMMs.randomhmm( HMMs.fullyconnected(1), calc=Brob, seed=2 )
n = 1_000_000
y1 = rand( nhmm1, n );

nhmm2 = copy( nhmm1 )
HMMs.setobservations( nhmm2, y1 );
HMMs.em( nhmm2, debug=2 )

C = convert( Matrix{Float64}, HMMs.sandwich( nhmm2 ) )
d2logl = HMMs.d2loglikelihood( nhmm2 );
d2logb = HMMs.d2logprobabilities( nhmm2 );
Ihat1 = inv(-convert( Matrix{Float64}, d2logl[2:3,2:3,end] )/n)
Ihat2 = inv(-convert( Matrix{Float64}, sum( d2logb[2:3,2:3,1,1:end], dims=3 )[:,:,1,1] )/n)
@assert( maximum(abs.(Ihat1 - n*C[2:3,2:3])) < 0.01 )
@assert( maximum(abs.(Ihat2 - n*C[2:3,2:3])) < 0.01 )
sigma = nhmm2.stateparameters[2,1]
@assert( maximum(abs.(inv([1/sigma^2 0.0; 0.0 2/sigma^2]) - n*C[2:3,2:3])) < 0.01 )

nhmm3 = HMMs.randomhmm( HMMs.fullyconnected(2), calc=Brob, seed=2 )
n = 1_000_000
y3 = rand( nhmm3, n );

pathname = joinpath( homedir(), "experiments", "test", "nhmms2" )
filename = joinpath( pathname, "hmm2.2" )
if !isfile( filename )
    nhmm4 = copy( nhmm3 )
    HMMs.setobservations( nhmm4, y3 );
    HMMs.em( nhmm4, debug=2, keepintermediates=true, acceleration=10 )

    mkpath( pathname )
    open( filename, "w" ) do file
        HMMs.write( file, nhmm4 )
    end
else
    nhmm4 = open( filename, "r" ) do file
        read( file, HMMs.GaussianHMM{Brob, Float64} )
    end
end

C = convert( Matrix{Float64}, HMMs.sandwich( nhmm4 ) )
l = HMMs.likelihood( nhmm4 );
dl = HMMs.dlikelihood( nhmm4 );
dlogl = dl ./ l';
ddlogl = diff( dlogl, dims=2 )

J = convert( Matrix{Float64}, sum([ddlogl[:,t] * ddlogl[:,t]' for t in 1:size(ddlogl,2)])/n)

d2logl = HMMs.d2loglikelihood( nhmm4 );
I = -convert( Matrix{Float64}, d2logl[:,:,end] )/n

@assert( maximum(abs.(I[5:8,5:8] - J[5:8,5:8])) < 0.001 )

Ihat1 = inv( I )
@assert( maximum(abs.(Ihat1 - n*C)) < 0.01 )


hmm1 = HMMs.randomhmm( HMMs.fullyconnected(1), dist=HMMs.GenTDist, calc=Brob, seed=1 )
n = 1_000_000
y1 = rand( hmm1, n )

hmm2 = copy( hmm1 )
HMMs.setobservations( hmm2, y1 )
HMMs.em( hmm2, debug=2 )

hmm1
hmm2

C = convert( Matrix{Float64}, HMMs.sandwich( hmm2 ) )
d2logl = HMMs.d2loglikelihood( hmm2 );
d2logb = HMMs.d2logprobabilities( hmm2 );
Ihat1 = inv(-convert( Matrix{Float64}, d2logl[2:4,2:4,end] )/n)
Ihat2 = inv(-convert( Matrix{Float64}, sum( d2logb[2:4,2:4,1,1:end], dims=3 )[:,:,1,1] )/n)
@assert( maximum(abs.((Ihat1 - n*C[2:4,2:4]) ./ (1 .+ Ihat1))) < 0.01 )

n*C[2:4,2:4]

(mu, sigma, nu) = hmm1.stateparameters
I = [
(nu+1)/(nu+3)*mu^2/sigma^2 0.0 0.0;
0.0 (nu+1)/((nu+3)*2*sigma^4) 1/((nu+3)*(nu+1)*sigma^2);
0.0 1/((nu+3)*(nu+1)*sigma^2) -(polygamma(1, (nu+1)/2)/2 - polygamma(1, nu/2)/2 + 1/(nu*(nu+1)) - 1/(nu+1) + (nu+2)/(nu*(nu+3)))/2;
]

inv(I)
n*C[2:4,2:4]

d2logl = HMMs.d2loglikelihood( hmm2 );
-convert( Matrix{Float64}, d2logl[2:4,2:4,end]/n )
n*C[2:4,2:4]
inv(I)

d2logb = HMMs.d2logprobabilities( hmm2 );
sum(d2logb, dims=4)
Ihat1 = reshape( sum(d2logb, dims=4), (4,4) )[2:4,2:4]
Ihat2 = d2logl[2:4,2:4,end]

@assert( convert( Float64, maximum(abs.(Ihat1 - Ihat2)./Ihat1) ) < 0.005 )

hmm3 = HMMs.randomhmm( HMMs.fullyconnected(2), dist=HMMs.GenTDist, calc=Brob, seed=1 )
n = 1_000_000
y3 = rand( hmm3, n );

hmm4 = copy( hmm3 )
HMMs.setobservations( hmm4, y3 );
HMMs.em( hmm4, debug=2 )

C = convert( Matrix{Float64}, HMMs.sandwich( hmm4 ) )
d2logl = HMMs.d2loglikelihood( hmm4 );
Ihat1 = convert( Matrix{Float64}, d2logl[:,:,end]/n )
n*C
