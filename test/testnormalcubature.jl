using HMMs
using LinearAlgebra
using Distributions

mean = zeros(2)
cov = zeros(2,2) + I
A = [zeros(1,1) ones(1,1)]
b = [0.0]
(x,tol) = HMMs.lenormalcubature( mean, cov, A, b )
@assert( abs(x - 0.5) < tol )

A = zeros(2,2) + I
b = zeros(2)
(x,tol) = HMMs.lenormalcubature( mean, cov, A, b )
@assert( abs(x - 0.25) < tol )

b = [0.0,1.0]
(x,tol) = HMMs.lenormalcubature( mean, cov, A, b )
n1 = cdf( Normal(0,1), 1 )
@assert( abs(x - n1/2) < tol )

b = [1.0,1.0]
(x,tol) = HMMs.lenormalcubature( mean, cov, A, b )
@assert( abs(x - n1^2) < tol )

A = [1.0 0.0;-1.0 -1.0]
b = zeros(2)
@time (x,tol) = HMMs.lenormalcubature( mean, cov, A, b, maxevals=10^7 )
# tol doesn't seem to work for this one
@assert( abs(x - 1/8) < 1e-5 )
