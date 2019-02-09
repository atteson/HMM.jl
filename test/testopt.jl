using Ipopt
using MathProgBase
using Random

struct Quadratic <: MathProgBase.AbstractNLPEvaluator
    V::Matrix{Float64}
    m::Vector{Float64}
    A::Matrix{Float64}
end

function MathProgBase.initialize( ::Quadratic, requested_features::Vector{Symbol} )
    unimplemented = setdiff( requested_features, [:Grad,:Jac] )
    if !isempty( unimplemented )
        error( "The following features aren't implemented: " * join( string.(unimplemented), ", " ) )
    end
end

MathProgBase.features_available( ::Quadratic ) = [:Grad]

MathProgBase.eval_f( q::Quadratic, x ) = (x' * 0.5 * q.V + q.m') * x

function MathProgBase.eval_g( q::Quadratic, g, x )
    g[:] = q.A * x
end

function MathProgBase.eval_grad_f( q::Quadratic, grad_f, x )
    grad_f[:] = q.V * x + q.m
end

MathProgBase.jac_structure( q::Quadratic ) =
    ((m,n) -> ([(i % m) + 1 for i in 1:m*n], vcat([fill(i,m) for i in 1:n]...)))(size(q.A)...)

function MathProgBase.eval_jac_g( q::Quadratic, J, x )
    J[:] = q.A
end

solver = IpoptSolver()

model = MathProgBase.NonlinearModel(solver)

Random.seed!(1)
S = rand( 3, 3 )
V = S'*S
m = rand(3)
A = rand(1,3)
b = rand(1)
q = Quadratic(V,m,A)
x0 = rand(3)
MathProgBase.loadproblem!(model, 3, 1, fill(-Inf,3), fill(Inf,3), fill(-Inf,1), b, :Min, q)
MathProgBase.setwarmstart!( model, rand(3) )
MathProgBase.optimize!(model)
@assert( MathProgBase.status(model) == :Optimal )
x = MathProgBase.getsolution(model)

@assert( maximum(abs.(x + q.V\q.m)) < 1e-8 )

if hasmethod( MathProgBase.freemodel!, Tuple{typeof(model)} )
    MathProgBsae.freemodel!( model )
end

b = [-5.0]
model = MathProgBase.NonlinearModel(solver)
MathProgBase.loadproblem!(model, 3, 1, fill(-Inf,3), fill(Inf,3), fill(-Inf,1), b, :Min, q)
x0 = rand(3)
MathProgBase.setwarmstart!( model, x0 )
MathProgBase.optimize!(model)
@assert( MathProgBase.status(model) == :Optimal )
x = MathProgBase.getsolution(model)

@assert( q.A * x <= b .+ 1e-6 )

lambda = -((q.A * inv(q.V) * q.m + b)/(q.A * inv(q.V) * q.A'))[1]
@assert( maximum(abs.(x - inv(V)*(- q.m' - lambda * q.A)')) < 1e-6 )

if hasmethod( MathProgBase.freemodel!, Tuple{typeof(model)} )
    MathProgBsae.freemodel!( model )
end

# the following optimization takes many iterations
# I'm not sure why
#A = rand(2,3)
#b = rand(2)
#q = Quadratic( V, m, A )
#model = MathProgBase.NonlinearModel(solver)
#MathProgBase.loadproblem!(model, 3, 2, fill(-Inf,3), fill(Inf,3), fill(-Inf,2), b, :Min, q)
#x0 = rand(3)
#MathProgBase.setwarmstart!( model, x0
#MathProgBase.optimize!(model)

#if hasmethod( MathProgBase.freemodel!, Tuple{typeof(model)} )
#    MathProgBsae.freemodel!( model )
#end

# looseing the constraints does work
Ap = A[1:1,:]
bp = b[1:1]
q = Quadratic( V, m, Ap )
model = MathProgBase.NonlinearModel(solver)
MathProgBase.loadproblem!(model, 3, 1, fill(-Inf,3), fill(Inf,3), fill(-Inf,1), bp, :Min, q)
MathProgBase.setwarmstart!( model, rand(3) )
MathProgBase.optimize!(model)
x = MathProgBase.getsolution(model)
@assert( all(Ap * x .<= bp .+ 1e-6) )



