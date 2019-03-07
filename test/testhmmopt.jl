using HMMs
using Brobdingnag
using Ipopt
using MathProgBase
using Random

struct HMMOptimizer{Dist,Calc,Out} <: MathProgBase.AbstractNLPEvaluator
    hmm::HMMs.HMM{Dist,Calc,Out}
    current::Vector{Float64}
end

function HMMOptimizer( hmm::HMMs.HMM{Dist,Calc,Out} ) where {Dist,Calc,Out}
    m = length(hmm.initialprobabilities)
    return HMMOptimizer( hmm, fill( NaN, m*(m+2) ) )
end

function MathProgBase.initialize( ::HMMOptimizer{Dist,Calc,Out}, requested_features::Vector{Symbol} ) where {Dist,Calc,Out}
    unimplemented = setdiff( requested_features, [:Grad,:Jac] )
    if !isempty( unimplemented )
        error( "The following features aren't implemented: " * join( string.(unimplemented), ", " ) )
    end
end

MathProgBase.features_available( ::HMMOptimizer{Dist,Calc,Out} ) where {Dist,Calc,Out} = [:Grad]

function MathProgBase.eval_f( opt::HMMOptimizer{Dist,Brob,Float64}, x ) where {Dist}
    HMMs.setparameters!( opt.hmm, x )
    return HMMs.likelihood( opt.hmm )[end].log
end
    
function MathProgBase.eval_g( opt::HMMOptimizer{Dist,Calc,Out}, g, x ) where {Dist,Calc,Out}
    HMMs.setparameters!( opt.hmm, x )
    g[:] = sum(opt.hmm.transitionprobabilities, dims=2)
end

function MathProgBase.eval_grad_f( opt::HMMOptimizer{Dist,Brob,Float64}, grad_f, x ) where {Dist}
    HMMs.setparameters!( opt.hmm, x )
    l = HMMs.likelihood( opt.hmm )[end]
    dl = HMMs.dlikelihood( opt.hmm )[:,end]
    grad_f[:] = dl ./ l
end

function MathProgBase.jac_structure( opt::HMMOptimizer{Dist,Calc,Out} ) where {Dist,Calc,Out}
    m = length(opt.hmm.initialprobabilities)
    rows = zeros(Int, m^2)
    cols = zeros(Int, m^2)
    inner = 1
    for i = 1:m
        for j = 1:m
            rows[inner] = i
            cols[inner] = (i-1)*m + j
            inner += 1
        end
    end
    return (rows,cols)
end

function MathProgBase.eval_jac_g( opt::HMMOptimizer{Dist,Calc,Out}, J, x ) where {Dist,Calc,Out}
    m = length(opt.hmm.initialprobabilities)
    J[:] = ones(m^2)
end

m = 2
graph = HMMs.fullyconnected( m )
hmm = HMMs.randomhmm( graph, calc=Brob, seed=1 )
y = rand( hmm, 10_000 );

mlhmm = HMMs.randomhmm( graph, calc=Brob, seed=2 )
HMMs.setobservations( mlhmm, y );

solver = IpoptSolver(
    max_iter=1_000,
    derivative_test="first-order",
    option_file_name="/home/atteson/ipopt.opt",
    print_level=5,
)
model = MathProgBase.NonlinearModel(solver)
opt = HMMOptimizer(mlhmm)

MathProgBase.jac_structure( opt )
MathProgBase.loadproblem!(model, m*(m+2), m, [zeros(m^2); fill(-Inf,m); zeros(m)], fill(Inf,m*(m+2)),
                          ones(m), ones(m), :Max, opt)
parameters = fill( NaN, m*(m+2) )
HMMs.getparameters!( opt.hmm, parameters )

grad_f = fill(NaN,m*(m+2))
MathProgBase.eval_grad_f( opt, grad_f, parameters )

errors = Float64[]
delta = 1e-8
for i = 1:length(parameters)
    parameters[i] += delta
    fp = MathProgBase.eval_f( opt, parameters )
    parameters[i] -= delta

    parameters[i] -= delta
    fm = MathProgBase.eval_f( opt, parameters )
    parameters[i] += delta

    d = (fp - fm)/(2*delta)
    push!( errors, abs(d / grad_f[i] - 1) )
    @assert( errors[end] < 1e-4, "mismatch at derivative $i" )
end

(rows,cols) = MathProgBase.jac_structure( opt )
    
MathProgBase.setwarmstart!( model, parameters )
MathProgBase.optimize!(model)

HMMs.likelihood( opt.hmm )

emhmm = HMMs.randomhmm( graph, calc=Brob, seed=2 )
HMMs.setobservations( emhmm, y );
HMMs.em( emhmm, debug=2 )
HMMs.likelihood( opt.hmm )
HMMs.likelihood( emhmm )

HMMs.em( opt.hmm, debug=2 )
HMMs.likelihood( opt.hmm )

HMMs.getparameters!( opt.hmm, parameters )
MathProgBase.eval_f( opt, parameters )
