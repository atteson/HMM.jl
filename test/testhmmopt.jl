using HMM
using Brobdingnag
using Ipopt
using MathProgBase
using Random

struct HMMOptimizer{Calc,Out} <: MathProgBase.AbstractNLPEvaluator
    hmm::HMM.GaussianHMM{Calc,Out}
    current::Vector{Float64}
end

function HMMOptimizer( hmm::HMM.GaussianHMM{Calc,Out} ) where {Calc,Out}
    m = length(hmm.initialprobabilities)
    return HMMOptimizer( hmm, fill( NaN, m*(m+2) ) )
end

function MathProgBase.initialize( ::HMMOptimizer{Calc,Out}, requested_features::Vector{Symbol} ) where {Calc,Out}
    unimplemented = setdiff( requested_features, [:Grad,:Jac] )
    if !isempty( unimplemented )
        error( "The following features aren't implemented: " * join( string.(unimplemented), ", " ) )
    end
end

MathProgBase.features_available( ::HMMOptimizer{Calc,Out} ) where {Calc,Out} = [:Grad]

function setparameters!( opt::HMMOptimizer{Calc,Out}, x ) where {Calc,Out}
    if any(opt.current .!= x)
        m = length(opt.hmm.initialprobabilities)
        index = 1
        for i = 1:m
            for j = 1:m
                opt.hmm.transitionprobabilities[i,j] = x[index]
                index += 1
            end
        end
        opt.hmm.means[:] = x[m^2+1:m*(m+1)]
        opt.hmm.stds[:] = x[m*(m+1)+1:m*(m+2)]
        HMM.clear( opt.hmm )
    end
end

function getparameters!( opt::HMMOptimizer{Calc,Out}, x ) where {Calc,Out}
    m = length(opt.hmm.initialprobabilities)
    index = 1
    for i = 1:m
        for j = 1:m
            x[index] = opt.hmm.transitionprobabilities[i,j]
            index += 1
        end
    end
    x[m^2+1:m*(m+1)] = opt.hmm.means
    x[m*(m+1)+1:m*(m+2)] = opt.hmm.stds
end

function MathProgBase.eval_f( opt::HMMOptimizer{Brob,Float64}, x )
    setparameters!( opt, x )
    return HMM.likelihood( opt.hmm ).log
end
    
function MathProgBase.eval_g( opt::HMMOptimizer{Calc,Out}, g, x ) where {Calc,Out}
    setparameters!( opt, x )
    g[:] = sum(opt.hmm.transitionprobabilities, dims=2)
end

function MathProgBase.eval_grad_f( opt::HMMOptimizer{Brob,Float64}, grad_f, x )
    setparameters!( opt, x )
    l = HMM.likelihood( opt.hmm )
    dl = HMM.dlikelihood( opt.hmm )
    grad_f[:] = dl ./ l
end

function MathProgBase.jac_structure( opt::HMMOptimizer{Calc,Out} ) where {Calc,Out}
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

function MathProgBase.eval_jac_g( opt::HMMOptimizer{Calc,Out}, J, x ) where {Calc,Out}
    m = length(opt.hmm.initialprobabilities)
    J[:] = ones(m^2)
end

m = 2
graph = HMM.fullyconnected( m )
hmm = HMM.randomhmm( graph, calc=Brob, seed=1 )
y = rand( hmm, 10_000 );

mlhmm = HMM.randomhmm( graph, calc=Brob, seed=2 )
HMM.setobservations( mlhmm, y );

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
getparameters!( opt, parameters )

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

    fd = (Brob(true,fp) - Brob(true,fm))/(2*delta)
    push!( errors, abs(fd.log / grad_f[i] - 1) )
    @assert( errors[end] < 1e-4, "mismatch at derivative $i" )
end

(rows,cols) = MathProgBase.jac_structure( opt )
    
MathProgBase.setwarmstart!( model, parameters )
MathProgBase.optimize!(model)

HMM.likelihood( opt.hmm )

emhmm = HMM.randomhmm( graph, calc=Brob, seed=2 )
HMM.setobservations( emhmm, y );
HMM.em( emhmm, debug=2 )
HMM.likelihood( opt.hmm )
HMM.likelihood( emhmm )

HMM.em( opt.hmm, debug=2 )
HMM.likelihood( opt.hmm )

getparameters!( opt, parameters )
MathProgBase.eval_f( opt, parameters )
