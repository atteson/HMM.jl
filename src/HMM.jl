module HMM

using Distributions

type Digraph
    from::Array{Int}
    to::Array{Int}
end

function addedge( g::Digraph, from::Int, to::Int )
    push!( g.from, from )
    push!( g.to, to )
end

fullyconnected( n::Int ) = Digraph( vcat( [collect(1:n) for i in 1:n]... ), vcat( [fill(i,n) for i in 1:n]... ) )

type GaussianHMM
    graph::Digraph
    initialprobabilities::Vector{Float64}
    transitionprobabilities::Matrix{Float64}
    means::Vector{Float64}
    stds::Vector{Float64}
    scratch::Dict{Symbol,Any}

    GaussianHMM( g::Digraph, pi::Vector{Float64}, a::Matrix{Float64}, mu::Vector{Float64}, sigma::Vector{Float64} ) =
        new( g, pi, a, mu, sigma, Dict{Symbol,Any}() )
end

function randomhmm( g::Digraph )
    numstates = max( maximum( g.from ), maximum( g.to ) )
    initialprobabilities = rand( numstates )
    initialprobabilities ./= sum( initialprobabilities )
    transitionprobabilities = rand( numstates, numstates )
    transitionprobabilities ./= sum( transitionprobabilities, 2 )
    means = randn( numstates )
    stds = randn( numstates ).^2
    return GaussianHMM( g, initialprobabilities, transitionprobabilities, means, stds )
end

function Base.rand( hmm::GaussianHMM, n::Int )
    cdfs = [ cumsum( hmm.transitionprobabilities[i,:] ) for i in 1:length(hmm.initialprobabilities) ]
    
    state = searchsorted( cumsum( hmm.initialprobabilities ), rand() ).start
    observations = [rand( Normal(hmm.means[state], hmm.stds[state]) )]
    
    for t = 2:n
        state = searchsorted( cdfs[state], rand() ).start
        push!( observations, rand( Normal(hmm.means[state], hmm.stds[state]) ) )
    end
    return observations
end

function setobservations( hmm::GaussianHMM, y::Vector{Float64} )
    hmm.scratch = Dict{Symbol,Any}()
    hmm.scratch[:y] = y
end

function observations( hmm::GaussianHMM )
    if !haskey( hmm.scratch, :y )
        error( "Need to set observations in order to perform calculation" )
    end
    return hmm.scratch[:y]
end

function pdfvalues( hmm::GaussianHMM )
    if !haskey( hmm.scratch, :b )
        y = observations( hmm )
        hmm.scratch[:b] = [[pdf( Normal( hmm.means[i], hmm.stds[i] ), y[t] ) for i in 1:length(hmm.means)] for t in 1:length(y)]
    end
    return hmm.scratch[:b]
end

function forwardprobabilities( hmm::GaussianHMM )
    if !haskey( hmm.scratch, :alpha )
        y = observations( hmm )
        N = length(hmm.initialprobabilities)
        b = pdfvalues( hmm )
        probabilities = [hmm.initialprobabilities .* b[1]]
        for i = 2:length(y)
            joint = hmm.transitionprobabilities' * probabilities[end] .* b[i]
            push!( probabilities, joint )
        end
        hmm.scratch[:alpha] = probabilities
    end
    return hmm.scratch[:alpha]
end

function backwardprobabilities( hmm::GaussianHMM )
    if !haskey( hmm.scratch, :beta )
        y = observations( hmm )
        N = length(hmm.initialprobabilities)
        probabilities = [ones(length(hmm.initialprobabilities))]
        b = pdfvalues( hmm )
        for i = length(y):-1:2
            joint = hmm.transitionprobabilities * (probabilities[end] .* b[i])
            push!( probabilities, joint )
        end
        hmm.scratch[:beta] = reverse(probabilities)
    end
    return hmm.scratch[:beta]
end

function conditionalstateprobabilities( hmm::GaussianHMM )
    if !haskey( hmm.scratch, :gamma )
        y = observations( hmm )
        T = length(y)
        alpha = forwardprobabilities( hmm )
        beta = backwardprobabilities( hmm )
        proby = sum(alpha[end])
        hmm.scratch[:gamma] = [alpha[i] .* beta[i]/proby for i in 1:T-1]
    end
end

function conditionaljointstateprobabilities( hmm::GaussianHMM )
    if !haskey( hmm.scratch, :xi )
        y = observations( hmm )
        T = length(y)
        alpha = forwardprobabilities( hmm )
        beta = backwardprobabilities( hmm )
        proby = sum(alpha[end])
        b = pdfvalues( hmm )
        hmm.scratch[:xi] = [hmm.transitionprobabilities.*(alpha[i]*(beta[i+1].*b[i+1])')/proby for i in 1:T-1]
    end
    return hmm.scratch[:xi]
end
    
end # module
