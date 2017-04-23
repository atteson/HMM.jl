module HMM

using Distributions

type Digraph
    from::Array{Int}
    to::Array{Int}
end

Base.copy( g::Digraph ) = Digraph( copy( g.from ), copy( g.to ) )

function addedge( g::Digraph, from::Int, to::Int )
    push!( g.from, from )
    push!( g.to, to )
end

fullyconnected( n::Int ) = Digraph( vcat( [collect(1:n) for i in 1:n]... ), vcat( [fill(i,n) for i in 1:n]... ) )

type GaussianHMM{T}
    graph::Digraph
    initialprobabilities::Vector{T}
    transitionprobabilities::Matrix{T}
    means::Vector{T}
    stds::Vector{T}
    scratch::Dict{Symbol,Any}
end

GaussianHMM{T <: Real}( g::Digraph, pi::Vector{T}, a::Matrix{T}, mu::Vector{T}, sigma::Vector{T} ) =
    GaussianHMM{T}( g, pi, a, mu, sigma, Dict{Symbol,Any}() )

Base.copy( hmm::GaussianHMM ) =
    GaussianHMM( copy( hmm.graph ), copy( hmm.initialprobabilities ), copy( hmm.transitionprobabilities ), copy( hmm.means ), copy( hmm.stds ), copy( hmm.scratch ) )

function randomhmm( g::Digraph; float::DataType = Float64 )
    numstates = max( maximum( g.from ), maximum( g.to ) )
    initialprobabilities = Vector{float}(rand( numstates ))
    initialprobabilities ./= sum( initialprobabilities )
    transitionprobabilities = Matrix{float}(rand( numstates, numstates ))
    transitionprobabilities ./= sum( transitionprobabilities, 2 )
    means = Vector{float}(randn( numstates ))
    stds = Vector{float}(randn( numstates ).^2)
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

function setobservations{T}( hmm::GaussianHMM{T}, y::Vector{T} )
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
        probabilities = [ones(BigFloat,length(hmm.initialprobabilities))]
        b = pdfvalues( hmm )
        for i = length(y):-1:2
            joint = hmm.transitionprobabilities * (probabilities[end] .* b[i])
            push!( probabilities, joint )
        end
        hmm.scratch[:beta] = reverse(probabilities)
    end
    return hmm.scratch[:beta]
end

function likelihood( hmm::GaussianHMM )
    if !haskey( hmm.scratch, :likelihood )
        alpha = forwardprobabilities( hmm )        
        hmm.scratch[:likelihood] = sum(alpha[end])
    end
    return hmm.scratch[:likelihood]
end

function conditionalstateprobabilities( hmm::GaussianHMM )
    if !haskey( hmm.scratch, :gamma )
        y = observations( hmm )
        T = length(y)
        alpha = forwardprobabilities( hmm )
        beta = backwardprobabilities( hmm )
        proby = likelihood( hmm )
        hmm.scratch[:gamma] = [alpha[i][j] * beta[i][j]/proby for i in 1:T, j in 1:length(alpha[1])]
    end
    return hmm.scratch[:gamma]
end

function conditionaljointstateprobabilities( hmm::GaussianHMM )
    if !haskey( hmm.scratch, :xi )
        y = observations( hmm )
        T = length(y)
        alpha = forwardprobabilities( hmm )
        beta = backwardprobabilities( hmm )
        proby = likelihood( hmm )
        b = pdfvalues( hmm )
        hmm.scratch[:xi] = [hmm.transitionprobabilities.*(alpha[i]*(beta[i+1].*b[i+1])')/proby for i in 1:T-1]
    end
    return hmm.scratch[:xi]
end

function emstep( hmm::GaussianHMM, nexthmm::GaussianHMM )
    y = observations( hmm )
    T = length(y)
    gamma = conditionalstateprobabilities( hmm )
    occupation = sum(gamma[1:end-1,:],1)
    xi = conditionaljointstateprobabilities( hmm )
    
    nexthmm.initialprobabilities = gamma[1,:]
    nexthmm.transitionprobabilities = sum(xi)./occupation'
    nexthmm.means = sum([gamma[i,:]*y[i] for i in 1:T])./vec(occupation)
    nexthmm.stds = sqrt(sum([gamma[i,:].*(y[i] - hmm.means).^2 for i in 1:T])./vec(occupation))
    
    delete!( nexthmm.scratch, :alpha )
    delete!( nexthmm.scratch, :beta )
    delete!( nexthmm.scratch, :gamma )
    delete!( nexthmm.scratch, :xi )
    delete!( nexthmm.scratch, :b )
    delete!( nexthmm.scratch, :likelihood )
end

function em{T}( hmm::GaussianHMM{T}; epsilon::Float64 = 0.0, debug::Int = 0, maxiterations::Float64 = Inf )
    t0 = Base.time()
    nexthmm = randomhmm( hmm.graph, float=T )
    setobservations( nexthmm, observations( hmm ) )
    hmms = [hmm, nexthmm]
    oldlikelihood = BigFloat(0.0)
    newlikelihood = likelihood( hmm )
    done = false
    i = 1

    iterations = 1
    while !done
        if debug >= 2
            println( "Likelihood = $newlikelihood" )
        end
        emstep( hmms[i], hmms[3-i] )
        oldlikelihood = newlikelihood
        done = any(isnan(hmms[3-i].initialprobabilities)) || any(isnan(hmms[3-i].transitionprobabilities)) ||
            any(isnan(hmms[3-i].means)) || any(isnan(hmms[3-i].stds)) || any(hmms[3-i].stds.<=0) || iterations >= maxiterations
        if !done
            newlikelihood = likelihood( hmms[3-i] )
            done = newlikelihood / oldlikelihood - 1 <= epsilon
        end
        i = 3-i
        iterations += 1
    end
    if debug >= 1
        println( "Final likelihood = $oldlikelihood; iterations = $iterations" )
    end

    hmm.initialprobabilities = hmms[3-i].initialprobabilities
    hmm.transitionprobabilities = hmms[3-i].transitionprobabilities
    hmm.means = hmms[3-i].means
    hmm.stds = hmms[3-i].stds
    hmm.scratch = hmms[3-i].scratch
    
    hmm.scratch[:iterations] = iterations
    hmm.scratch[:time] = Base.time() - t0
end

time( hmm::GaussianHMM ) = hmm.scratch[:time]

iterations( hmm::GaussianHMM ) = hmm.scratch[:iterations]

end # module
