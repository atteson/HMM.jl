using Commas
using HMMs
using PyPlot

spx = Commas.readcomma( joinpath( homedir(), "myenvironment", "dev", "SPXHMM", "data", "SPX" ) )
y = log.(1 .+ spx[:vwretd])

parameters = zeros(3)
HMMs.fit_mle!( HMMs.GenTDist, parameters, y, ones(length(y)), Dict{Symbol,Any}() )

delta = 0.0002
bins = collect(-0.01:delta:0.01)

clf()
hist( y, bins=bins )

midpoints = (bins[1:end-1] + bins[2:end])/2
densities = delta*length(y).*pdf.( HMMs.GenTDist( parameters... ), midpoints )
plot( midpoints, densities )


delta = 0.001
bins = collect(-0.25:delta:0.25)

clf()
hist( y, bins=bins, log=true )

midpoints = (bins[1:end-1] + bins[2:end])/2
densities = delta*length(y).*pdf.( HMMs.GenTDist( parameters... ), midpoints )
plot( midpoints, densities )

mu = mean(y)
sigma = std(y)
abs(mean(((y .- mu)./sigma).^3))^(1/3)

symy = [y; 2*mu .- y]
symparameters = zeros(3)
HMMs.fit_mle!( HMMs.GenTDist, symparameters, symy, ones(length(symy)), Dict{Symbol,Any}() )
parameters


t = HMMs.GenTDist( hmm.stateparameters[:,2]... )
using PyPlot
delta = 0.0005
bins = collect(-0.1:delta:0.1)
midpoints = (bins[1:end-1] + bins[2:end])/2
tdensities = pdf.( t, midpoints )
clf()
semilogy( midpoints, tdensities )
mu = hmm.stateparameters[1,2]
sigma = hmm.stateparameters[2,2]
nu = hmm.stateparameters[3,2]
densities = pdf.( Normal( mu, sigma*sqrt(nu/(nu-2)) ), midpoints )
plot( midpoints, densities )


