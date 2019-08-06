using HMMs
using Brobdingnag
using Dates

hmm = rand( HMMs.HMM{2,HMMs.GenTDist,Brob,Float64,Date} )
size( hmm.inequalityconstraintmatrix )

hmmtype = Models.FittableModel{Date, Float64, HMMs.HMM{statecount, HMMs.GenTDist, Brob, Float64, Date}, typeof(HMMs.em)}

using Models
using BusinessDays
using Dependencies
using Commas
using Distributed

startdate = Date(2004,7,2)
enddate = Date(2005,7,1)
statecount = 2
processes = 20

hmmtype = Models.FittableModel{Date, Float64, HMMs.HMM{statecount, HMMs.GenTDist, Brob, Float64, Date}, typeof(HMMs.em)}
criterion( model ) = HMMs.likelihood(model.model)[end]
modeldates = advancebdays( :USNYSE, startdate, -1 ):Year(1):enddate

    multistartmodeltype = Models.MultiStartModel{Date, Float64, hmmtype, typeof(criterion)}
    fitfunction = FunctionNode(Models.fit)
    singledatemodeltype = Models.FittableModel{Date, Float64, multistartmodeltype, typeof(fitfunction)}
    anmodeltype = Models.ANModel{Date, Float64, Models.RewindableModel{Date, Float64, singledatemodeltype}}
    modeltype = Models.AdaptedModel{Date, Float64, Models.LogReturnModel{Date, anmodeltype}}

    # now update it
    spx = Commas.readcomma( joinpath( homedir(), "data", "SPX" ) )

    index = 2
    dates = Date[]
    prices = Float64[]
    while true
        if spx.time[index-1] <= Time(16) && ( spx.time[index] > Time(16) || spx.date[index] > spx.date[index-1] )
            push!( dates, spx.date[index-1] )
            push!( prices, spx.price[index-1] )
        end
        if spx.date[index] >= startdate
            break
        end
global        index += 1
    end

    if processes > 1
        workers = addprocs(processes)
    end

    model = rand( modeltype, modelperiods=modeldates, seeds=1:100, t=dates[1:1], u=prices[1:1] )
#    model = rand( modeltype, modelperiods=modeldates, seeds=1:2, t=dates[1:1], u=prices[1:1] )
Models.update( model, dates[2:end], prices[2:end], modules=[:HMMs, :Brobdingnag, :Models] )
model.models[1].model.rootmodel.model.model.models[2].model.inequalityconstraintmatrix
msm = model.models[1].model.rootmodel.model.model
msm.models[msm.optimumindex].model.inequalityconstraintmatrix

    if processes > 1
        rmprocs(workers)
    end
