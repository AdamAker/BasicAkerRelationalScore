module BARS

export splitData, calcMAE, setupBARS, calcSelfBARS, calcBARS, plotTREE

using MLJ
using CSV
using DataFrames
using Plots
using LaTeXStrings
using DataStructures
using Statistics
using Random
using Measurements

function splitData(dataFrame, testTrainSplit, targetCol, featureCol)

    #Load feature and target
    targets, features = dataFrame[:,targetCol], dataFrame[:,featureCol]
    features = hcat(features)

    ##Calculate the median of the feature and the target
    medianValueFeat = median(dataFrame[:,featureCol])
    medianValueTar = median(dataFrame[:,targetCol])

    ##Split the data into testing and training
    train, test = partition(dataFrame, testTrainSplit,rng=Random.GLOBAL_RNG,shuffle=true)

    ##Group the columns together
    trainData = [ train[:,featureCol] train[:,targetCol] ]
    testData = [ test[:,featureCol] test[:,targetCol] ]

    ##Extract the training Features and Targets
    trainFeatures = trainData[:,1]
    trainTargets = trainData[:,2]

    ##Extract the testing Features and Targets
    testFeatures = testData[:,1]
    testTargets = testData[:,2]

    DataDict = Dict("targets" => targets,
                    "features" => features,
                    "feature name" => propertynames(dataFrame)[featureCol],
                    "target name" => propertynames(dataFrame)[targetCol],
                    "median value of target" => medianValueTar,
                    "median value of feature" => medianValueFeat,
                    "train data" => trainData,
                    "test data" => testData,
                    "train features" => trainFeatures,
                    "train targets" => trainTargets,
                    "test features" => testFeatures,
                    "test targets" => testTargets)

    return DataDict
end

function searchFields(array,fieldname,object)
    for field in fieldnames(typeof(object))
        if isequal(field,fieldname)
            push!(array, getfield(object, field))
        else
            try
                searchFields(array,fieldname, getfield(object, field))
            catch e
                println("Didn't work")
            end
        end
    end
end

function calcMAE(trueValues,predValues)
        N=length(predValues)
        M=length(trueValues)
        MAE = 0
        if N == M
            MAE = sum(abs.(trueValues.-predValues))/N
            return MAE
        else
            print("lengths of trueValues and predValues are different")
    end
end

function setupBARS(DataDict)

    #Machine Learning
    ## train regression tree

    trainFeatures = table(reshape(DataDict["train features"], length(DataDict["train features"]),1))
    features = table(reshape(DataDict["features"], length(DataDict["features"]),1))
    trainTargets = DataDict["train targets"]
    targets = DataDict["targets"]

    #Make the tree container
    Tree = @MLJ.load DecisionTreeRegressor pkg=DecisionTree verbosity=0;
    tree = Tree()

    #tuning the min_purity_increase hyperparameter
    r1 = range(tree, :merge_purity_threshold, lower=0.001, upper=1.0, scale=:log);
    self_tuning_tree = TunedModel(model=tree,
							  resampling=CV(nfolds=6,rng=Random.GLOBAL_RNG,shuffle=true),
							  tuning=Grid(resolution=10),
							  range=r1,
							  measure=mae);

    #store the self tuning tree in a machine
    mach = machine(self_tuning_tree, trainFeatures, trainTargets);

    #fit the model
    MLJ.fit!(mach, verbosity=2)

    #print_tree(tree)

    ## apply model to make predictions
    smartPredictions = MLJ.predict(mach,features)

    ##find the splits for the tree
    splits = []
    searchFields(splits,:featval,fitted_params(mach)[2].tree.node)

    #Calculate BARS
    dumbPredictions = []
    for i∈1:length(DataDict["targets"])
        push!(dumbPredictions, DataDict["median value of target"])
    end

    dumbMAE = calcMAE(DataDict["targets"],dumbPredictions)

    smartMAE = calcMAE(DataDict["targets"],smartPredictions)

    #model performance
    SS_Res = sum((targets-smartPredictions).^2)
    SS_Tot = sum((targets.-mean(targets)).^2)
    R2 = 1-SS_Res/SS_Tot
    #R2 = report(mach).best_history_entry[3][1]

    BARSDict = Dict("R squared" => R2,
                   "smart MAE" => smartMAE,
                   "dumb MAE" => dumbMAE,
                   "dumb predictions" => dumbPredictions,
                   "smart predictions" => smartPredictions,
                   "decision tree splits" => splits,
                   "Data" => DataDict)
    
    return BARSDict

end

function calcSelfBARS(selfBARSDict)

    smartMAE = selfBARSDict["smart MAE"]
    dumbMAE = selfBARSDict["dumb MAE"]

    α = dumbMAE/smartMAE
    x = 0
    selfBARSDict["α"] = α
    if smartMAE>dumbMAE
        BARS = NaN
    else
        BARS = 1.0
    end

    selfBARSDict["selfBARS"] = BARS
    selfBARSDict["x"] = x

    return selfBARSDict
end

function calcBARS(BARSDict,selfBARSDict, acceptance)

    smartMAE = BARSDict["smart MAE"]
    dumbMAE = BARSDict["dumb MAE"]
    smartMAEself = selfBARSDict["smart MAE"]
    dumbMAEself = selfBARSDict["dumb MAE"]
    β=acceptance
    rSquared = BARSDict["R squared"]
    x = (smartMAE-smartMAEself)/(dumbMAE)
    if rSquared < 0 || rSquared > 1
        BARS = NaN
    else
        BARS = exp.(-(x/β)^2)
    end

    BARSDict["BARS"] = BARS
    BARSDict["x"] = x
    BARSDict["β"] = β

    return BARSDict
end

function plotTREE(BARSDict)
    theBARSScore = string(round(BARSDict["BARS"],sigdigits=3))
    rSquared = string(round(BARSDict["R squared"],sigdigits = 3))
    regressionPlot = scatter(BARSDict["Data"]["test features"] ,
    BARSDict["Data"]["test targets"] ,
        makercolor = :red ,
        markershape = :rect ,
        label = "Testing Data",
        legend = :outerbottom,
        title = L"BARS= "*theBARSScore*", "*L"R^2_{avg}= "*rSquared)
    scatter!(BARSDict["Data"]["train features"] ,
        BARSDict["Data"]["train targets"] ,
        makercolor = :blue ,
        markershape = :circ ,
        label = "Training Data")
    scatter!(BARSDict["Data"]["features"] ,
        BARSDict["smart predictions"] ,
        markercolor = :purple ,
        markershape = :cross ,
        label = "Smart Predictions")
    scatter!(BARSDict["Data"]["features"],
        BARSDict["dumb predictions"] ,
        markercolor = :black ,
        markershape = :xcross ,
        label = "Dumb Predictions")
    if length(BARSDict["decision tree splits"])>0
        vline!(BARSDict["decision tree splits"],
        label = "Decision Tree Splits",
        linecolor = :purple)
    end
    xlabel!("Feature: "*string(BARSDict["Data"]["feature name"]))
    ylabel!("Target: "*string(BARSDict["Data"]["target name"]))

    return regressionPlot
end

end
