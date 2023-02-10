module BARS

export makeDATADict, makeBARSDict, bars, plotTREE

using MLJ
using MLJDecisionTreeInterface
using CSV
using DataFrames
using Plots
using LaTeXStrings
using DataStructures
using Statistics
using Random
using Measurements

function makeDataDict(dataFrame, testTrainSplit)

    #Load feature and target
    targets, features = dataFrame[:,2], dataFrame[:,1]
    features = hcat(features)

    ##Calculate the median of the target
    medianValueFeat = median(dataFrame[:,1])
    medianValueTar = median(dataFrame[:,2])

    ##Split the data into testing and training
    train, test = partition(dataFrame, testTrainSplit,rng=Random.GLOBAL_RNG,shuffle=true)

    ##Group the columns together
    trainData = [ train[:,1] train[:,2] ]
    testData = [ test[:,1] test[:,2] ]

    ##Extract the training Features and Targets
    trainFeatures = trainData[:,1]
    trainTargets = trainData[:,2]

    ##Extract the testing Features and Targets
    testFeatures = testData[:,1]
    testTargets = testData[:,2]

    dataDict = Dict(:targets => targets,
                    :features => features,
                    :featurename => propertynames(dataFrame)[featureCol],
                    :targetname => propertynames(dataFrame)[targetCol],
                    :mediantarget => medianValueTar,
                    :medianfeature => medianValueFeat,
                    :traindata => trainData,
                    :testdata => testData,
                    :trainfeatures => trainFeatures,
                    :traintargets => trainTargets,
                    :testfeatures => testFeatures,
                    :testtargets => testTargets)

    return dataDict

end 

function makeDataDict(dataFrame)
    
    #Load feature and target
    targets, features = dataFrame[:,2], dataFrame[:,1]
    features = hcat(features)

    ##Calculate the median of the feature and the target
    medianValueFeat = median(dataFrame[:,1])
    medianValueTar = median(dataFrame[:,2])

    dataDict = Dict(:targets => targets,
                    :features => features,
                    :featurename => propertynames(dataFrame)[1],
                    :targetname => propertynames(dataFrame)[2],
                    :mediantarget => medianValueTar,
                    :medianfeature => medianValueFeat)

    return dataDict

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

function makeBARSDict(dataDict)

    if haskey(dataDict,:trainfeatures)

        trainFeatures = table(reshape(dataDict[:trainfeatures], length(dataDict[:trainfeatures]),1))
        features = table(reshape(dataDict[:features], length(dataDict[:features]),1))
        trainTargets = dataDict[:traintargets]
        targets = dataDict[:targets]

        #Make the tree container
        Tree = @MLJ.load DecisionTreeRegressor pkg=DecisionTree verbosity=0;
        tree = Tree()

        #tuning the min_purity_increase hyperparameter
        r1 = range(tree, :merge_purity_threshold, lower=0.001, upper=1.0, scale=:log);
        self_tuning_tree = TunedModel(model=tree,
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
        naivePredictions = []
        for i∈1:length(dataDict[:targets])
            push!(naivePredictions, dataDict[:mediantarget])
        end

        naiveMAE = calcMAE(dataDict[:targets],naivePredictions)

        smartMAE = calcMAE(dataDict[:targets],smartPredictions)

        #model performance
        SS_Res = sum((targets-smartPredictions).^2)
        SS_Tot = sum((targets.-mean(targets)).^2)
        R² = 1-SS_Res/SS_Tot
        #R² = report(mach).best_history_entry[3][1]

        BARSDict = Dict(:R² => R²,
                        :smartMAE => smartMAE,
                        :naiveMAE => naiveMAE,
                        :naivepredictions => naivePredictions,
                        :smartpredictions => smartPredictions,
                        :treesplits => splits,
                        :data => dataDict)

    else

        features = table(reshape(dataDict[:features], length(dataDict[:features]),1))
        targets = dataDict[:targets]

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
        mach = machine(self_tuning_tree, features, targets);

        #fit the model
        MLJ.fit!(mach, verbosity=2)

        #print_tree(tree)

        ## apply model to make predictions
        smartPredictions = MLJ.predict(mach,features)

        ##find the splits for the tree
        splits = []
        searchFields(splits,:featval,fitted_params(mach)[2].tree.node)

        #Calculate BARS
        naivePredictions = []
        for i∈1:length(dataDict[:targets])
            push!(naivePredictions, dataDict[:mediantarget])
        end

        naiveMAE = calcMAE(dataDict[:targets],naivePredictions)

        smartMAE = calcMAE(dataDict[:targets],smartPredictions)

        #model performance
        SS_Res = sum((targets-smartPredictions).^2)
        SS_Tot = sum((targets.-mean(targets)).^2)
        R² = 1-SS_Res/SS_Tot
        #R² = report(mach).best_history_entry[3][1]

        BARSDict = Dict(:R² => R²,
                   :smartMAE => smartMAE,
                   :naiveMAE => naiveMAE,
                   :naivepredictions => naivePredictions,
                   :smartpredictions => smartPredictions,
                   :treesplits => splits,
                   :data => dataDict)

    end
    
    return BARSDict

end

function calcSelfBARS(selfBARSDict)

    smartMAE = selfBARSDict[:smartMAE]
    naiveMAE = selfBARSDict[:naiveMAE]

    r₀ = naiveMAE/smartMAE

    selfBARSDict[:r₀] = r₀
    if smartMAE>naiveMAE
        BARS = NaN
    else
        BARS = 1.0
    end

    selfBARSDict[:selfBARS] = BARS

    return selfBARSDict
end

function calcBARS(BARSDict,selfBARSDict, acceptance)

    smartMAE = BARSDict[:smartMAE]
    naiveMAE = BARSDict[:naiveMAE]
    smartMAEself = selfBARSDict[:smartMAE]
    α=acceptance 
    R² = BARSDict[:R²]
    r = (smartMAE)/(naiveMAE)
    r₀= (smartMAEself)/(naiveMAE)
    if R² < 0 || R² > 1
        BARS = 0
    else
        BARS = exp.(-((r-r₀)/α)^2)
    end

    BARSDict[:BARS] = BARS
    BARSDict[:r] = r
    BARSDict[:r₀] = r₀
    BARSDict[:α] = α

    return BARSDict
end

function plotTREE(BARSDict)
    theBARSScore = string(round(BARSDict[:BARS],sigdigits=3))
    R² = string(round(BARSDict[:R²],sigdigits = 3))
    regressionPlot = scatter(BARSDict[:data][:features] ,
        BARSDict[:smartpredictions] ,
        markercolor = :purple ,
        markershape = :cross ,
        legend = :outerbottom,
        title = L"BARS= "*theBARSScore*", "*L"R^2= "*R²,
        label = "Smart Predictions")
    scatter!(BARSDict[:data][:features],
        BARSDict[:naivepredictions] ,
        markercolor = :black ,
        markershape = :xcross ,
        label = "Naive Predictions")
    if length(BARSDict[:treesplits])>0
        vline!(BARSDict[:treesplits],
        label = "Decision Tree Splits",
        linecolor = :purple)
    end

    xlabel!("Feature: "*string(BARSDict[:data][:featurename]))
    ylabel!("Target: "*string(BARSDict[:data][:targetname]))

    if haskey(BARSDict[:data],:trainfeatures)
        scatter!(BARSDict[:data][:testfeatures] ,
        BARSDict[:data][:testtargets] ,
            makercolor = :red ,
            markershape = :rect ,
            label = "Testing Data")
        scatter!(BARSDict[:data][:trainfeatures] ,
            BARSDict[:data][:traintargets] ,
            makercolor = :blue ,
            markershape = :circ ,
            label = "Training Data")
    end

    return regressionPlot
end

function plotTREE(BARSDict)
    theBARSScore = string(round(BARSDict[:BARS],sigdigits=3))
    R² = string(round(BARSDict[:R²],sigdigits = 3))
    regressionPlot = scatter(BARSDict[:data][:features] ,
    BARSDict[:data][:targets] ,
        makercolor = :red ,
        markershape = :rect ,
        label = "Testing Data",
        legend = :outerbottom,
        title = L"BARS= "*theBARSScore*", "*L"R^2= "*R²)
    scatter!(BARSDict[:data][:features] ,
        BARSDict[:smartpredictions] ,
        markercolor = :purple ,
        markershape = :cross ,
        label = "Smart Predictions")
    scatter!(BARSDict[:data][:features],
        BARSDict[:naivepredictions] ,
        markercolor = :black ,
        markershape = :xcross ,
        label = "Naive Predictions")
    if length(BARSDict[:treesplits])>0
        vline!(BARSDict[:treesplits],
        label = "Decision Tree Splits",
        linecolor = :purple)
    end
    xlabel!("Feature: "*string(BARSDict[:data][:featurename]))
    ylabel!("Target: "*string(BARSDict[:data][:targetname]))

    return regressionPlot
end

function bars(dataFrame)

    dataDict = makeDataDict(dataFrame)

    BARSDict = makeBARSDict(dataDict)

    return BARSDict

end

end