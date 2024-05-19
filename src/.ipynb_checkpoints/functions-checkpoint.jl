function setup(dataFrame,targetName::Symbol,featureName::Symbol)

    stdf = standardizeData(dataFrame)
    if targetName == featureName
        pairDf = select(stdf,targetName)
    else
        pairDf = select(stdf,[targetName,featureName])
    end
    return pairDf
end

function calcPreds(pairDf,targetName,testTrainSplit)

    y,X=unpack(pairDf,==(targetName),rng=123);
    if isequal(size(X),(0,0))
        X=y
        X=table(reshape(X,length(X),1))
    else
        X=table(reshape(X,length(X),1))
    end
    tree = @MLJ.load DecisionTreeRegressor pkg = DecisionTree
    tree = MLJDecisionTreeInterface.DecisionTreeRegressor()
    evaluation = evaluate(tree, X, y,
            #resampling=CV(nfolds=6),
                        measures=[l1,l2],
                        verbosity=0)
    #store the self tuning tree in a machine
    mach = machine(tree,X, y);

    train,test = partition(eachindex(y), testTrainSplit)
    #fit the model
    MLJ.fit!(mach,rows=train)

    ## apply model to make predictions
    smartPreds = MLJ.predict(mach,X)

    #Calculate the median of the target
    medianValueTar = median(y)
    naivePreds = []
    for i∈1:length(y)
        push!(naivePreds, medianValueTar)
    end

    return y,smartPreds,naivePreds

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
    
function calcBars(trueValues,smartPreds,naivePreds)

    naiveMAE = calcMAE(trueValues,naivePreds)

    smartMAE = calcMAE(trueValues,smartPreds)

    r = (smartMAE)/(naiveMAE)
    
    bars = 10*exp.(-(r).^2)
    
    return round(bars,digits=2)

end

function sortNameValue(names,values)
    for i∈1:length(values)
        for j∈i:length(values)
            if values[j]>values[i]
                swapVal = values[j]
                values[j] = values[i]
                values[i]= swapVal

                swapNam = names[j]
                names[j] = names[i]
                names[i] = swapNam
            end
        end
    end
    return names,values
end

function barsFeatures(targetName,dataFrame)
    barsMatrixRow = []
    for featureName ∈ propertynames(dataFrame)
        pairDf = setup(dataFrame,targetName,featureName)
        trueValues,smartPreds,naivePreds = calcPreds(pairDf,featureName,.7)
        bars = calcBars(trueValues,smartPreds,naivePreds)
        push!(barsMatrixRow, bars)
    end
    return barsMatrixRow 
end