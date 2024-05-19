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

"""
    standardizeData(dataframe)

    Given a time series, subtract the mean and divide the result by the 
    standard deviation. This is very useful to do when using non-tree 
    based models in ML.

TBW
"""
function standardizeData(dataframe)
    stDataFrame = DataFrame()
    for name∈propertynames(dataframe)
        μ=mean(dataframe[!,name])
        σ=std(dataframe[!,name])
        z=(dataframe[!,name].-μ)./σ
        stDataFrame[!,name] = z
    end

    return stDataFrame
    
end

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

function bars(dataFrame,targetName,featureName)
    pairDftrial = setup(dataFrame,targetName,featureName)
    trueValues,smartPreds,naivePreds = calcPreds(pairDftrial,featureName,.7)
    calcedScore = calcBars(trueValues,smartPreds,naivePreds)
    return calcedScore
end

function barsFeatures(targetName,dataFrame)
    barsMatrixRow = []
    for featureName ∈ propertynames(dataFrame)
        score = bars(dataFrame,targetName,featureName)
        push!(barsMatrixRow, score)
    end
    return barsMatrixRow
end

function barsMatrix(dataFrame)
    varNames = propertynames(dataFrame)
    n = length(varNames)
    barsMatrix = zeros(n,n)
    for i∈1:n
        for j∈1:n
            scr = bars(dataFrame,varNames[i],varNames[j])
            barsMatrix[i,j] = scr
        end
    end

    return barsMatrix
end

function matrixPlot(name,matrix,dataFrame,cutoff,color,cScheme,fontSize)
    n = length(propertynames(dataFrame))
    newmatrix = zeros(size(matrix)[1],size(matrix)[2])
    
    for j∈1:size(matrix)[2]
        newmatrix[:,j]=reverse(barsmatrix[:,j])
    end
    
    hmap=Plots.plot(heatmap(
        xticks = (1:1:n, propertynames(dataFrame)),
        yticks = (n:-1:1, propertynames(dataFrame)),
        newmatrix, 
        title = name*" Heat Map",
        xlabel = "targets",
        ylabel = "features",
        xrotation = 40.0,
        c=cScheme
    ))

    vline!([.5], color=color, label=false)
    hline!([.5], color=color, label=false)
    for tick∈1:n
        vline!([tick+.5], color=color, label=false)
        hline!([tick+.5], color=color, label=false)
    end

    xpoints = [ i for i∈1:n]
    ypoints = [ j for j∈1:n]
    for ycoor∈ypoints
        for xcoor∈xpoints
            value = newmatrix[ycoor,xcoor]
            if value<cutoff
                value = 0
            end
            plot!((xcoor,ycoor),
                legend = false,
                series_annotation=text.(round((value),sigdigits=2),fontSize,color)
            )
        end
    end

    return hmap
end