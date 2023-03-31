using BasicAkerRelationalScore
using OrderedCollections
using LaTeXStrings

function splitDataFrame(bigDataFrame,featureNames)
    
    featuresDataFrame = bigDataFrame[:,featureNames]
    targetsDataFrame = bigDataFrame[:,Not(featureNames)]

    return featuresDataFrame,targetsDataFrame

end

function makeFeaturesDict(featuresDataFrame,targetsDataFrame,acceptance)

    featuresDict = OrderedDict()
    featureList = []
    powerBARSs = []

    for featureName∈propertynames(featuresDataFrame)

        df = DataFrame()
        df[:,featureName]=featuresDataFrame[:,featureName]
        df = hcat(df,targetsDataFrame)
        
        targetsDict = BasicAkerRelationalScore.makeTargetsDict(df,featureName,acceptance)
        targetsDict = BasicAkerRelationalScore.calcPBARS(targetsDict)

        featuresDict[featureName] = targetsDict
        push!(powerBARSs,targetsDict[:pBARS])
        push!(featureList, featureName)

    end

    featuresDict[:powerBARSs] = powerBARSs
    featuresDict[:featureList] = propertynames(featuresDataFrame)

    return featuresDict

end

function plotPowerBARSs(featuresDict)

    finalIndex = length(featuresDict[:powerBARSs])
    sortedPowerBARSs = [0.0 for i∈1:finalIndex]
    sortedFeatureNames = ["0" for i∈1:finalIndex]
    sortedIndicies = sortperm(featuresDict[:powerBARSs])

    for sortingIndex ∈ sortedIndicies
        sortedPowerBARSs[finalIndex] = featuresDict[:powerBARSs][sortingIndex]
        sortedFeatureNames[finalIndex] = string(featuresDict[:featureList][sortingIndex])
        finalIndex-=1
    end

    PowerBARSsPlot = Plots.bar(sortedFeatureNames,
                        sortedPowerBARSs,
                        title = "PBARSs of the Features",
                        label = L"PBARS=\sum_{i=1}^{N}R_{i}^2BARS_i",
                        xticks = :all,
                        xrotation = 60.0)
    xlabel!("Feature Name")
    ylabel!("PowerBARS")

    return PowerBARSsPlot

end

function generateModelvars(featuresDict)

    D = Dict(zip(featuresDict[:featureList],featuresDict[:powerBARSs]))
    sortedD = sort(D, byvalue = true, rev=true)

    modelVariables = OrderedDict()
    for keyNames∈ sortedD.keys
        if length(featuresDict[keyNames][:targetsAbove])>0
            modelVariables[keyNames] = featuresDict[keyNames][:targetsAbove]
        end
    end

    return modelVariables

end
