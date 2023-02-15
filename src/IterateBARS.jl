module pBARS

using BARS

function makeTargetsDict(dataFrame,featureName,acceptance)

    α=acceptance

    BARScutoff = exp(-1)
    BARSsAbove = []
    BARSsBelow = []
    rAbove = []
    rBelow =[]
    R²Above = []
    R²Below =[]
    targetsAbove =[]
    targetsBelow = []

    targetDataFrame = dataFrame[:,Not(featureName)]
    featureDataFrame = dataFrame[:,featureName]

    selfBARSDict = bars(featureDataFrame,α)

    targetsDict = Dict()
    targetsDict[featureName] = selfBARSDict

    for targetName∈propertynames(targetDataFrame)
        
        featureDataFrame[:,targetName]=dataFrame[:,targetName]

        BARSDict = bars(df,selfBARSDict,α)

        if BARSDict[:BARS]>BARScutoff

            push!(BARSsAbove, BARSDict[:BARS])
            push!(rAbove, BARSDict[:r])
            push!(R²Above, BARSDict[:R²])
            push!(targetsAbove, targetName)

        elseif BARSDict[:BARS]≤BARScutoff && !isequal(BARSDict[:BARS],NaN)

            push!(BARSsBelow, BARSDict[:BARS])
            push!(rBelow, BARSDict[:r])
            push!(R²Below, BARSDict[:R²])
            push!(targetsBelow, targetName)

        end

        select!(featureDataFrame,Not(targetName))

end

end