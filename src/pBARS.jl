module pBARS

using BARS
using OrderedCollections

function makeTargetsDict(dataFrame,featureName,acceptance)

    α=acceptance

    BARScutoff = .7
    BARSsAbove = []
    BARSsBelow = []
    rAbove = []
    rBelow =[]
    R²Above = []
    R²Below =[]
    targetsAbove =[]
    targetsBelow = []

    targetDataFrame = dataFrame[:,Not(featureName)]
    featureDataFrame = DataFrame()
    featureDataFrame[:,featureName] = dataFrame[:,featureName]

    selfBARSDict = bars(featureDataFrame,α)

    targetsDict = OrderedDict()
    targetsDict[:featureNameDict] = selfBARSDict
    targetsDict[featureName] = string(featureName)
    targetsDict[:r₀] = selfBARSDict[:r₀]
    targetsDict[:α] = α

    for targetName∈propertynames(targetDataFrame)
        
        featureDataFrame[:,targetName]=dataFrame[:,targetName]

        BARSDict = bars(featureDataFrame,selfBARSDict,α)

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

targetsDict[:targetsAbove] = targetsAbove 
targetsDict[:BARSsAbove] = BARSsAbove
targetsDict[:rAbove] = rAbove
targetsDict[:R²Above] = R²Above
targetsDict[:targetsBelow] = targetsBelow
targetsDict[:BARSsBelow] = BARSsBelow
targetsDict[:rBelow] = rBelow
targetsDict[:R²Below] = R²Below


return targetsDict


end

function calcPBARS(targetsDict)

    if length(targetsDict[:R²Above])>0 && length(targetsDict[:BARSsAbove])>0

        targetsDict[:pBARS] = sum(targetsDict[:R²Above].*targetsDict[:BARSsAbove])

    else

        targetsDict[:pBARS] = 0

    end

    return targetsDict

end

function makeFeaturesDict(featuresDataFrame,targetsDataFrame,acceptance)

    featuresDict = OrderedDict()
    powerBARSs = []

    for featureName∈propertynames(featuresDataFrame)

        df = DataFrame()
        df[:,featureName]=featuresDataFrame[:,featureName]
        df = hcat(df,targetsDataFrame)
        
        targetsDict = makeTargetsDict(df,featureName,acceptance)
        targetsDict = calcPBARS(targetsDict)

        featuresDict[:featureName] = targetsDict
        push!(powerBARSs,targetDict[:pBARS])

    end

    featuresDict[:powerBARSs]
    featuresDict[propertyname(featuresDataFrame)]

    return featuresDict

end

function plotFeatureBARS(targetsDict)

    BARScutoff = string(round(.7,sigdigits=3))
    r₀=targetsDict[:r₀]
    α=targetsDict[:α]
    BARSplot = plot(r₀-5:.01:r₀+5, r->exp.(-(r-r₀).^2/α^2),
    label = L"e^{-\frac{(r-r_0)^2}{\alpha^2}}",
    title = "Feature: "*targetsDict[featureName]*", PBARS= "*string(round(targetsDict[:pBARS],sigdigits=3)),
    legend = :outertopright)
    scatter!((targetsDict[:rAbove].+r₀),
            (targetsDict[:BARSsAbove]),
            markershape = :x,
            markercolor = :red,
            label = L"BARS_{Above}")
    scatter!((targetsDict[:rBelow].+r₀),
            (targetsDict[:BARSsBelow]),
            markershape = :x,
            markercolor = :blue,
            label = L"BARS_{Below}")
    hline!([.7],
            label=L"BARS_{cutoff}\approx"*BARScutoff,
            linestyle = :dash,
            linecolor = :red)
    vline!([r₀],
            label=L"r_0="*string(round(r₀,sigdigits=3)),
            linestyle = :dash,
            linecolor = :black)
    xlabel!(L"r-r_0=\frac{sMAE-sMAE_{\text{self}}}{nMAE}")
    ylabel!(L"BARS")
end

end