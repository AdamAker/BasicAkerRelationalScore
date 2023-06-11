using BasicAkerRelationalScore
using Test
using DataFrames
using OrderedCollections

@testset "BasicAkerRelationalScore.jl" begin

    α = .1
    t₀=0
    t = [t₀+i for i∈-10π:.1:10π]
    w(t)=t
    x(t)=t.^2
    y(t)=sin.(t)
    z(t)=exp.(t)
    dfself = DataFrame(X=x(t))
    df = DataFrame(X=x(t),Y=y(t))
    bigDataFrame = DataFrame(W=w(t),X=x(t),Y=y(t),Z=z(t),X²=x(t).*x(t),Y²=y(t).*y(t),WpY=w(t).+y(t),WY=w(t).*y(t))
    featureName = :X
    featureNames = [:W,:X,:Y,:Z]

    @testset "BasicAkerRelationalScore tests" begin

        @test typeof(BasicAkerRelationalScore.makeDataDict(dfself)) == Dict{Symbol,Any}

        selfDataDict = BasicAkerRelationalScore.makeDataDict(dfself)

        @test typeof(BasicAkerRelationalScore.makeBARSDict(selfDataDict)) == Dict{Symbol,Any}

        selfBARSDict = BasicAkerRelationalScore.makeBARSDict(selfDataDict)

        @test typeof(BasicAkerRelationalScore.calcSelfBARS(selfBARSDict)) == Dict{Symbol,Any}

        selfBARSDict = BasicAkerRelationalScore.calcSelfBARS(selfBARSDict)

        @test typeof(BasicAkerRelationalScore.makeDataDict(df)) == Dict{Symbol,Any}
        
        dataDict = BasicAkerRelationalScore.makeDataDict(df)

        @test typeof(BasicAkerRelationalScore.makeBARSDict(dataDict)) == Dict{Symbol,Any}

        BARSDict = BasicAkerRelationalScore.makeBARSDict(dataDict)

        @test typeof(BasicAkerRelationalScore.calcBARS(BARSDict,selfBARSDict, α)) == Dict{Symbol,Any}
    
        BARSDict = BasicAkerRelationalScore.calcBARS(BARSDict,selfBARSDict, α)

    end

    @testset "PBARS.jl" begin
        
        @test typeof(BasicAkerRelationalScore.makeTargetsDict(bigDataFrame,featureName,α)) == OrderedDict{Any,Any}
    
        targetsDict = BasicAkerRelationalScore.makeTargetsDict(bigDataFrame,featureName,α)
    
        @test typeof(BasicAkerRelationalScore.calcPBARS(targetsDict)) == OrderedDict{Any,Any}
    
        targetsDict = BasicAkerRelationalScore.calcPBARS(targetsDict)
    
    end

    @testset "modelVariables.jl" begin

        featuresDataFrame,targetsDataFrame = BasicAkerRelationalScore.splitDataFrame(bigDataFrame,featureNames)
        
        @test typeof(BasicAkerRelationalScore.makeFeaturesDict(featuresDataFrame,targetsDataFrame,α)) == OrderedDict{Any,Any}

        featuresDict = BasicAkerRelationalScore.makeFeaturesDict(featuresDataFrame,targetsDataFrame,α)

        @test typeof(BasicAkerRelationalScore.generateModelvars(featuresDict)) == OrderedDict{Any,Any}

        modelVars = BasicAkerRelationalScore.generateModelvars(featuresDict)

        featuresDict2 = BasicAkerRelationalScore.makeFeaturesDict(bigDataFrame,α)

        BasicAkerRelationalScore.matrixPlot("BARS ",featuresDict2[:BARSMatrix],bigDataFrame,.01,:orange,:tokyo,8)

    end

end
