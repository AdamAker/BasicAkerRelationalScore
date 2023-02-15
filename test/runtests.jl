using BARS
using Test
using DataFrames

@testset "BARS.jl" begin

    α = .1
    t₀=0
    t = [t₀+i for i∈0:.01:π/2]
    x(t)=sin.(t)
    y(t)=t.^2
    dfself = DataFrame(X=x(t))
    df = DataFrame(X=x(t),Y=y(t))
    bigDataFrame = DataFrame(X=x(t),Y=y(t),X²=x(t).*x(t),Y²=y(t).*y(t))

    @testset "BARS tests" begin

        @test typeof(BARS.makeDataDict(dfself)) == Dict{Symbol,Any}

        selfDataDict = BARS.makeDataDict(dfself)

        @test typeof(BARS.makeBARSDict(selfDataDict)) == Dict{Symbol,Any}

        selfBARSDict = BARS.makeBARSDict(selfDataDict)

        @test typeof(BARS.calcSelfBARS(selfBARSDict)) == Dict{Symbol,Any}

        selfBARSDict = BARS.calcSelfBARS(selfBARSDict)

        @test typeof(BARS.makeDataDict(df)) == Dict{Symbol,Any}
        
        dataDict = BARS.makeDataDict(df)

        @test typeof(BARS.makeBARSDict(dataDict)) == Dict{Symbol,Any}

        BARSDict = BARS.makeBARSDict(dataDict)

        @test typeof(BARS.calcBARS(BARSDict,selfBARSDict, α)) == Dict{Symbol,Any}
    
        BARSDict = BARS.calcBARS(BARSDict,selfBARSDict, α)

    end

    @testset "PBARS Tests" begin
        
        @test true

    end

end