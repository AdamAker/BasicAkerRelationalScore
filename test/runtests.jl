using BARS
using Test
using DataFrames

@testset "BARS.jl" begin

    α = .01
    t₀=0
    t = [t₀+i for i∈0:.01:π/2]
    x(t)=sin.(t)
    y(t)=t.^2
    dfself = DataFrame(X=x(t),Y=x(t))
    df = DataFrame(X=x(t),Y=y(t))

    @testset "BARS Tree tests" begin
        
        @test typeof(bars(dfself)) == Dict{Symbol,Any}

        @test typeof(bars(df)) == Dict{Symbol,Any}

    end

end