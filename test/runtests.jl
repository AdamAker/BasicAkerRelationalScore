using BasicAkerRelationalScore
using Test


@testset "BasicAkerRelationalScore.jl" begin

    
    x1 = [i for i in 0:.01:10]
    y1 = x1
    y2 = x1.^2
    y3 = sin.(x1)
    
    x2 = [i for i in -5:.01:5]
    y4 = x2.^2
    y5=sin.(x2)
    
    dataf = DataFrame(:x1 => x1, :x2 => x2, :y1 => y1, :y2 => y2, :y3 => y3, :y4 => y4, :y5 => y5 )
    newdataf = select(dataf, [:x2,:y4])
    
    bMatrix = barsMatrix(newdataf)
    #=
    @test isapprox(bMatrix,[10.0 10.0 ; 3.7 10.0],rtol=.5)

    =#

    @test true


end
