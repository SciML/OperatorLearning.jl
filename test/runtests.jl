using NeuralOperator
using Test
using Random

Random.seed!(0)

@testset "FourierLayer" begin
    include("fourierlayer.jl")
end
