using OperatorLearning
using Test
using Random

Random.seed!(0)

@testset "FourierLayer" begin
    include("fourierlayer.jl")
end

@testset "DeepONet" begin
    include("deeponet.jl")
end

@testset "Weights" begin
    include("complexweights.jl")
end
