using Pkg
const LONGER_TESTS = false
const GROUP = get(ENV, "GROUP", "All")

using OperatorLearning
using Test
using Random

Random.seed!(0)

if GROUP == "All" || GROUP == "Core"
    @testset "FourierLayer" begin
        include("fourierlayer.jl")
    end

    @testset "DeepONet" begin
        include("deeponet.jl")
    end

    @testset "Weights" begin
        include("complexweights.jl")
    end
end

if GROUP == "GPU"
    # Add GPU Tests Here
end
