using Test, Random, Flux

@testset "Weights" begin
    @testset "uniform" begin
        @test size(OperatorLearning.cglorot_uniform(128, 64, 10)) == (128, 64, 10)
        @test eltype(OperatorLearning.cglorot_uniform(128, 64, 10)) == ComplexF32
    end

    @testset "normal" begin
        @test size(OperatorLearning.cglorot_normal(128, 64, 10)) == (128, 64, 10)
        @test eltype(OperatorLearning.cglorot_normal(128, 64, 10)) == ComplexF32
    end
end
