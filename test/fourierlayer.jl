using Test, Random, Flux

@testset "FourierLayer" begin
    # Test the proper construction
    @test size(FourierLayer(128, 64, 200, 100, 20).weight_f) == (128, 64, 51)
    @test size(FourierLayer(128, 64, 200, 100, 20).weight_l) == (64, 200, 128)
    #@test size(FourierLayer(10, 100).bias_f) == (51,)
    #@test size(FourierLayer(10, 100).bias_l) == (100,)

    # Accept only Int as architecture parameters
    @test_throws MethodError FourierLayer(128.5, 64, 200, 100, 20)
    @test_throws MethodError FourierLayer(128.5, 64, 200, 100, 20, tanh)
    @test_throws AssertionError FourierLayer(100, 100, 100, 100, 60, σ)
    @test_throws AssertionError FourierLayer(100, 100, 100, 100, 60)
end