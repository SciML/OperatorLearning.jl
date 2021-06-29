using Test, Random
import Flux: activations

@testset "FourierLayer" begin
    # Test the proper construction
    @test size(FourierLayer(10, 100).weight_f) == (6, 6)
    @test size(FourierLayer(10, 100).weight_l) == (100, 10)
    @test size(FourierLayer(10, 100).bias_f) == (6,)
    @test size(FourierLayer(10, 100).bias_l) == (100,)
    # Accept only Int as architecture parameters
    @test_throws MethodError FourierLayer(10, 10.5)
    @test_throws MethodError FourierLayer(10, 10.5, tanh)
end