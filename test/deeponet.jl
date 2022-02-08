using Test, Random, Flux

@testset "DeepONet" begin
    @testset "dimensions" begin
        # Test the proper construction
        # Branch net
        @test size(DeepONet((32,64,72), (24,48,72), σ, tanh).branch_net.layers[end].weight) == (72,64)
        @test size(DeepONet((32,64,72), (24,48,72), σ, tanh).branch_net.layers[end].bias) == (72,)
        # Trunk net
        @test size(DeepONet((32,64,72), (24,48,72), σ, tanh).trunk_net.layers[end].weight) == (72,48)
        @test size(DeepONet((32,64,72), (24,48,72), σ, tanh).trunk_net.layers[end].bias) == (72,)
    end

    # Accept only Int as architecture parameters
    @test_throws MethodError DeepONet((32.5,64,72), (24,48,72), σ, tanh)
    @test_throws MethodError DeepONet((32,64,72), (24.1,48,72))
end