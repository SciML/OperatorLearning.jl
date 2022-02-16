using Test, Random, Flux

@testset "FourierLayer" begin
    @testset "dimensions" begin
        @testset "1D" begin
            # Test the proper construction
            @test size(FourierLayer(128, 64, 100, 20).Wf) == (128, 64, 100)
            @test size(FourierLayer(128, 64, 100, 20).Wl) == (64, 128)
            @test size(FourierLayer(128, 64, 100, 20).bf) == (1, 64, 100)
            @test size(FourierLayer(128, 64, 100, 20).bl) == (1, 64, 100)
        end
        @testset "3D" begin
            # Test the proper construction
            @test size(FourierLayer(128, 64, (100,150,200), (20,30,40)).Wf) == (128, 64, 100, 150, 200)
            @test size(FourierLayer(128, 64, (100,150,200), (20,30,40)).Wl) == (64, 128)
            @test size(FourierLayer(128, 64, (100,150,200), (20,30,40)).bf) == (1, 64, 100, 150, 200)
            @test size(FourierLayer(128, 64, (100,150,200), (20,30,40)).bl) == (1, 64, 100, 150, 200)
        end
    end

    # Test proper parameter assignment
    # We only use a subset of the weight tensors for training
    @testset "parameters" begin
        # Wf
        @test size(params(FourierLayer(128, 64, 100, 20))[1]) == (128, 64, 20)
        # Wl
        @test size(params(FourierLayer(128, 64, 100, 20))[2]) == (64, 128)
        # bf
        @test size(params(FourierLayer(128, 64, 100, 20))[3]) == (1, 64, 20)
        # bl
        @test size(params(FourierLayer(128, 64, 100, 20))[4]) == (1, 64, 100)
    end

    # Accept only Int as architecture parameters
    @test_throws MethodError FourierLayer(128.5, 64, 100, 20)
    @test_throws MethodError FourierLayer(128.5, 64, 100, 20, tanh)
    # Test max amount of modes
    @test_throws AssertionError FourierLayer(100, 100, 100, 60, σ)
    @test_throws AssertionError FourierLayer(100, 100, 100, 60)
end