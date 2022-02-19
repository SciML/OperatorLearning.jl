using Test, Random, Flux

@testset "FourierLayer" begin
    @testset "dimensions" begin
        # Test the proper construction
        @test size(FourierLayer(128, 64, 100, 20).Wf) == (128, 64, 51)
        @test size(FourierLayer(128, 64, 100, 20).Wl) == (64, 128)
        @test size(FourierLayer(128, 64, 100, 20).bf) == (1, 64, 51)
        @test size(FourierLayer(128, 64, 100, 20).bl) == (1, 64, 100)
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

#Just the first 16 data points from Burgers' equation dataset
xtrain = Float32[0.83541104, 0.83479851, 0.83404712, 0.83315711, 0.83212979, 0.83096755, 0.82967374, 0.82825263, 0.82670928, 0.82504949, 0.82327962, 0.82140651, 0.81943734, 0.81737952, 0.8152405, 0.81302771]
grid = Float32.(collect(range(0, 1, length=16))')

x = cat(reshape(xtrain,(1,16,1)),
        reshape(repeat(grid,1),(1,16,1));
        dims=3)

x = permutedims(x,(3,2,1))
layer = FourierLayer(64, 64, 16, 8, gelu, bias_fourier=false)
model = Chain(Dense(2,64;bias=false), layer, layer, layer, layer,
                Dense(64,2;bias=false))

model(x)

#forward pass
@test size(model(x)) == (2, 16, 1)

Flux.Zygote.gradient((x)->sum(model(x)), x)

#gradient test
@test !iszero(Flux.Zygote.gradient((x)->sum(model(x)), x)[1])
