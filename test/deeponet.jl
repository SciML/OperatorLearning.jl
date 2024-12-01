using Test, Random, Flux, MAT

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

#Just the first 16 datapoints from the Burgers' equation dataset
a = [0.83541104, 0.83479851, 0.83404712, 0.83315711, 0.83212979, 0.83096755, 0.82967374, 0.82825263, 0.82670928, 0.82504949, 0.82327962, 0.82140651, 0.81943734, 0.81737952, 0.8152405, 0.81302771]
sensors = collect(range(0, 1, length=16))'

model = DeepONet((16, 22, 30), (1, 16, 24, 30), σ, tanh; init_branch=Flux.glorot_normal, bias_trunk=false)

model(a,sensors)

#forward pass
@test size(model(a, sensors)) == (1, 16)

mgrad = Flux.Zygote.gradient((x,p)->sum(model(x,p)),a,sensors)

#gradients
@test !iszero(Flux.Zygote.gradient((x,p)->sum(model(x,p)),a,sensors)[1])
@test !iszero(Flux.Zygote.gradient((x,p)->sum(model(x,p)),a,sensors)[2])

#training
vars = matread("burgerset.mat")

xtrain = vars["a"][1:280, :]'
xval = vars["a"][end-19:end, :]'

ytrain = vars["u"][1:280, :]
yval = vars["u"][end-19:end, :]

grid = collect(range(0, 1, length=1024))'
model = DeepONet((1024,1024,1024),(1,1024,1024),gelu,gelu)

learning_rate = 0.001
opt = ADAM(learning_rate)

parameters = params(model)

loss(xtrain,ytrain,sensor) = Flux.Losses.mse(model(xtrain,sensor),ytrain)

evalcb() = @show(loss(xval,yval,grid))

Flux.@epochs 400 Flux.train!(loss, parameters, [(xtrain,ytrain,grid)], opt, cb = evalcb)

ỹ = model(xval, grid)

diffvec = vec(abs.((yval .- ỹ)))
mean_diff = sum(diffvec)/length(diffvec)
@test mean_diff < 0.4
