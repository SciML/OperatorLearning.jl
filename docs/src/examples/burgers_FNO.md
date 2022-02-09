# Solving the Burgers Equation with the Fourier Neural Operator

This example mostly replicates the original work by [Li et al](https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_1d.py).

We try to create an operator for the Burgers equation

$$ \partial_t u(x,t) + \partial_x (u^2(x,t)/2) = \nu \partial_{xx} u(x,t) $$

in one dimension for a unit spacial and temporal domain. The operator maps the initial condition $u(x,0) = u_0(x)$ to the flow field at the final time $u(x,1)$.

So overall, we need an approximation function that does the following:

```julia
function foo(u0,x)
    # Do something
    return u1
```

We sample from a dataset that contains several instances of the initial condition (`a`) and the final velocity field (`u`).
The data is given on a grid of 8192 points, however we would like to only sample 1024 points.

```julia
using Flux: length, reshape, train!, throttle, @epochs
using OperatorLearning, Flux, MAT

device = gpu;

# Read the data from MAT file and store it in a dict
vars = matread("burgers_data_R10.mat") |> device

# For trial purposes, we might want to train with different resolutions
# So we sample only every n-th element
subsample = 2^3;

# create the x training array, according to our desired grid size
xtrain = vars["a"][1:1000, 1:subsample:end] |> device;
# create the x test array
xtest = vars["a"][end-99:end, 1:subsample:end] |> device;

# Create the y training array
ytrain = vars["u"][1:1000, 1:subsample:end] |> device;
# Create the y test array
ytest = vars["u"][end-99:end, 1:subsample:end] |> device;
```

For now, we only have one input and one output array. In addition, we need corresponding x values for a(x) and u(x) as the second input array which at this point are still missing. The data were sampled from an equispaced grid (otherwise the FFT in our architecture wouldn't work anyway), so manually creating them is fairly straightforward:

```julia
# The data is missing grid data, so we create it
# `collect` converts data type `range` into an array
grid = collect(range(0, 1, length=length(xtrain[1,:]))) |> device

# Merge the created grid with the data
# Output has the dims: batch x grid points x 2  (a(x), x)
# First, reshape the data to a 3D tensor,
# Then, create a 3D tensor from the synthetic grid data
# and concatenate them along the newly created 3rd dim
xtrain = cat(reshape(xtrain,(1000,1024,1)),
            reshape(repeat(grid,1000),(1000,1024,1));
            dims=3) |> device
ytrain = cat(reshape(ytrain,(1000,1024,1)),
            reshape(repeat(grid,1000),(1000,1024,1));
            dims=3) |> device
# Same treatment with the test data
xtest = cat(reshape(xtest,(100,1024,1)),
            reshape(repeat(grid,100),(100,1024,1));
            dims=3) |> device
ytest = cat(reshape(ytest,(100,1024,1)),
            reshape(repeat(grid,100),(100,1024,1));
            dims=3) |> device
```

Next we need to consider the shape that the `FourierLayer` expects the inputs to be, i.e. `[numInputs, grid, batch]`. But our dataset contains the batching dim as the first one, so we need to do some permuting:

```julia
# Our net wants the input in the form (2,grid,batch), though,
# So we permute
xtrain, xtest = permutedims(xtrain,(3,2,1)), permutedims(xtest,(3,2,1)) |> device
ytrain, ytest = permutedims(ytrain,(3,2,1)), permutedims(ytest,(3,2,1)) |> device
```

In order to slice the data into mini-batches, we pass the arrays to the Flux `DataLoader`.

```julia
# Pass the data to the Flux DataLoader and give it a batch of 20
train_loader = Flux.Data.DataLoader((xtrain, ytrain), batchsize=20, shuffle=true) |> device
test_loader = Flux.Data.DataLoader((xtest, ytest), batchsize=20, shuffle=false) |> device
```

We can now set up the architecture. We lift the inputs to a higher-dimensional space via a simple linear transform using a `Dense` layer. The input dimensionality is 2, we will transform it to 128. After that, we set up 4 instances of a Fourier Layer where we keep only 16 of the `N/2 + 1 = 513` modes that the FFT provides and use the GeLU activation. Finally, we reduce the latent space to the two output arrays we wish to obtain - `u1(x)` and `x`:

```julia
# Set up the Fourier Layer
# 128 in- and outputs, batch size 20 as given above, grid size 1024
# 16 modes to keep, σ activation on the gpu
layer = FourierLayer(128,128,1024,16,gelu,bias_fourier=false) |> device

# The whole architecture
# linear transform into the latent space, 4 Fourier Layers,
# then transform it back
model = Chain(Dense(2,128;bias=false), layer, layer, layer, layer,
                Dense(128,2;bias=false)) |> device
```

The rest is more or less boilerplate training code for a DNN. We employ the ADAM optimizer with a fixed learning rate of 1e-3, use the mean squared error as loss, evaluate the test loss as callback and train the FNO for 500 epochs.

```julia
# We use the ADAM optimizer for training
learning_rate = 0.001
opt = ADAM(learning_rate)

# Specify the model parameters
parameters = params(model)

# The loss function
loss(x,y) = Flux.Losses.mse(model(x),y)

# Define a callback function that gives some output during training
evalcb() = @show(loss(xtest,ytest))
# Print the callback only every 5 seconds, 
throttled_cb = throttle(evalcb, 5)

# Do the training loop
Flux.@epochs 500 train!(loss, parameters, train_loader, opt, cb = throttled_cb)
```
