# Solving the Burgers Equation with DeepONet

This example mostly adapts the original work by [Li et al](https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_1d.py) to solving with DeepONet and is intended to provide an analogue to [the FNO example](burgers_FNO.md).

We try to create an operator for the Burgers equation

$\partial_t u(x,t) + \partial_x (u^2(x,t)/2) = \nu \partial_{xx} u(x,t)$

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

device = cpu;

#=
We would like to implement and train a DeepONet that infers the solution
u(x) of the burgers equation on a grid of 1024 points at time one based
on the initial condition a(x) = u(x,0)
=#

# Read the data from MAT file and store it in a dict
# key "a" is the IC
# key "u" is the desired solution at time 1
vars = matread("burgers_data_R10.mat") |> device

# For trial purposes, we might want to train with different resolutions
# So we sample only every n-th element
subsample = 2^3;

# create the x training array, according to our desired grid size
xtrain = vars["a"][1:1000, 1:subsample:end]' |> device;
# create the x test array
xtest = vars["a"][end-99:end, 1:subsample:end]' |> device;

# Create the y training array
ytrain = vars["u"][1:1000, 1:subsample:end] |> device;
# Create the y test array
ytest = vars["u"][end-99:end, 1:subsample:end] |> device;
```

One particular thing to note here is that we need to permute the array containing the initial condition so that the inner product of DeepONet works. This is because we need to do the following contraction:

$\sum\limits_i t_{ji} b_{ik} = u_{jk}$

For now, we only have one input and one output array. In addition, we need another input array that provides the probing locations for the operator $u_1(x) = \mathcal{G}(u_0)(x)$. In theory, we could choose those arbitrarily. For sake of simplicity though, we simply create the same equispaced grid that the original data was sampled from, i.e. a 1-D grid of 1024 equispaced points in [0;1]. Again, we need to transpose the array so that the array dim that is transformed by the trunk network is in the first column - otherwise the inner product would be much more cumbersome to handle.

```julia
# The data is missing grid data, so we create it
# `collect` converts data type `range` into an array
grid = collect(range(0, 1, length=1024))' |> device
```

We can now set up the DeepONet. We choose the latent space to have dimensionality 1024 and use the vanilla DeepONet architecture, i.e. we use `Dense` layers in both branch and trunk net. Both contain two layers and use the GeLU activation function:

```julia
# Create the DeepONet:
# IC is given on grid of 1024 points, and we solve for a fixed time t in one
# spatial dimension x, making the branch input of size 1024 and trunk size 1
# We choose GeLU activation for both subnets
model = DeepONet((1024,1024,1024),(1,1024,1024),gelu,gelu) |> device
```

The rest is more or less boilerplate training code for a DNN, *with one exception*: For the loss to compute properly, we need to pass two separate input arrays for the branch and trunk network each. We employ the ADAM optimizer with a fixed learning rate of 1e-3, use the mean squared error as loss, evaluate the test loss as callback and train the FNO for 500 epochs.

```julia
# We use the ADAM optimizer for training
learning_rate = 0.001
opt = ADAM(learning_rate)

# Specify the model parameters
parameters = params(model)

# The loss function
# We can't use the "vanilla" implementation of the mse here since we have
# two distinct inputs to our DeepONet, so we wrap them into a tuple
loss(xtrain,ytrain,sensor) = Flux.Losses.mse(model(xtrain,sensor),ytrain)

# Define a callback function that gives some output during training
evalcb() = @show(loss(xtest,ytest,grid))
# Print the callback only every 5 seconds
throttled_cb = throttle(evalcb, 5)

# Do the training loop
Flux.@epochs 500 train!(loss, parameters, [(xtrain,ytrain,grid)], opt, cb = evalcb)
```
