using Flux: length, reshape
using NeuralOperator, Flux, MAT

# Read the data from MAT file and store it in a dict
vars = matread("burgers_data_R10.mat")

# For trial purposes, we might want to train with different resolutions
# So we sample only every n-th element
subsample = 2^3;

# create the x training array, according to our desired grid size
xtrain = vars["a"][1:1000, 1:subsample:end];
# create the x test array
xtest = vars["a"][end-99:end, 1:subsample:end];

# Create the y training array
ytrain = vars["u"][1:1000, 1:subsample:end];
# Create the y test array
ytest = vars["u"][end-99:end, 1:subsample:end];

# The data is missing grid data, so we create it
# `collect` converts data type `range` into an array
grid = collect(range(0, 1, length=length(xtrain[1,:])))

# Merge the created grid with the data
# Output has the dims: batch x grid points x 2  (a(x), x)
# First, reshape the data to a 3D tensor,
# Then, create a 3D tensor from the synthetic grid data
# and concatenate them along the newly created 3rd dim
xtrain = cat(reshape(xtrain,(1000,1024,1)),
            reshape(repeat(grid,1000),(1000,1024,1));
            dims=3)
ytrain = cat(reshape(ytrain,(1000,1024,1)),
            reshape(repeat(grid,1000),(1000,1024,1));
            dims=3)
# Same treatment with the test data
xtest = cat(reshape(xtest,(100,1024,1)),
            reshape(repeat(grid,100),(100,1024,1));
            dims=3)
ytest = cat(reshape(ytest,(100,1024,1)),
            reshape(repeat(grid,100),(100,1024,1));
            dims=3)

# Our net wants the input in the form (2,batch,grid), though,
# So we permute
xtrain, xtest = permutedims(xtrain,(3,1,2)), permutedims(xtest,(3,1,2))
ytrain, ytest = permutedims(ytrain,(3,1,2)), permutedims(ytest,(3,1,2))

# Pass the data to the Flux DataLoader and give it a batch of 20
train_loader = Flux.Data.DataLoader((xtrain, ytrain), batchsize=20)
test_loader = Flux.Data.DataLoader((xtest, ytest), batchsize=20)

# Set up the Fourier Layer
# 128 in- and outputs, batch size 20 as given above, grid size 1024
# 16 modes to keep, σ activation on the gpu
layer = FourierLayer(128,128,20,1024,16,σ)

# The whole architecture
# linear transform into the latent space, 4 Fourier Layers,
# then transform it back
model = Chain(Dense(2,128;bias=false), layer, layer, layer, layer,
                Dense(128,2;bias=false))

# We use the ADAM optimizer for training
opt = ADAM()

# Specify the model parameters
parameters = params(model)

# The loss function
loss(x,y) = Flux.Losses.mse(model(x),y)

# Define a callback function that gives some output during training
evalcb() = @show(loss(x,y))