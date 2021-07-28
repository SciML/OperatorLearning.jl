using Flux: length, reshape
using NeuralOperator, Flux, MAT

# Read the data from MAT file and store it in a dict
vars = matread("../burgers_data_R10.mat")

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
# Same treatment with the test data
xtest = cat(reshape(xtest,(1000,1024,1)),
            reshape(repeat(grid,1000),(1000,1024,1));
            dims=3)

# Our net wants the input in the form (2,batch,grid), though,
# So we permute
xtrain, xtest = permutedims(xtrain,(3,1,2)), permutedims(xtest,(3,1,2))