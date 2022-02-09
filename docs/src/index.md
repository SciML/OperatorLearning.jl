```@meta
CurrentModule = OperatorLearning
```

# OperatorLearning

A Package that provides Layers for the learning of (nonlinear) operators in order to solve parametric PDEs.

!!! note

    This package is still under heavy development and there are likely still a few things to iron out. If you find a bug or something to improve, please feel free to open a new issue or submit a PR in the [GitHub Repo](https://github.com/pzimbrod/OperatorLearning.jl)

## Installation

Simply install by running in a REPL:

```julia
pkg> add OperatorLearning
```

## Usage

In total, the exported layers behave like you would expect from ones that `Flux.jl` provides, i.e. you can use basically all the tools that come along with `Flux` to do training.

### Fourier Neural Operator

The basic workflow is more or less in line with the layer architectures that `Flux` provides, i.e. you construct individual layers, chain them if desired and pass the inputs as arguments to the layers.

The Fourier Layer performs a linear transform as well as convolution (linear transform in fourier space), adds them and passes it through the activation.
Additionally, higher Fourier modes are filtered out in the convolution path where you can specify the amount of modes to be kept.

The syntax for a single Fourier Layer is:

```julia
using OperatorLearning
using Flux

# Input = 101, Output = 101, Grid points = 100, Fourier modes = 16
# Activation: sigmoid (you need to import Flux in your Script to access the activations)
model = FourierLayer(101, 101, 100, 16, σ)

# Same as above, but perform strict convolution in Fourier Space
model = FourierLayer(101, 101, 100, 16, σ; bias_fourier=false)
```

To see a full implementation, check the corresponding [Burgers equation example](examples/burgers_FNO.md).

### DeepONet

The workflow here is a little different than with Fourier Neural Operator. In this case, you create the entire architecture by specifying two tuples corresponding to the architecture of branch and trunk net.

This creates a "vanilla" DeepONet where branch and trunk net are simply Chains of Dense layers. You can however use any other architecture in the subnets as well, as long as the outputs of the two match. Otherwise, the contraction operation won't work due to dimension mismatch.

```julia
using OperatorLearning
using Flux

# Create a DeepONet with branch 32 -> 64 -> 72 and sigmoid activation
# and trunk 24 -> 64 -> 72 and tanh activation without biases
model = DeepONet((32,64,72), (24,64,72), σ, tanh; init_branch=Flux.glorot_normal, bias_trunk=false)

# Alternatively, set up your own nets altogether and pass them to DeepONet
branch = Chain(Dense(2,128),Dense(128,64),Dense(64,72))
trunk = Chain(Dense(1,24),Dense(24,72))
model = DeepONet(branch,trunk)
```

To see a full implementation, check the corresponding [Burgers equation example](examples/burgers_DeepONet.md).
