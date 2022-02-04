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

## Usage/Examples

The basic workflow is more or less in line with the layer architectures that `Flux` provides, i.e. you construct individual layers, chain them if desired and pass the inputs as arguments to the layers.

The Fourier Layer performs a linear transform as well as convolution (linear transform in fourier space), adds them and passes it through the activation.
Additionally, higher Fourier modes are filtered out in the convolution path where you can specify the amount of modes to be kept.

The syntax for a single Fourier Layer is:

```julia
using OperatorLearning
using Flux

# Input = 101, Output = 101, Batch size = 200, Grid points = 100, Fourier modes = 16
# Activation: sigmoid (you need to import Flux in your Script to access the activations)
model = FourierLayer(101, 101, 200, 100, 16, σ)

# Same as above, but perform strict convolution in Fourier Space
model = FourierLayer(101, 101, 200, 100, 16, σ; bias_fourier=false)
```

To see a full implementation, check the Burgers equation example at `examples/burgers.jl`.
