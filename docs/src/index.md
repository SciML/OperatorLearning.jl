```@meta
CurrentModule = NeuralOperator
```

# NeuralOperator

Documentation for [NeuralOperator](https://github.com/Patrick Zimbrod/OperatorLearning.jl).

```@index
```

## Installation

Simply install by running in a REPL:

```julia
pkg> add NeuralOperator
```

## Usage/Examples

The basic workflow is more or less in line with the layer architectures that `Flux` provides, i.e. you construct individual layers, chain them if desired and pass the inputs as arguments to the layers.

The Fourier Layer performs a linear transform as well as convolution (linear transform in fourier space), adds them and passes it through the activation.
Additionally, higher Fourier modes are filtered out in the convolution path where you can specify the amount of modes to be kept.

The syntax for a single Fourier Layer is:

```julia
using NeuralOperator
using Flux

# Input = 101, Output = 101, Batch size = 200, Grid points = 100, Fourier modes = 16
# Activation: sigmoid (you need to import Flux in your Script to access the activations)
model = FourierLayer(101, 101, 200, 100, 16, σ)

# Same as above, but perform strict convolution in Fourier Space
model = FourierLayer(101, 101, 200, 100, 16, σ; bias_fourier=false)
```

To see a full implementation, check the Burgers equation example at `examples/burgers.jl`.
