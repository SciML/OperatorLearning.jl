
# OperatorLearning.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://pzimbrod.github.io/OperatorLearning.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://pzimbrod.github.io/OperatorLearning.jl/dev)
[![Build Status](https://github.com/pzimbrod/OperatorLearning.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/pzimbrod/OperatorLearning.jl/actions/workflows/CI.yml?query=branch%3Amaster++)

A `Julia` implementation of the Fourier Neural Operator conceived by [Zongyi et al.](https://arxiv.org/abs/2010.08895) 
using (mainly) [Flux.jl](https://github.com/FluxML/Flux.jl) and [FFTW.jl](https://github.com/JuliaMath/FFTW.jl).

I decided to implement this method in Julia because coding up a layer using PyTorch in Python is rather cumbersome in comparison and Julia as a whole simply runs at comparable or faster speed than Python. Please do check out the [original work](https://github.com/zongyi-li/fourier_neural_operator) at GitHub as well.

The implementation of the layers is influenced heavily by the basic layers provided in the [Flux.jl](https://github.com/FluxML/Flux.jl) package.

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

## License

[MIT](https://choosealicense.com/licenses/mit/)

## ToDos

- [x] 1D Fourier Layer
- [ ] 2D / 3D Fourier Layer
- [ ] DeepONet
- [ ] Physics informed Loss

## Contributing

Contributions are always welcome!
