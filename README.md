
# OperatorLearning.jl

<p align="center">
<img width="400px" src="https://operatorlearning.sciml.ai/dev/assets/logo.png"/>
</p>

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://operatorlearning.sciml.ai/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://operatorlearning.sciml.ai/dev)
[![Build Status](https://github.com/pzimbrod/OperatorLearning.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/pzimbrod/OperatorLearning.jl/actions/workflows/CI.yml?query=branch%3Amaster++)
[![Build status](https://badge.buildkite.com/f0b3743420ce32c7b6f8fe974440b6fed08d68c5a244348924.svg)](https://buildkite.com/julialang/operatorlearning-dot-jl)
[![codecov](https://codecov.io/gh/pzimbrod/OperatorLearning.jl/branch/master/graph/badge.svg?token=NM16L5S4FX)](https://codecov.io/gh/pzimbrod/OperatorLearning.jl)

A Package that provides Layers for the learning of (nonlinear) operators in order to solve parametric PDEs.

For now, this package contains the Fourier Neural Operator originally proposed by Li et al [1] as well as the DeepONet conceived by Lu et al [2].

I decided to implement this method in Julia because coding up a layer using PyTorch in Python is rather cumbersome in comparison and Julia as a whole simply runs at comparable or faster speed than Python.

The implementation of the layers is influenced heavily by the basic layers provided in the [Flux.jl](https://github.com/FluxML/Flux.jl) package.

## Installation

Simply install by running in a REPL:

```julia
pkg> add OperatorLearning
```

## Usage/Examples

### Fourier Layer

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

To see a full implementation, check the Burgers equation example at `examples/burgers_FNO.jl`.
Compared to the original implementation by [Li et al.](https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_1d.py) using PyTorch, this version written in Julia clocks in about 20 - 25% faster when running on a NVIDIA RTX A5000 GPU.

If you'd like to replicate the example, you need to get the dataset for learning the Burgers equation. You can get it [here](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) or alternatively use the provided [scripts](https://github.com/zongyi-li/fourier_neural_operator/tree/master/data_generation/burgers).

### DeepONet

The `DeepONet` function basically sets up two separate Flux `Chain` structs and transforms the two input arrays into one via einsum/dot product.

You can either set up a "vanilla" DeepONet via the constructor function which sets up `Dense` layers for you or, if you feel fancy, pass two Chains directly to the function so you can use other architectures such as CNN or RNN as well.
The former takes two tuples that describe each architecture. E.g. `(32,64,72)` sets up a DNN with 32 neurons in the first, 64 in the second and 72 in the last layer.

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

For usage, check the Burgers equation example at `examples/burgers_DeepONet.jl`.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## ToDos

- [x] 1D Fourier Layer
- [ ] 2D / 3D Fourier Layer
- [x] DeepONet
- [ ] Physics informed Loss

## Contributing

Contributions are always welcome! Please submit a PR if you'd like to participate in the project.

## References

[1] Z. Li et al., „Fourier Neural Operator for Parametric Partial Differential Equations“, [arXiv:2010.08895](https://arxiv.org/abs/2010.08895) [cs, math], May 2021

[2] L. Lu, P. Jin, and G. E. Karniadakis, „DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators“, [arXiv:1910.03193](http://arxiv.org/abs/1910.03193) [cs, stat], Apr. 2020
