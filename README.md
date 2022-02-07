
# OperatorLearning.jl

<p align="center">
<img width="400px" src="https://pzimbrod.github.io/OperatorLearning.jl/dev/assets/logo.png"/>
</p>

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://pzimbrod.github.io/OperatorLearning.jl/dev)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://pzimbrod.github.io/OperatorLearning.jl/stable)
[![Build Status](https://github.com/pzimbrod/OperatorLearning.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/pzimbrod/OperatorLearning.jl/actions/workflows/CI.yml?query=branch%3Amaster++)
[![codecov](https://codecov.io/gh/pzimbrod/OperatorLearning.jl/branch/master/graph/badge.svg?token=NM16L5S4FX)](https://codecov.io/gh/pzimbrod/OperatorLearning.jl)

A Package that provides Layers for the learning of (nonlinear) operators in order to solve parametric PDEs.

For now, this package contains the Fourier Neural Operator originally proposed by Li et al [1] as well as the DeepONet conceived by Lu et al [2].

I decided to implement this method in Julia because coding up a layer using PyTorch in Python is rather cumbersome in comparison and Julia as a whole simply runs at comparable or faster speed than Python. Please do check out the [original work](https://github.com/zongyi-li/fourier_neural_operator) at GitHub as well.

The implementation of the layers is influenced heavily by the basic layers provided in the [Flux.jl](https://github.com/FluxML/Flux.jl) package.

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

# Input = 101, Output = 101, Grid points = 100, Fourier modes = 16
# Activation: sigmoid (you need to import Flux in your Script to access the activations)
model = FourierLayer(101, 101, 100, 16, σ)

# Same as above, but perform strict convolution in Fourier Space
model = FourierLayer(101, 101, 100, 16, σ; bias_fourier=false)
```

To see a full implementation, check the Burgers equation example at `examples/burgers.jl`.
Compared to the original implementation by [Li et al.](https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_1d.py) using PyTorch, this version written in Julia clocks in about 20 - 25% faster when running on a NVIDIA RTX A5000 GPU.

If you'd like to replicate the example, you need to get the dataset for learning the Burgers equation. You can get it [here](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) or alternatively use the provided [scripts](https://github.com/zongyi-li/fourier_neural_operator/tree/master/data_generation/burgers).

## License

[MIT](https://choosealicense.com/licenses/mit/)

## ToDos

- [x] 1D Fourier Layer
- [ ] 2D / 3D Fourier Layer
- [ ] DeepONet
- [ ] Physics informed Loss

## Contributing

Contributions are always welcome! Please submit a PR if you'd like to participate in the project.

## References

[1] Z. Li et al., „Fourier Neural Operator for Parametric Partial Differential Equations“, [arXiv:2010.08895](https://arxiv.org/abs/2010.08895) [cs, math], May 2021
[2] L. Lu, P. Jin, and G. E. Karniadakis, „DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators“, [arXiv:1910.03193](http://arxiv.org/abs/1910.03193) [cs, stat], Apr. 2020
