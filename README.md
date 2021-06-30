
# NeuralOperator.jl

A `Julia` implementation of the Fourier Neural Operator conceived by [Zongyi et al.](https://arxiv.org/abs/2010.08895) 
using [Flux.jl](https://github.com/FluxML/Flux.jl) and [FFTW.jl](https://github.com/JuliaMath/FFTW.jl).

I decided to implement this method in Julia because coding up a layer using PyTorch in Python is rather cumbersome in comparison and Julia as a whole simply runs at comparable or faster speed than Python. Please do check out the [original work](https://github.com/zongyi-li/fourier_neural_operator) at GitHub as well.

The implementation of the layers is influenced heavily by the basic layers provided in the [Flux.jl](https://github.com/FluxML/Flux.jl) package.

## Installation

This package is not yet released on the official channels. Therefore you have to add it through its GitHub link:

```julia
pkg> add https://github.com/pzimbrod/NeuralOperator.jl
```

## Usage/Examples

Usage is basically identical to Flux. When calling `FourierLayer`, a linear and a convolution path is constructed. The activation function acts on the sum of both paths and not on each individually.
For now, the convolution path does not filter modes, but does a full linear transform in Fourier space. You can make this an affine transformation by specifying a nonzero bias.

```julia
using NeuralOperator

# Some input
t = [0:0.1:10;]
x = sin.(t) + atan.(t)

# Create a matching Fourier Layer
model = FourierLayer(101, 101)

# Do some calcs
res = model(x)

# Perform strict convolution in Fourier Space
model = FourierLayer(101, 101; bias_fourier=false)

```

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Contributing

Contributions are always welcome!
