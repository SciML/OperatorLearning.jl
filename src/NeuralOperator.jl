module NeuralOperator

using Base: Integer, ident_cmp
using Flux
using FFTW

export FourierLayer

include("FourierLayer.jl")

end # module
