module NeuralOperator

using Tullio: zeros
using Base: Integer, ident_cmp
using Flux
using FFTW
using Random
using Random: AbstractRNG
using Flux: nfan, glorot_uniform, batch
using Tullio

export FourierLayer

include("FourierLayer.jl")
include("ComplexWeights.jl")

end # module
