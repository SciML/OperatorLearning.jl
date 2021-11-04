module NeuralOperator

using Base: Integer, ident_cmp, Float32
using CUDA
using Flux
using FFTW
using FFTW: assert_applicable, unsafe_execute!, FORWARD, BACKWARD, rFFTWPlan
using Random
using Random: AbstractRNG
using Flux: nfan, glorot_uniform, batch

export FourierLayer
export batched_mul!

include("FourierLayer.jl")
include("ComplexWeights.jl")
include("batched.jl")

end # module
