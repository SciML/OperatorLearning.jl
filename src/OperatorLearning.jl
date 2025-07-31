module OperatorLearning

using Base: Integer, ident_cmp, Float32
using CUDA
using Flux
using FFTW
using FFTW: assert_applicable, unsafe_execute!, FORWARD, BACKWARD, rFFTWPlan
using Random
using Random: AbstractRNG
using Flux: nfan, glorot_uniform, batch
using OMEinsum

export FourierLayer, DeepONet

include("FourierLayer.jl")
include("DeepONet.jl")
include("ComplexWeights.jl")
include("batched.jl")
include("subnets.jl")

end # module
