using OperatorLearning, Flux

versioninfo()
include("bench_utils.jl")

@info "Benchmark FourierLayer"
include("benchFourierLayer.jl")
