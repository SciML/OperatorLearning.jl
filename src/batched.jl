using CUDA: DenseCuArray
using CUDA.CUFFT: CuFFTPlan

"""
Function overloads for `batched_mul` provided by `NNlib` so that FFTW Plans can be handled.

For `mul!`, `FFTW.jl` already provides an implementation. However, we need to loop over the batch dimension.
"""

# Extension for CPU Arrays
for (Tr, Tc) in ((:Float32, :(Complex{Float32})), (:Float64, :(Complex{Float64})))
    # Note: use $FORWARD and $BACKWARD below because of issue #9775
    @eval begin
        function NNlib.batched_mul!(y::StridedArray{$Tc}, p::rFFTWPlan{$Tr, $FORWARD}, x::StridedArray{$Tr})
            assert_applicable(p, x[:, :, 1], y[:, :, 1]) # no need to check every batch dim
            @inbounds for k in 1:size(y, 3)
                @views unsafe_execute!(p, x[:, :, k], y[:, :, k])
            end
            return y
        end
        function NNlib.batched_mul!(y::StridedArray{$Tr}, p::rFFTWPlan{$Tc, $BACKWARD}, x::StridedArray{$Tc})
            assert_applicable(p, x[:, :, 1], y[:, :, 1]) # no need to check every batch dim
            @inbounds for k in 1:size(y, 3)
                @views unsafe_execute!(p, x[:, :, k], y[:, :, k])
            end
            return y
        end
    end
end

# Methods for GPU Arrays, borrowed from CUDA.jl -> fft.jl #490
function NNlib.batched_mul!(y::DenseCuArray{Ty}, p::CuFFTPlan{T, K, false}, x::DenseCuArray{T}
) where {Ty, T, K}
    CUFFT.assert_applicable(p, x[:, :, 1], y[:, :, 1])
    @inbounds for k in 1:size(y, 3)
        @views CUFFT.unsafe_execute!(p, x[:, :, k], y[:, :, k])
    end
    return y
end
