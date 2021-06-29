"""
FourierLayer(in, out, σ=identity, init=glorot_uniform)
FourierLayer(W::AbstractMatrix, [bias, σ])

Create a Layer of the Fourier Neural Operator as proposed by Zongyi et al.
arXiv: 2010.08895

The input `x` should be a vector of length `in`, or batch of vectors represented
as an `in × N` matrix, or any array with `size(x,1) == in`.
The out `y` will be a vector  of length `out`, or a batch with
`size(y) == (out, size(x)[2:end]...)`

You can specify biases for the paths as you like, though the convolutional path is
originally not intended to perform an affine transformation.
"""
# Create the data structure
struct FourierLayer{F, Mf<:AbstractMatrix, Ml<:AbstractMatrix, Bf, Bl}
    weight_f::Mf
    weight_l::Ml
    bias_f::Bf
    bias_l::Bl
    σ::F
    # Constructor for the entire fourier layer
    function FourierLayer(Wf::Mf, Wl::Ml, bias_l = true, bias_f = true, σ::F = identity) where {Mf<:AbstractMatrix, Ml<:AbstractMatrix, F}
        bf = Flux.create_bias(Wf, bias_f, size(Wf,1))
        bl = Flux.create_bias(Wl, bias_l, size(Wl, 1))
        new{F,Mf,Ml,typeof(bf),typeof(bl)}(Wf, Wl, bf, bl, σ)
    end
end

# Declare the function that assigns Weights and biases to the layer
function FourierLayer(in::Integer, out::Integer, σ = identity;
                    init = Flux.glorot_uniform, bias_linear=true, bias_fourier=true)
    
    Wf = init(floor(Int, out / 2)+1, floor(Int, in / 2)+1)
    Wl = init(out, in)

    bf = bias_linear
    bl = bias_fourier

    return FourierLayer(Wf, Wl, bf, bl, σ)
end

Flux.@functor FourierLayer

# The actual layer that does stuff
function (a::FourierLayer)(x::AbstractVecOrMat)
    # Assign the parameters
    Wf, Wl, bf, bl, σ = a.weight_f, a.weight_l, a.bias_f, a.bias_l, a.σ
    # The linear path
    linear = Wl * x .+ bl
    # The convolution path
    fourier = irfft((Wf * rfft(x) .+ bf), length(Wl[:,1]))
    # Return the activated sum
    return σ.(linear + fourier)
end

# Overload function to deal with higher-dimensional input arrays
(a::FourierLayer)(x::AbstractArray) = reshape(a(reshape(x, size(x, 1), :)), :, size(x)[2:end]...)

# Print nicely
function Base.show(io::IO, l::FourierLayer)
    print(io, "FourierLayer with\nConvolution path: (", size(l.weight_f, 2), ", ", size(l.weight_f, 1))
    print(io, ")\n")
    print(io, "Linear path: (", size(l.weight_l, 2), ", ", size(l.weight_l, 1))
    print(io, ")\n")
    l.σ == identity || print(io, "Activation: ", l.σ)
end