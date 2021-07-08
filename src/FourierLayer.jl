"""
FourierLayer(in, out, σ=identity, init=glorot_uniform)
FourierLayer(W::AbstractMatrix, [bias, σ])

Create a Layer of the Fourier Neural Operator as proposed by Zongyi et al.
arXiv: 2010.08895

The layer does a fourier transform on the last axis (the coeffs) of the input array,
filters higher modes out by the weight matrix and transforms the second axis to the
specified output dimension such that M x In x N -> M x Out x N.
The output though only contains the relevant Fourier modes with the rest padded to zero
in the last axis as a result of the filtering.

The input `x` should be a 3D tensor of length `in` (number of grid points), 
width of the number of parameters considered plus the sparial coordinates and depth
of the number of training samples.
The out `y` will be a 3D tensor of length `out` (number of grid points), 
width of the number of solutions plus one spatial coordinate and depth of the
number of training samples.

You can specify biases for the paths as you like, though the convolutional path is
originally not intended to perform an affine transformation.

# Examples
Say you're considering a 1D diffusion problem on a 64 point grid. The input is comprised
of the grid points as well as the IC at this point.
The data consists of 200 instances of the solution.
So the input takes the dimension `64 x 2 x 200`.
The output would be the diffused variable at a later time, which makes the output of the form
`64 x 2 x 200` as well.
"""
# Create the data structure
struct FourierLayer{F, Mf<:AbstractArray, Ml<:AbstractArray, Bf, Bl}
    weight_f::Mf
    weight_l::Ml
    bias_f::Bf
    bias_l::Bl
    σ::F
    # Constructor for the entire fourier layer
    function FourierLayer(Wf::Mf, Wl::Ml, bias_f = true, bias_l = true, σ::F = identity) where {Mf<:AbstractArray, Ml<:AbstractArray, F}
        bf = Flux.create_bias(Wf, bias_f, size(Wf,1))
        bl = Flux.create_bias(Wl, bias_l, size(Wl, 1))
        new{F,Mf,Ml,typeof(bf),typeof(bl)}(Wf, Wl, bf, bl, σ)
    end
end

# Declare the function that assigns Weights and biases to the layer
# `in` and `out` refer to the dimensionality of the number of parameters
# `modes` specifies the number of modes not to be filtered out
# `grid` specifies the number of grid points in the data
function FourierLayer(in::Integer, out::Integer, batch::Integer, grid::Integer, modes::Integer,
                        σ = identity; initf = cglorot_uniform, initl = Flux.glorot_uniform, bias_fourier=true, bias_linear=true)

    # Initialize Fourier weight matrix (only with relevant modes)
    Wf = initf(in, out, modes)
    # Make sure filtering works
    @assert modes <= grid/2 + 1 "Specified modes exceed allowed maximum. The number of modes to filter must be smaller than N/2 + 1"
    # Pad the fourier weight matrix with additional zeros
    Wf = cat(Wf, zeros(size(Wf,1), size(Wf,2), floor(Int, grid/2 + 1) - modes); dims=3)

    # Initialize Linear weight matrix
    Wl = initl(batch, out, in)

    bf = bias_fourier
    bl = bias_linear

    return FourierLayer(Wf, Wl, bf, bl, σ)
end

Flux.@functor FourierLayer

# The actual layer that does stuff
function (a::FourierLayer)(x::AbstractArray)
    # Assign the parameters
    Wf, Wl, bf, bl, σ = a.weight_f, a.weight_l, a.bias_f, a.bias_l, a.σ

    # The linear path
    @tullio linear[batchsize, dim_out, dim_grid] := Wl[batchsize, dim_out, dim_in] *
                            x[batchsize, dim_in, dim_grid]

    # The convolution path
    # Do the Fourier transform (FFT) along the last axis of the input
    ft = rfft(x,3)

    # Multiply the weight matrix with the input using the Einstein convention
    @tullio 𝔉[batchsize, dim_out, dim_grid] := Wf[dim_in, dim_out, dim_grid] *
                ft[batchsize, dim_in, dim_grid] 
    # Do the inverse transform (WIP)
    fourier = irfft(𝔉, size(x,3), 3)

    # Return the activated sum
    return σ.(linear + fourier)
end

# Overload function to deal with higher-dimensional input arrays
#(a::FourierLayer)(x::AbstractArray) = reshape(a(reshape(x, size(x, 1), :)), :, size(x)[2:end]...)

# Print nicely
function Base.show(io::IO, l::FourierLayer)
    print(io, "FourierLayer with\nConvolution path: (", size(l.weight_f, 2), ", ",
            size(l.weight_f, 1), ", ", size(l.weight_f, 3))
    print(io, ")\n")
    print(io, "Linear path: (", size(l.weight_l, 2), ", ", size(l.weight_l, 1))
    print(io, ")\n")
    l.σ == identity || print(io, "Activation: ", l.σ)
end