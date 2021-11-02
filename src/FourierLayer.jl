"""
FourierLayer(in, out, batch, grid, modes, σ=identity, init=glorot_uniform)
FourierLayer(Wf::AbstractArray, Wl::AbstractArray, [bias_f, bias_l, σ])

Create a Layer of the Fourier Neural Operator as proposed by Zongyi et al.
arXiv: 2010.08895

The layer does a fourier transform on the last axis (the coeffs) of the input array,
filters higher modes out by the weight matrix and transforms the second axis to the
specified output dimension such that In x M x N -> Out x M x N.
The output though only contains the relevant Fourier modes with the rest padded to zero
in the last axis as a result of the filtering.

The input `x` should be a 3D tensor of shape
(num parameters (`in`) x batch size (`batch`) x num grid points (`grid`))
The output `y` will be a 3D tensor of shape
(`out` x batch size (`batch`) x num grid points (`grid`))

You can specify biases for the paths as you like, though the convolutional path is
originally not intended to perform an affine transformation.

# Examples
Say you're considering a 1D diffusion problem on a 64 point grid. The input is comprised
of the grid points as well as the IC at this point.
The data consists of 200 instances of the solution.
So the input takes the dimension `2 x 200 x 64`.
The output would be the diffused variable at a later time, which makes the output of the form
`2 x 200 x 64` as well.
"""
struct FourierLayer{F, Mf<:AbstractArray, Ml<:AbstractArray, Bf<:AbstractArray,
                Bl<:AbstractArray, fplan, ifplan,
                Modes<:Int}
    weight_f::Mf
    weight_l::Ml
    bias_f::Bf
    bias_l::Bl
    𝔉::fplan
    i𝔉::ifplan
    σ::F
    λ::Modes
    # Constructor for the entire fourier layer
    function FourierLayer(Wf::Mf, Wl::Ml, bf::Bf, bl::Bl, 𝔉::fplan, i𝔉::ifplan,
        σ::F = identity, λ::Modes = 12) where {Mf<:AbstractArray, Ml<:AbstractArray,
        Bf<:AbstractArray, Bl<:AbstractArray, fplan,
        ifplan, F, Modes<:Int}
        new{F,Mf,Ml,Bf,Bl,fplan,ifplan,Modes}(Wf, Wl, bf, bl, 𝔉, i𝔉, σ, λ)
    end
end

# Declare the function that assigns Weights and biases to the layer
# `in` and `out` refer to the dimensionality of the number of parameters
# `modes` specifies the number of modes not to be filtered out
# `grid` specifies the number of grid points in the data
function FourierLayer(in::Integer, out::Integer, batch::Integer, grid::Integer, modes = 12,
                        σ = identity; initf = cglorot_uniform, initl = Flux.glorot_uniform,
                        bias_fourier=true, bias_linear=true)

    # Initialize Fourier weight matrix (only with relevant modes)
    Wf = initf(in, out, modes)
    # Make sure filtering works
    @assert modes <= floor(Int, grid/2 + 1) "Specified modes exceed allowed maximum. 
    The number of modes to filter must be smaller than N/2 + 1"
    # Pad the fourier weight matrix with additional zeros
    Wf = pad_zeros(Wf, (0, floor(Int, grid/2 + 1) - modes), dims=3)

    # Initialize Linear weight matrix
    Wl = initl(out, in)

    bf = Flux.create_bias(Wf, bias_fourier, out, 1, floor(Int, grid/2 + 1))
    bl = Flux.create_bias(Wl, bias_linear, out, 1, grid)

    # Pass the modes for output
    λ = modes

    # Create linear operators for the FFT and IFFT for efficiency
    # So that it has to be only pre-allocated once
    # First, an ugly workaround: FFTW.jl passes keywords that cuFFT complains about when the
    # constructor is wrapped with |> gpu. Instead, you have to pass a CuArray as input to plan_rfft
    # Ugh.
    template𝔉 = Flux.use_cuda[] != Nothing ? Array{Float32}(undef,in,batch,grid) :
                    CuArray{Float32}(undef,in,batch,grid)
    templatei𝔉 = Flux.use_cuda[] != Nothing ? Array{Complex{Float32}}(undef,out,batch,floor(Int, grid/2 + 1)) :
                    CuArray{Complex{Float32}}(undef,out,batch,floor(Int, grid/2 + 1))

    𝔉 = plan_rfft(template𝔉,3)
    i𝔉 = plan_irfft(templatei𝔉,grid, 3)

    return FourierLayer(Wf, Wl, bf, bl, 𝔉, i𝔉, σ, λ)
end

Flux.@functor FourierLayer

# The actual layer that does stuff
function (a::FourierLayer)(x::AbstractArray)
    # Assign the parameters
    Wf, Wl, bf, bl, σ, 𝔉, i𝔉 = a.weight_f, a.weight_l, a.bias_f, a.bias_l, a.σ, a.𝔉, a.i𝔉

    # The linear path
    # x -> Wl
    linear = Wl ⊠ x .+ bl
    #@ein linear[dim_out, batchsize, dim_grid] := Wl[dim_out, dim_in] *
    #                        x[dim_in, batchsize, dim_grid]
    #linear += bl

    # The convolution path
    # x -> 𝔉 -> Wf -> i𝔉
    # Do the Fourier transform (FFT) along the last axis of the input
    fourier = 𝔉 * x

    # Multiply the weight matrix with the input using the Einstein convention
    fourier = Wf ⊠ fourier .+ bf
    #@ein fourier[dim_out, batchsize, dim_grid] := Wf[dim_in, dim_out, dim_grid] *
    #            fourier[dim_in, batchsize, dim_grid]
    #fourier += bf
    # Do the inverse transform
    fourier = i𝔉 * fourier

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
    print(io, "Linear path: (", size(l.weight_l, 2), ", ", size(l.weight_l, 1), ", ",
            size(l.weight_l, 3))
    print(io, ")\n")
    print(io, "Fourier modes: ", l.λ)
    print(io, "\n")
    l.σ == identity || print(io, "Activation: ", l.σ)
end