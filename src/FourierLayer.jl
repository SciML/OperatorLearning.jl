"""
`FourierLayer(in, out, grid, modes, σ=identity, init=glorot_uniform)`
`FourierLayer(Wf::AbstractArray, Wl::AbstractArray, [bias_f, bias_l, σ])`

Create a Layer of the Fourier Neural Operator as proposed by Li et al.
arXiv: 2010.08895

The layer does a fourier transform on the grid dimension of the input array,
filters higher modes out by the weight matrix and transforms it to the
specified output dimension such that In x M x N -> Out x M x N.
The output though only contains the relevant Fourier modes with the rest padded to zero
in the last axis as a result of the filtering.

The input `x` should be a rank 3 tensor of shape
(num parameters (`in`) x num grid points (`grid`) x batch size (`batch`))
The output `y` will be a rank 3 tensor of shape
(`out` x num grid points (`grid`) x batch size (`batch`))

You can specify biases for the paths as you like, though the convolutional path is
originally not intended to perform an affine transformation.

# Examples
Say you're considering a 1D diffusion problem on a 64 point grid. The input is comprised
of the grid points as well as the IC at this point.
The data consists of 200 instances of the solution.
Beforehand we convert the two input channels into a higher-dimensional latent space with 128 nodes by using a regular `Dense` layer.
So the input takes the dimension `128 x 64 x 200`.
The output would be the diffused variable at a later time, which initially makes the output of the form `128 x 64 x 200` as well. Finally, we have to squeeze this high-dimensional ouptut into the one quantity of interest again by using a `Dense` layer.

We wish to only keep the first 16 modes of the input and work with the classic sigmoid function as activation.

So we would have:

```julia
model = FourierLayer(128, 128, 100, 16, σ)
```
"""
struct FourierLayer{F,Tc<:Complex{<:AbstractFloat},Tr<:AbstractFloat,Bf,Bl}
    # F: Activation, Tc/Tr: Complex/Real eltype
    Wf::AbstractArray{Tc,3}
    Wl::AbstractMatrix{Tr}
    grid::Int
    σ::F
    λ::Int
    bf::Bf
    bl::Bl
    # Constructor for the entire fourier layer
    function FourierLayer(
        Wf::AbstractArray{Tc,3}, Wl::AbstractMatrix{Tr},
        grid::Int, σ::F = identity,
        λ::Int = 12, bf = true, bl = true) where
        {F,Tc<:Complex{<:AbstractFloat},Tr<:AbstractFloat}

        # create the biases with one singleton dimension
        bf = Flux.create_bias(Wf, bf, 1, size(Wf,2), size(Wf,3))
        bl = Flux.create_bias(Wl, bl, 1, size(Wl,1), grid)
        new{F,Tc,Tr,typeof(bf),typeof(bl)}(Wf, Wl, grid, σ, λ, bf, bl)
    end
end

# Declare the function that assigns Weights and biases to the layer
# `in` and `out` refer to the dimensionality of the number of parameters
# `modes` specifies the number of modes not to be filtered out
# `grid` specifies the number of grid points in the data
function FourierLayer(in::Integer, out::Integer, grid::Integer, modes = 12,
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

    # Pass the bias bools
    bf = bias_fourier
    bl = bias_linear

    # Pass the modes for output
    λ = modes

    return FourierLayer(Wf, Wl, grid, σ, λ, bf, bl)
end

# Only train the weight array with non-zero modes
Flux.@functor FourierLayer

# Flux.trainable(a::FourierLayer) = (a.Wf[:,:,1:a.λ], a.Wl,
#                                 typeof(a.bf) != Flux.Zeros ? a.bf[:,:,1:a.λ] : nothing,
#                                 typeof(a.bl) != Flux.Zeros ? a.bl : nothing)

function Flux.trainable(a::FourierLayer)
    trainable_params = []
    push!(trainable_params,a.Wf[:,:,1:a.λ])
    push!(trainable_params,a.Wl)
    if a.bf!=false
        push!(trainable_params,typeof(a.bf) != Flux.Zeros ? a.bf[:,:,1:a.λ] : nothing)
    end
    if a.bl!=false
        push!(trainable_params,typeof(a.bl) != Flux.Zeros ? a.bl : nothing)
    end
    return trainable_params
end

# The actual layer that does stuff
function (a::FourierLayer)(x::AbstractArray)
    # Assign the parameters
    Wf, Wl, bf, bl, σ, = a.Wf, a.Wl, a.bf, a.bl, NNlib.fast_act(a.σ, x)

    # Do a permutation: DataLoader requires batch to be the last dim
    # for the rest, it's more convenient to have it in the first one
    xp = permutedims(x, [3,1,2])

    # The linear path
    # x -> Wl
    @ein linear[batch, out, grid] := Wl[out, in] * xp[batch, in, grid]
    linear .+ bl

    # The convolution path
    # x -> 𝔉 -> Wf -> i𝔉
    # Do the Fourier transform (FFT) along the grid dimension of the input and
    # Multiply the weight matrix with the input using batched multiplication
    # We need to permute the input to (channel,batch,grid), otherwise batching won't work
    @ein 𝔉[batch, out, grid] := Wf[in, out, grid] * rfft(xp, 3)[batch, in, grid]
    𝔉 .+ bf

    # Do the inverse transform
    # We need to permute back to match the shape of the linear path
    i𝔉 = irfft(𝔉, size(xp,3),3)

    # Return the activated sum
    return permutedims(σ.(linear + i𝔉), [2,3,1])
end

# Print nicely
function Base.show(io::IO, l::FourierLayer)
    print(io, "FourierLayer with\nConvolution path: (", size(l.Wf, 2), ", ",
            size(l.Wf, 1), ", ", size(l.Wf, 3))
    print(io, ")\n")
    print(io, "Linear path: (", size(l.Wl, 2), ", ", size(l.Wl, 1))
    print(io, ")\n")
    print(io, "Fourier modes: ", l.λ)
    print(io, "\n")
    l.σ == identity || print(io, "Activation: ", l.σ)
end
