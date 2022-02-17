"""
`FourierLayer(in, out, (grid), (modes), σ=identity, init=glorot_uniform)`
`FourierLayer(Wf::AbstractArray, Wl::AbstractArray, [bias_f, bias_l, σ])`

Create a Layer of the Fourier Neural Operator as proposed by Li et al.
arXiv: 2010.08895

The layer does a fourier transform on the grid dimension(s) of the input array,
filters higher modes out by the weight matrix and transforms it to the
specified output dimension such that In x M x N -> Out x M x N.
The output though only contains the relevant Fourier modes with the rest padded to zero
in the last axis as a result of the filtering.

The input `x` should be a tensor of shape
(num parameters (`in`) x num grid points (`grid`)[n] x batch size (`batch`))
The output `y` will be a tensor of shape
(`out` x num grid points (`grid`)[n] x batch size (`batch`))

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
struct FourierLayer{F,Tc<:Complex{<:AbstractFloat},N,Tr<:AbstractFloat,Bf,Bl}
    # F: Activation, Tc/Tr: Complex/Real eltype
    Wf::Array{Tc,N}
    Wl::Array{Tr,2}
    grid::Tuple
    σ::F
    λ::Tuple
    bf::Bf
    bl::Bl
    # Constructor for the entire fourier layer
    function FourierLayer(
        Wf::Array{Tc,N}, Wl::Array{Tr,2},
        grid::Tuple,σ::F = identity,
        λ::Tuple = (12), bf = true, bl = true) where
        {F,Tc<:Complex{<:AbstractFloat},N,Tr<:AbstractFloat}

        # create the biases with one singleton dimension for broadcasting
        bf = Flux.create_bias(Wf, bf, size(Wf,2), grid..., 1)
        bl = Flux.create_bias(Wl, bl, size(Wl,1), grid..., 1)
        new{F,Tc,N,Tr,typeof(bf),typeof(bl)}(Wf, Wl, grid, σ, λ, bf, bl)
    end
end

# Declare the function that assigns Weights and biases to the layer
# `in` and `out` refer to the dimensionality of the number of parameters
# `modes` specifies the number of modes not to be filtered out
# `grid` specifies the number of grid points in the data
function FourierLayer(in::Integer, out::Integer, grid::Tuple, modes::Tuple,
                        σ = identity; initf = cglorot_uniform, initl = Flux.glorot_uniform,
                        bias_fourier=true, bias_linear=true)

    # Number of grid dims and modes must match
    @assert length(modes) == length(grid) "Number of grid dimensions and number of Fourier modes do not match."
    # Make sure filtering works
    @assert modes <=  floor.(Int, grid./2 .+ 1) "Specified modes exceed allowed maximum.
    The number of modes to filter must be smaller than N/2 + 1"

    # Initialize Fourier weight tensor (only with relevant modes)
    Wf = initf(in, out, modes...)

    # Pad the fourier weight tensor with additional zeros up to n/2+1
    # padding tuple must be (0,numZeros1,0,numZeros2,...,0,numZerosN)
    # in and out dims are untouched, hence the two first ordered pairs of the tuple
    # are zero
    Wf = begin
        g = zeros(Int,4+2*length(modes))
        pad = grid .- modes
        for i ∈ eachindex(pad)
            g[4+2*i] = pad[i]
        end
        pad_zeros(Wf, tuple(g...))
    end

    # Initialize Linear weight matrix
    Wl = initl(out, in)

    # Pass the bias bools
    bf = bias_fourier
    bl = bias_linear

    # Pass the modes for output
    λ = modes

    return FourierLayer(Wf, Wl, grid, σ, λ, bf, bl)
end

# Compat for 1D FourierLayer
FourierLayer(in::Integer, out::Integer, grid::Int, modes::Int,
            σ = identity; initf = cglorot_uniform, initl = Flux.glorot_uniform,
            bias_fourier=true, bias_linear=true) =
            FourierLayer(in,out,(grid,),(modes,),σ;
            initf, initl, bias_fourier, bias_linear)

# Only train the weight array with non-zero modes
Flux.@functor FourierLayer
# The amount of grid dimensions is variable
function Flux.trainable(a::FourierLayer)
    (a.Wf[:,:,train_modes(a.λ)...],
    a.Wl,
    typeof(a.bf) != Flux.Zeros ? a.bf[:,:,train_modes(a.λ)...] : nothing,
    typeof(a.bl) != Flux.Zeros ? a.bl : nothing)
end

#= The actual layer that does the transformation
Since the dimensions of the input array varies with the grid,
this is implemented as a generated function =#
@generated function (a::FourierLayer)(x::AbstractArray{T,N}) where {T,N}
    #= Assign the parameters =#
    params = quote
        Wf = a.Wf
        Wl  = a.Wl
        bf = a.bf
        bl  = a.bl
        σ  = fast_act(a.σ, x)
    end

    #= Do a permutation
    DataLoader requires batch to be the last dim
    for the rest, it's more convenient to have it in the first one
    For this we need to generate the permutation tuple first
    experm evaluates to a tuple (N,1,2,...,N-1) =#

    #= The linear path
    x -> Wl
    As an argument to the einsum macro we need a list of named grid dimensions
    grids evaluates to a tuple of names of schema (grid_1, grid_2, ..., grid_N) =#
    grids = [Symbol("grid_$(i)") for i ∈ 1:N-2]
    linear_mul = :(@ein 𝔏[out, $(grids...), batch] := 
        Wl[out, in] * x[in, $(grids...), batch])
    linear_bias = :(𝔏 .+= bl)

    #= The convolution path
    x -> 𝔉 -> Wf -> i𝔉
    Do the Fourier transform (FFT) along the grid dimensions of the input and
    Multiply the weight tensor with the input using einsum
    To do the FFT we need to pass the grid dims to perform on
    fourier_dims evaluates to a tuple of Ints with range 3:N since the grid dims
    are sequential up to the last dim of the input =#
    fourier_dims = :([n for n ∈ 3:N])
    fourier_mul = :(@ein 𝔉[out, $(grids...), batch] := 
        Wf[in, out, $(grids...)] * fft(x, $(fourier_dims))[in, $(grids...), batch])
    fourier_bias = :(𝔉 .+= bf)

    #= Do the inverse transform
    We need to permute back to match the shape of the linear path =#
    fourier_inv = :(i𝔉 = ifft(𝔉, $(fourier_dims)))

    #= Undo the initial permutation
    experm_inv evaluates to a tuple (2,3,...,N,1) =#

    return Expr(
        :block,
        params,
        linear_mul,
        linear_bias,
        fourier_mul,
        fourier_bias,
        fourier_inv,
        :(return σ.(𝔏 + real(i𝔉)))
    )
end

# Print nicely
function Base.show(io::IO, l::FourierLayer)
    print(io, "FourierLayer with\nConvolution path: ", size(l.Wf))
    print(io, "\n")
    print(io, "Linear path: ", size(l.Wl))
    print(io, "\n")
    print(io, "Fourier modes: ", l.λ)
    print(io, "\n")
    l.σ == identity || print(io, "Activation: ", l.σ)
end