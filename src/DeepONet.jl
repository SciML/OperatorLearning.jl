"""
`DeepONet(in, out, grid, modes, σ=identity, init=glorot_uniform)`
`DeepONet(Wf::AbstractArray, Wl::AbstractArray, [bias_f, bias_l, σ])`

Create a DeepONet architecture as proposed by Lu et al.
arXiv:1910.03193

The model works as follows:

x --- branch --
               |
                -⊠--u-
               |
y --- trunk ---

Where `x` represent the parameters of the PDE, discretely evaluated at its respective sensors,
and `y` are the probing locations for the operator to be trained.
`u` is the solution of the queried instance of the PDE, given by the specific choice of parameters.

Both inputs `x` and `y` are multiplied together via dot product Σᵢ bᵢⱼ tᵢₖ.

```julia
model = DeepONet()
```
"""
struct DeepONet
    branch_net::Flux.Chain
    trunk_net::Flux.Chain
    # Constructor for the DeepONet
    function DeepONet(
        branch_net::Flux.Chain,
        trunk_net::Flux.Chain)
        new(branch_net, trunk_net)
    end
end

# Declare the function that assigns Weights and biases to the layer
function DeepONet(architecture_branch::Tuple, architecture_trunk::Tuple,
                        act_branch = identity, act_trunk = identity;
                        init = Flux.glorot_uniform,
                        bias_branch=true, bias_trunk=true)

    # To construct the subnets we use the helper function in subnets.jl
    # Initialize the branch net
    branch_net = construct_subnet(architecture_branch, act_branch; bias=bias_branch)
    # Initialize the trunk net
    trunk_net = construct_subnet(architecture_trunk, act_trunk; bias=bias_trunk)

    return DeepONet(branch_net, trunk_net)
end

Flux.@functor DeepONet

# The actual layer that does stuff
# x needs to be at least a 2-dim array,
# since we need n inputs, evaluated at m locations
function (a::DeepONet)(x::AbstractMatrix, y::AbstractVecOrMat)
    # Assign the parameters
    branch, trunk = a.branch_net, a.trunk_net

    # Dot product needs a dim to contract
    # However, inputs are normally given with batching done in the same dim
    # so we need to adjust (i.e. transpose) one of the inputs,
    # and that's easiest on the matrix-type input
    return branch(x) * trunk(y)'
end

# Handling batches:
# We use basically the same function, but using NNlib's batched_mul instead of
# regular matrix-matrix multiplication
function (a::DeepONet)(x::AbstractArray, y::AbstractVecOrMat)
    # Assign the parameters
    branch, trunk = a.branch_net, a.trunk_net

    # Dot product needs a dim to contract
    # However, inputs are normally given with batching done in the same dim
    # so we need to adjust (i.e. transpose) one of the inputs,
    # and that's easiest on the matrix-type input
    return branch(x) ⊠ trunk(y)'
end

# Sensors stay the same and shouldn't be batched
(a::DeepONet)(x::AbstractArray, y::AbstractArray) = 
  throw(ArgumentError("Sensor locations fed to trunk net can't be batched."))

# Print nicely
function Base.show(io::IO, l::DeepONet)
    print(io, "DeepONet with\nbranch net: (",l.branch_net)
    print(io, ")\n")
    print(io, "Trunk net: (", l.trunk_net)
    print(io, ")\n")
end