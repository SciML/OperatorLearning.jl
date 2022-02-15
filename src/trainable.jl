"""
A helper function to create ranges of the trainable Fourier modes.
Those can be parsed and evaluated in order to specify the trainable parameters by Flux

# Input
The FourierLayer where the trainable modes are required

# Output
A vector that contains a comma-separated list of ranges for each Fourier dimension:

```julia
julia> OperatorLearning.train_modes((2,5,12,8))
4-element Vector{UnitRange}:
 1:2
 1:5
 1:12
 1:8
```
"""
function train_modes(λ::Tuple)
    modes = Array{UnitRange}(undef,length(λ))
    for (index,numModes) in enumerate(λ)
        modes[index] = 1:numModes
    end
    return modes
end
