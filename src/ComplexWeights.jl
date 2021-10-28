"""
cglorot_uniform([rng=GLOBAL_RNG], dims...)

A modification of the `glorot_uniform` function provided by `Flux` to accommodate Complex numbers.
This is necessary since the parameters of the global convolution operator in the Fourier Layer generally has complex weights.
"""
cglorot_uniform(rng::AbstractRNG, dims...) = (rand(rng, ComplexF32, dims...) .- 0.5f0) .* sqrt(24.0f0 / sum(nfan(dims...)))
cglorot_uniform(dims...) = cglorot_uniform(Random.GLOBAL_RNG, dims...)
cglorot_uniform(rng::AbstractRNG) = (dims...) -> cglorot_uniform(rng, dims...)

"""
cglorot_normal([rng=GLOBAL_RNG], dims...)

A modification of the `glorot_normal` function provided by `Flux` to accommodate Complex numbers.
This is necessary since the parameters of the global convolution operator in the Fourier Layer generally has complex weights.
"""
cglorot_normal(rng::AbstractRNG, dims...) = randn(rng, ComplexF32, dims...) .* sqrt(2.0f0 / sum(nfan(dims...)))
cglorot_normal(dims...) = cglorot_normal(Random.GLOBAL_RNG, dims...)
cglorot_normal(rng::AbstractRNG) = (dims...) -> cglorot_normal(rng, dims...)