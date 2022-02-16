@generated function (a::FourierLayer)(x::AbstractArray{T,N}) where {T,N}
    #= Assign the parameters
    Nothing out of the ordinary here =#
    params = :(Wf, Wl, bf, bl, σ = a.Wf, a.Wl, a.bf, a.bl, NNlib.fast_act(a.σ, x))

    #= Do a permutation
    DataLoader requires batch to be the last dim
    for the rest, it's more convenient to have it in the first one
    For this we need to generate the permutation tuple first
    experm evaluates to a tuple (N,1,2,...,N-1) =#
    experm = :(tuple(N,$:([k for k = 1:N-1]...)))
    permute = :(xp = permutedims(x, $experm))

    #= The linear path
    x -> Wl
    As an argument to the einsum macro we need a list of named grid dimensions
    grids evaluates to a tuple of names of schema (grid_1, grid_2, ..., grid_N) =#
    grids = Expr(:tuple, [Symbol("grid_$(i)") for i ∈ 1:N-2]...)
    linear_mul = :(@ein linear[batch, out, $grids...] := Wl[out, in] * xp[batch, in, $grids...])
    linear_bias = :(linear .+ bl)

    #= The convolution path
    x -> 𝔉 -> Wf -> i𝔉
    Do the Fourier transform (FFT) along the grid dimensions of the input and
    Multiply the weight tensor with the input using einsum
    To do the FFT we need to pass the grid dims to perform on
    fourier_dims evaluates to a tuple of Ints with range 3:N since the grid dims
    are sequential up to the last dim of the input =#
    fourier_dims = :(tuple($:([n for n ∈ 3:N])))
    fourier_mul = :(@ein 𝔉[batch, out, $grids...] := Wf[in, out, $grids...] * fft(xp, $fourier_dims...)[batch, in, $grids...])
    fourier_bias = :(𝔉 .+ bf)

    #= Do the inverse transform
    We need to permute back to match the shape of the linear path =#
    fourier_inv = :(i𝔉 = ifft(𝔉, $fourier_dims...))

    #= Undo the initial permutation
    experm_inv evaluates to a tuple (2,3,...,N,1) =#
    experm_inv = :(tuple($:([k for k = 2:N]...),1))

    return Expr(
        :block,
        params,
        permute,
        linear_mul,
        linear_bias,
        fourier_mul,
        fourier_bias,
        fourier_inv,
        :(return permutedims(σ.(linear + i𝔉), $experm_inv))
    )
end