# FAQ

## What's the status of this package?

The package as a whole is still under heavy development. However, the layers and models released work well and can be used. See the examples for usage.

## What do I need to train an operator mapping? What are the input data?

Currently, you need solved instances of the system you're trying to approximate the solution operator of.

That is, you'll need to gather data (probably using numerical simulations) that include the solution vector, the grid and the parameters of the PDE (system).

However, future work includes implementing physics-informed operator approximations which have been shown to be able to lighten the amount of training data needed or even alleviate it altogether (see e.g. [[1](https://doi.org/10.1126/sciadv.abi8605)] or [[2](http://arxiv.org/abs/2111.03794)]).

## What about hardware and distributed computing?

Just like `Flux.jl`, this package runs nicely on GPUs using `CUDA.jl`. you can simply pipe your data and function calls using `|> gpu` using the macro that Flux provides. For usage, see the Burgers equation example. Running on multiple GPUs has however not been tested yet.
