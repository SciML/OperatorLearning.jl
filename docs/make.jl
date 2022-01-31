using NeuralOperator
using Documenter

DocMeta.setdocmeta!(NeuralOperator, :DocTestSetup, :(using NeuralOperator); recursive=true)

makedocs(;
    modules=[NeuralOperator],
    authors="Patrick Zimbrod <patrick.zimbrod@gmail.com> and contributors",
    repo="https://github.com/pzimbrod/NeuralOperator.jl/blob/{commit}{path}#{line}",
    sitename="NeuralOperator.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://pzimbrod.github.io/NeuralOperator.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/pzimbrod/NeuralOperator.jl",
    #devbranch="main",
)
