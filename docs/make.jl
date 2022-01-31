using NeuralOperator
using Documenter

DocMeta.setdocmeta!(NeuralOperator, :DocTestSetup, :(using NeuralOperator); recursive=true)

makedocs(;
    modules=[NeuralOperator],
    authors="Patrick Zimbrod <patrick.zimbrod@gmail.com> and contributors",
    repo="https://github.com/pzimbrod/OperatorLearning.jl/blob/{commit}{path}#{line}",
    sitename="OperatorLearning.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://pzimbrod.github.io/OperatorLearning.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Module Reference" => "reference.md",
    ],
)

deploydocs(;
    repo="github.com/pzimbrod/OperatorLearning.jl",
    versions = ["stable" => "v^", "v#.#", devurl => devurl],
    #devbranch="main",
)
