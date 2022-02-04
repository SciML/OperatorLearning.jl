using OperatorLearning
using Documenter, DocumenterTools

DocMeta.setdocmeta!(OperatorLearning, :DocTestSetup, :(using OperatorLearning); recursive=true)

makedocs(;
    modules=[OperatorLearning],
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
        "Frequently Asked Questions" => "faq.md",
    ],
)

deploydocs(;
    repo="github.com/pzimbrod/OperatorLearning.jl",
    #devbranch="main",
)
