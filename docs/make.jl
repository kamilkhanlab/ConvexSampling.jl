push!(LOAD_PATH,"C:/Users/maha-/Documents/GitHub/convex-sampling")

using Documenter, ConvexSampling

makedocs(sitename="My Documentation",
        format = Documenter.HTML(prettyurls = false),
        pages = Any["Introduction" => "index.md",
                    "Exported Functions" => "functions.md"]
)
