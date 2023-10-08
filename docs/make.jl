push!(LOAD_PATH,"C:/Users/maha-/Documents/GitHub/convex-sampling")

using Documenter, ConvexSampling

makedocs(sitename="ConvexSampling",
        format = Documenter.HTML(prettyurls = false),
        pages = Any["Introduction" => "index.md",
                    "Method Outline" => "methodOutline.md",
                    "Exported Functions" => "functions.md"]
)
