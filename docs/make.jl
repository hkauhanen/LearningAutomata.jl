push!(LOAD_PATH,"../src/")


using Documenter, DocumenterCitations, LearningAutomata

bib = CitationBibliography(
                           joinpath(@__DIR__, "src", "refs.bib");
                           style=:authoryear
                          )

makedocs(sitename = "LearningAutomata.jl",
         pages = [
                  "Home" => "index.md",
                  "Quickstart" => "quickstart.md",
                  "Theory" => "theory.md",
                  "Guide" => [
                              "Learning in simple environments" => "simple.md",
                              "Agents.jl integration" => "agents.md",
                              "Working with mean dynamics" => "meandynamics.md",
                              "Advanced visualization" => "advis.md"
                             ],
                  "Developer notes" => "developing.md",
                  "API" => "api.md",
                  "References" => "references.md"
                 ],
         plugins = [bib],
         remotes = nothing)

#deploydocs(
#           repo = "github.com/hkauhanen/LearningAutomata.jl.git"
#          )
