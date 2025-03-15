using Documenter, DocumenterCitations, LearningAutomata

bib = CitationBibliography(
                           joinpath(@__DIR__, "src", "refs.bib");
                           style=:authoryear
                          )

makedocs(sitename = "LearningAutomata.jl",
         pages = [
                  "Home" => "index.md",
                  "Theory" => "theory.md",
                  "Guide" => [
                              "Learning in simple environments" => "simple.md",
                              "Agents.jl integration" => "agents.md"
                             ],
                  "API" => "api.md",
                  "References" => "references.md"
                 ],
         plugins = [bib])

#deploydocs(
#           repo = "github.com/hkauhanen/LearningAutomata.jl.git"
#          )
