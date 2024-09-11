using Documenter, LearningAutomata

makedocs(sitename = "LearningAutomata.jl",
         pages = [
                  "Home" => "index.md",
                  "Guide" => "guide.md",
                  "API" => "api.md"
                 ])

deploydocs(
           repo = "github.com/hkauhanen/LearningAutomata.jl.git"
          )
