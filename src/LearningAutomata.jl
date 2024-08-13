module LearningAutomata

using LinearAlgebra
using StatsBase

export AbstractLearner
export LRPLearner

export punish!
export reward!
export simulate

include("utilities.jl")

include("AbstractTypes.jl")

include("LRPLearner.jl")

include("simulation.jl")

end
