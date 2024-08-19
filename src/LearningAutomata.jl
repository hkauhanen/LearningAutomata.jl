module LearningAutomata

using Crayons
using LinearAlgebra
using StatsBase

export AbstractLearner
export AbstractLinearLearner
export LRPLearner

export get_probs
export punish!
export reward!
export interact!
export simulate!

include("utilities.jl")

include("AbstractTypes.jl")

include("LRPLearner.jl")

include("simulation.jl")

end
