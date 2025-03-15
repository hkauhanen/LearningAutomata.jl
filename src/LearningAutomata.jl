module LearningAutomata

using Agents
using Crayons
using LinearAlgebra
using StatsBase
using TernaryPlots

export AbstractLearner
export AbstractLinearLearner
export AbstractLRPLearner
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
