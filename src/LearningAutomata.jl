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
export reconfigure!
export revive_operators!
export interact!
export simulate!

include("utilities.jl")

include("AbstractTypes.jl")

include("generic.jl")

include("LRPLearner.jl")

include("simulation.jl")

end
