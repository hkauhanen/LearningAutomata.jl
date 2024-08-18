module LearningAutomata

using LinearAlgebra
using StatsBase

export AbstractLearner
export AbstractLinearLearner
export LRPLearner

export act
export punish!
export reward!
export revive_operators!
export interact!
export simulate!

include("utilities.jl")

include("AbstractTypes.jl")

include("generic.jl")

include("LRPLearner.jl")

include("simulation.jl")

end
