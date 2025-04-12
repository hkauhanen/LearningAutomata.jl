module LearningAutomata

using Crayons
using Distributions
using LinearAlgebra
using StatsBase
using TernaryPlots

export AbstractLearner
export AbstractLinearLearner
export AbstractLRPLearner
export AbstractLRILearner

export AbstractLearningEnvironment
export AbstractSRE

export LRPLearner
export LRILearner

export SRE

export act
export reset!
export punish!
export reward!
export interact!
export punishes
export limit
export limit_rand
export limit_pdf

include("utilities.jl")

include("AbstractTypes.jl")

include("LRPLearner.jl")

include("LRILearner.jl")

include("SRE.jl")

include("simulation.jl")

end
