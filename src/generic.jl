"""
    act(x::AbstractLearner)

Make a learner act.

Draws an action according to the learner's current action probability vector.
"""
function act(x::AbstractLearner)
    StatsBase.sample(1:x.n, StatsBase.Weights(x.W))
end


"""
    reward!(x::AbstractLinearLearner, i::Int)

Reward the `i`th action of an `AbstractLinearLearner`.
"""
function reward!(x::AbstractLinearLearner, i::Int)
    x.W = x.R[i] * x.W
end


"""
    punish!(x::AbstractLinearLearner, i::Int)

Punish the `i`th action of an `AbstractLinearLearner`.
"""
function punish!(x::AbstractLinearLearner, i::Int)
    x.W = x.P[i] * x.W
end
