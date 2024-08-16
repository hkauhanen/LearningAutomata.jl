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


"""
    interact!(x::AbstractLearner, y::AbstractLearner)

Make two learners `x` and `y` interact.

Interaction is symmetric, i.e. both learners update their state,
if `symmetric = true`. Else it is asymmetric, and only `y`
updates its state.
"""
function interact!(x::AbstractLearner, y::AbstractLearner; symmetric = true)
    # Action taken by x
    a = StatsBase.sample(1:x.n, Weights(x.W))

    # Action taken by y
    b = StatsBase.sample(1:y.n, Weights(y.W))

    # Outcome for y
    rand() < x.A[b,a] ? punish!(y, b) : reward!(y, b)

    # Outcome for x
    if symmetric
        rand() < y.A[a,b] ? punish!(x, a) : reward!(x, a)
    end
end
