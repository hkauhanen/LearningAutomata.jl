"""
    simulate!(x::AbstractLearner, iter::Int, c::Vector{Float64})

Simulate a learner for `iter` iterations in a stationary random environment
constituted by penalty probability vector `c`.

Returns the trajectory as a ``m \\times n`` matrix, where ``m`` is the trajectory
length (number of simulation iterations) and ``n`` is the learner's dimensionality
(number of actions). This can be redirected e.g. to `plot` in order to visualize
the learning trajectory.
"""
function simulate!(x::AbstractLearner, iter::Int, c::Vector{Float64})
    inputs = StatsBase.sample(1:x.n, Weights(c), iter)
    history = zeros(x.n, iter)

    for t in 1:iter
        g = StatsBase.sample(1:x.n, Weights(x.W))

        g == inputs[t] ? punish!(x, g) : reward!(x, g)

        history[:, t] = x.W
    end

    return transpose(history)
end


"""
    interact!(x::AbstractLearner, y::AbstractLearner)

Make two learners `x` and `y` interact.

By design, interaction is asymmetric: `x` carries out an action, `y` observes
this action and updates its state.
"""
function interact!(x::AbstractLearner, y::AbstractLearner)
    #FIXME
end
