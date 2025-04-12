"""
    simulate!(x::AbstractLearner, y::AbstractSRE, n::Int;
              collect_history = true)

Simulate a learner for `n` iterations in a stationary random environment
constituted by penalty probability vector `c`.

If `collect_history = true`, 
returns the trajectory as a ``m \\times k`` matrix, where ``m`` is the trajectory
length (number of simulation iterations) and ``k`` is the learner's dimensionality
(number of actions). This can be redirected e.g. to `plot` in order to visualize
the learning trajectory.

If `collect_history = false`, only the final state is returned.
"""
function simulate!(x::AbstractLearner, y::AbstractSRE, n::Int;
                   collect_history = true)
    history = zeros(x.n, n)

    for t in 1:n
        g = act(x)

        is_punished(y, g) ? punish!(x, g) : reward!(x, g)

        if collect_history
          history[:, t] = x.W
        end
    end

    if collect_history
      return transpose(history)
    else
      return x.W
    end
end

