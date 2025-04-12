"""
    AbstractLearner

An abstract learner; the supertype of all learner composite types.
"""
abstract type AbstractLearner end

"""
    AbstractLinearLearner <: AbstractLearner

An abstract type that subsumes all linear learners.

A learner is linear if each update and reward operator is a linear
transformation of the learner's knowledge state.
"""
abstract type AbstractLinearLearner <: AbstractLearner end

"""
    AbstractLRPLearner <: AbstractLinearLearner

Abstract type that subsumes all linear reward--penalty learners.
"""
abstract type AbstractLRPLearner <: AbstractLinearLearner end

"""
    AbstractLRILearner <: AbstractLinearLearner

Abstract type that subsumes all linear reward--inaction learners.
"""
abstract type AbstractLRILearner <: AbstractLinearLearner end

"""
    AbstractLearningEnvironment

The abstract supertype of all learning environments.
"""
abstract type AbstractLearningEnvironment end

"""
    AbstractSRE <: AbstractLearningEnvironment

Abstract type that subsumes stationary (constant) learning environments.
"""
abstract type AbstractSRE <: AbstractLearningEnvironment end


"""
    punishes(x::AbstractSRE, i::Int)

Checks whether the stationary random environment `x` punishes action `i`.

Returns `true` with probability `x.c[i]` (the penalty probability of action `i`
in the environment) and `false` with the remaining probability mass.
"""
function punishes(x::AbstractSRE, i::Int)
    rand() < x.c[i] ? true : false
end


"""
    limit(x::AbstractSRE)

Return the limit of the expected weight vector in a stationary random environment,
if this limit exists uniquely.
"""
function limit(x::AbstractSRE)
    zero_indices = findall(==(0), x.c)

    if length(zero_indices) == 0
        inverse_penalties = x.c .^ -1
        return inverse_penalties/sum(inverse_penalties)
    elseif length(zero_indices) == 1
        out = zeros(length(x.c))
        out[zero_indices] .= 1.0
        return out
    else
        error("No unique limit point for mean dynamic")
    end
end


"""
    limit_rand(x::AbstractSRE, σ::Float64)

Return a random vector from the vicinity of the limit of the expected
weight vector.
"""
function limit_rand(x::AbstractSRE, σ::Float64)
    dir = Distributions.Dirichlet(σ^-1 .* x.c .^-1)
    vec(rand(dir))
end


"""
    limit_rand(x::AbstractSRE, σ::Float64, n::Int)

Return `n` random vectors from the vicinity of the limit of the expected
weight vector.
"""
function limit_rand(x::AbstractSRE, σ::Float64, n::Int)
    dir = Distributions.Dirichlet(σ^-1 .* x.c .^-1)
    LinearAlgebra.transpose(rand(dir, n))
end



"""
    limit_pdf(x::AbstractSRE, σ::Float64, y::Vector{Float64})

Return the probability density of a Dirichlet distribution centred at
the expectation limit of a stationary random environment, at point `y`.
"""
function limit_pdf(x::AbstractSRE, σ::Float64, y::Vector{Float64})
    dir = Distributions.Dirichlet(σ .* x.c .^-1)
    pdf(dir, y)
end


function normalize(x::Vector{Float64})
    x ./ sum(x)
end


"""
    reward!(x::AbstractLinearLearner, i::Int)

Reward the `i`th action of an `AbstractLinearLearner`.
"""
function reward!(x::AbstractLinearLearner, i::Int)
    x.W = x.R[i] * x.W |> normalize
end


"""
    punish!(x::AbstractLinearLearner, i::Int)

Punish the `i`th action of an `AbstractLinearLearner`.
"""
function punish!(x::AbstractLinearLearner, i::Int)
    x.W = x.P[i] * x.W |> normalize
end


"""
    act(x::AbstractLearner)

Choose an action to perform.

Returns the index of the action.
"""
function act(x::AbstractLearner)
    StatsBase.sample(1:x.n, Weights(x.W))
end


"""
    act(x::AbstractLearner, n::Int)

Act `n` times. Returns the indices of `n` actions drawn from the learner's
current action weight distribution.
"""
function act(x::AbstractLearner, n::Int)
    #[act(x) for i in 1:n]
    StatsBase.sample(1:x.n, Weights(x.W), n)
end


"""
    reset!(x::AbstractLearner)

Reset a learner's weight vector to the maximum-entropy state ``(1/n, \\ldots , 1/n)``,
where ``n``is the number of actions.
"""
function reset!(x::AbstractLearner)
    x.W = ones(x.n)/x.n
end


"""
    interact!(x::AbstractLearner, y::AbstractSRE)

Undergo an interaction between a learner and a stationary random environment.
"""
function interact!(x::AbstractLearner, y::AbstractSRE)
    a = act(x)
    punishes(y, a) ? punish!(x, a) : reward!(x, a)
    #=
    if a == 1
        punishes(y, 1) ? punish!(x, 1) : reward!(x, 1)
    elseif a == 2
        punishes(y, 2) ? reward!(x, 1) : punish!(x, 1)
    end
    =#
    #=
    if a == 1
        punishes(y, 1) ? reward!(x, 2) : punish!(x, 2)
    elseif a == 2
        punishes(y, 2) ? punish!(x, 2) : reward!(x, 2)
    end
    =#
end


"""
    interact!(x::AbstractLearner, y::AbstractSRE, n::Int;
              collect_history = true)

Undergo `n` interactions between a learner and a stationary random environment.
"""
function interact!(x::AbstractLearner, y::AbstractSRE, n::Int;
                   collect_history = true)
    history = [interact!(x, y) for t in 1:n]
    
    if collect_history
        return hcat(history...)'
    else
        return history[end]
    end
end


"""
    interact!(x::AbstractLearner, y::AbstractLearner; reciprocal = true)

Make two learners `x` and `y` interact.

Interaction is reciprocal, i.e. both learners update their state,
if `reciprocal = true`. Else only `y` updates its state.
"""
function interact!(x::AbstractLearner, y::AbstractLearner; reciprocal = true)
    # Action taken by x
    a = act(x)

    # Action taken by y
    b = act(y)

    # Outcome for y
    rand() < x.A[b,a] ? punish!(y, b) : reward!(y, b)

    # Outcome for x
    if reciprocal
        rand() < y.A[a,b] ? punish!(x, a) : reward!(x, a)
    end

    # Return
    return [x.W, y.W]
end


"""
    interact!(x::AbstractLearner, y::AbstractLearner, n::Int;
              reciprocal = true, collect_history = true)

Make two learners `x` and `y` interact for `n` interactions.

Interaction is reciprocal, i.e. both learners update their state,
if `reciprocal = true`. Else only `y` updates its state. If
`collect_history = true`, the entire learning trajectories of both
learners are returned as a two-component vector of matrices; else,
only the final states are returned.
"""
function interact!(x::AbstractLearner, y::AbstractLearner, n::Int;
                   reciprocal = true,
                   collect_history = true)
    history = [interact!(x, y; reciprocal=reciprocal) for t in 1:n]

    if collect_history
        xhist = vcat(hcat(history...)'[:,1]...)
        yhist = vcat(hcat(history...)'[:,2]...)
        return [xhist, yhist]
    else
        return history[end]
    end
end


# TRIGGER revive_operators! ON REASSIGNMENT OF INTERNAL FIELDS
function Base.setproperty!(x::AbstractLinearLearner, s::Symbol, f)
    if s === :W
        setfield!(x, s, f)
    else
        setfield!(x, s, f)
        revive_operators!(x)
        #show(x)
    end
end


