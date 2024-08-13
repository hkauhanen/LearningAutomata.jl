"""
    LRPLearner(n::Int, W::Vector{Float64}, γ::Float64) <: AbstractLearner

A linear reward-penalty learner.

A general linear reward-penalty learner with `n` actions, action probability
vector `W` and learning rate `γ`.

# References

Bush and Mosteller FIXME
"""
mutable struct LRPLearner <: AbstractLearner
  dim::Int
  W::Vector{Float64}
  gamma::Float64
end


"""
    LRPLearner(n::Int, γ::Float64)

Create an `LRPLearner` with `n` actions, each with identical initial action
probability, and learning rate `γ`.
"""
LRPLearner(n::Int, γ::Float64) = LRPLearner(n, ones(n) ./ n, γ)


"""
    reward!(x::LRPLearner, i::Int)

Reward the `i`th action of an `LRPLearner`.
"""
function reward!(x::LRPLearner, i::Int)
  x.W = ((1 - x.gamma) * LinearAlgebra.I + x.gamma * matrixunit(x.dim, i) * ones(x.dim, x.dim)) * x.W
end


"""
    punish!(x::LRPLearner, i::Int)

Punish the `i`th action of an `LRPLearner`.
"""
function punish!(x::LRPLearner, i::Int)
  x.W = ((1 - x.gamma) * LinearAlgebra.I + (x.gamma/(x.dim - 1)) * (ones(x.dim, x.dim) - matrixunit(x.dim, i)*ones(x.dim, x.dim))) * x.W
end
