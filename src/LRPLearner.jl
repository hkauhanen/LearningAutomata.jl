"""
    LRPLearner(n::Int, a::Float64, b::Float64, W::Vector{Float64}, R::Vector{Matrix{Float64}}, P::Vector{Matrix{Float64}}) <: AbstractLinearLearner

A linear reward-penalty learner.

A general linear reward-penalty learner with `n` actions, learning rate `a`
for rewards, learning rate `b` for punishments, action probability
vector `W`, and vectors of reward and penalty operators `R` and `P`.

# References

Bush and Mosteller FIXME
"""
mutable struct LRPLearner <: AbstractLinearLearner
  n::Int
  a::Float64
  b::Float64
  W::Vector{Float64}
  R::Vector{Matrix{Float64}}
  P::Vector{Matrix{Float64}}
end


# custom pretty-printing for LRPLearners
Base.show(io::IO, z::LRPLearner) = print(io, "Linear reward-penalty learner (LRPLearner) with ", z.n, " actions\n\nLearning rate for reward: ", z.a, "\nLearning rate for penalty: ", z.b, "\n\nCurrent action probability vector: ", z.W)


"""
    LRPLearner(n::Int, γ::Float64)

Create an `LRPLearner` with `n` actions, each with identical initial action
probability, and common learning rate `γ`.
"""
LRPLearner(n::Int, γ::Float64) = LRPLearner(n, γ, γ)


"""
    LRPLearner(n::Int, a::Float64, b::Float64)

Create an `LRPLearner` with `n` actions, each with identical initial action
probability, and learning rates `a` and `b`.
"""
LRPLearner(n::Int, a::Float64, b::Float64) = LRPLearner(n, ones(n) ./ n, a, b)


"""
    LRPLearner(n::Int, W::Vector{Float64}, a::Float64, b::Float64)

Create an `LRPLearner` with `n` actions, initial action probability vector `W`,
and learning rates `a` and `b`.
"""
function LRPLearner(n::Int, W::Vector{Float64}, a::Float64, b::Float64)
    R = Vector{Matrix{Float64}}(undef, 0)
    P = Vector{Matrix{Float64}}(undef, 0)
    
    # construct reward and penalty matrices
    for i in 1:n
        push!(R, (1 - a) * LinearAlgebra.I + a * matrixunit(n, i) * ones(n, n))
        push!(P, (1 - b) * LinearAlgebra.I + (b/(n - 1)) * (ones(n, n) - matrixunit(n, i)* ones(n, n)))
    end
    
    return LRPLearner(n, a, b, W, R, P)
end


#=
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

=#
