# TYPE

"""
    LRPLearner(n::Int,
               a::Vector{Float64},
               b::Vector{Float64}, 
               c::Vector{Float64},
               W::Vector{Float64},
               A::Matrix{Float64},
               R::Vector{Matrix{Float64}}, 
               P::Vector{Matrix{Float64}}) <: AbstractLinearLearner

A linear reward-penalty learner.

A general linear reward-penalty learner with `n` actions, learning rates `a`
for rewards, learning rates `b` for punishments, action costs `c`, action probability
vector `W`, vectors of reward and penalty operators `R` and `P`, and
advantage matrix `A`.

# References

Bush and Mosteller FIXME
"""
mutable struct LRPLearner <: AbstractLinearLearner
    n::Int
    a::Vector{Float64}
    b::Vector{Float64}
    c::Vector{Float64}
    W::Vector{Float64}
    A::Matrix{Float64}
    R::Vector{Matrix{Float64}}
    P::Vector{Matrix{Float64}}
end


# PRETTY-PRINTING

Base.show(io::IO, z::LRPLearner) = print(io, "Linear reward-penalty learner (LRPLearner) with ", z.n, " actions\n\nLearning rates for reward: ", z.a, "\nLearning rates for penalty: ", z.b, "\n\nAction costs: ", z.c, "\n\nAdvantage matrix: ", z.A, "\n\nCurrent action probability vector: ", z.W)


# CONSTRUCTORS

"""
    LRPLearner(n::Int,
               a::Vector{Float64};
               b::Vector{Float64} = a,
               c::Vector{Float64} = zeros(n),
               W::Vector{Float64} = ones(n) ./ n,
               A::Matrix{Float64} = zeros(n, n))

Create an `LRPLearner` with `n` actions and learning rate vector `a`.
"""
function LRPLearner(n::Int,
           a::Vector{Float64};
           b::Vector{Float64} = a,
           c::Vector{Float64} = zeros(n),
           W::Vector{Float64} = ones(n) ./ n,
           A::Matrix{Float64} = zeros(n ,n))
    R = Vector{Matrix{Float64}}(undef, n)
    learner = LRPLearner(n, a, b, c, W, A, R, R)
    revive_operators!(learner)
    return learner
end


"""
    LRPLearner(n::Int,
               a::Float64;
               b::Vector{Float64} = a .* ones(n),
               c::Vector{Float64} = zeros(n),
               W::Vector{Float64} = ones(n) ./ n,
               A::Matrix{Float64} = zeros(n, n))

Create an `LRPLearner` with `n` actions and learning rate `a`.
"""
function LRPLearner(n::Int,
           a::Float64;
           b::Vector{Float64} = a .* ones(n),
           c::Vector{Float64} = zeros(n),
           W::Vector{Float64} = ones(n) ./ n,
           A::Matrix{Float64} = zeros(n ,n))
    R = Vector{Matrix{Float64}}(undef, n)
    learner = LRPLearner(n, a .* ones(n), b, c, W, A, R, R)
    revive_operators!(learner)
    return learner
end


# UTILITY FUNCTIONS

"""
    revive_operators!(x::LRPLearner)

Recompute the reward and penalty operators of an `LRPLearner`.

Call this if you change any of the following internal fields of an `LRPLearner`:
`a`, `b`, `c`.
"""
function revive_operators!(x::LRPLearner)
    for i in 1:x.n
        x.R[i] = (1 - x.a[i] - x.c[i]) * LinearAlgebra.I + x.a[i] * matrixunit(x.n, i) * ones(x.n, x.n) + (x.c[i]/(x.n - 1)) * (ones(x.n, x.n) - matrixunit(x.n, i) * ones(x.n, x.n))
        x.P[i] = (1 - x.b[i] - x.c[i]) * LinearAlgebra.I + ((x.b[i] + x.c[i])/(x.n - 1)) * (ones(x.n, x.n) - matrixunit(x.n, i)* ones(x.n, x.n))
    end
end


#=

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
LRPLearner(n::Int, W::Vector{Float64}, a::Float64, b::Float64) = LRPLearner(n, W, repeat([a], outer=n), repeat([b], outer=n), zeros(n))


"""
    LRPLearner(n::Int, W::Vector{Float64}, a::Vector{Float64}, b::Vector{Float64}, c::Vector{Float64})

Create an `LRPLearner` with `n` actions, initial action probability vector `W`,
learning rate vectors `a` and `b`, and action cost vector `c`.
"""
function LRPLearner(n::Int, W::Vector{Float64}, a::Vector{Float64}, b::Vector{Float64}, c::Vector{Float64})
    R = Vector{Matrix{Float64}}(undef, 0)
    P = Vector{Matrix{Float64}}(undef, 0)

    # construct reward and penalty matrices
    for i in 1:n
        push!(R, (1 - a[i] - c[i]) * LinearAlgebra.I + a[i] * matrixunit(n, i) * ones(n, n) + (c[i]/(n - 1)) * (ones(n, n) - matrixunit(n, i) * ones(n, n)))
        push!(P, (1 - b[i] - c[i]) * LinearAlgebra.I + ((b[i] + c[i])/(n - 1)) * (ones(n, n) - matrixunit(n, i)* ones(n, n)))
    end

    return LRPLearner(n, a, b, c, W, R, P, zeros(n, n))
end

=#
