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


"""
    reconfigure!(x::LRPLearner, p...)

Set the internal fields of an `LRPLearner` to those listed in dictionary `d`.

This is a [varargs function](https://docs.julialang.org/en/v1/manual/functions/#Varargs-Functions), where `p` is a list of pairs of arguments of the
kind `field => value`. 

# Examples

The following sets the learning rate vector `a` of a 
two-action `LRPLearner` to `[0.01, 0.02]` and the action cost vector `c` to 
`[0.01, 0.01]`:

```julia-repl
reconfigure!(x, :a => [0.01, 0.02], :c => [0.01, 0.01])
```
"""
function reconfigure!(x::LRPLearner, p...)
    [setfield!(x, i[1], i[2]) for i in p]
    revive_operators!(x)
    show(x)
end


