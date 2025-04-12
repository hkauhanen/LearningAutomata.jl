# TYPE

"""
    LRILearner(n::Int,
               γ::Vector{Float64},
               δ::Vector{Float64},
               W::Vector{Float64},
               A::Matrix{Float64},
               R::Vector{Matrix{Float64}}, 
               P::Vector{Matrix{Float64}}) <: AbstractLRILearner

A linear reward--inaction (LRI) learner.

A general linear reward--inaction learner with `n` actions, learning rates `γ`
for rewards, action costs `δ`, initial
action probability vector `W`, advantage matrix `A`, and vectors of reward 
and penalty operators `R` and `P`.

An LRI learner is like an LRP learner except it does nothing upon receiving
a punishment (inaction).

The end user normally need not worry about the `R` and `P` fields; these are
used internally by LearningAutomata.jl to implement the reward and penalty
functions and are automatically calculated from the other parameters.
"""
mutable struct LRILearner <: AbstractLRILearner
    n::Int
    γ::Vector{Float64}
    δ::Vector{Float64}
    W::Vector{Float64}
    A::Matrix{Float64}
    R::Vector{Matrix{Float64}}
    P::Vector{Matrix{Float64}}
end


# PRETTY-PRINTING

function Base.show(io::IO, z::AbstractLRILearner)
    print(io, "Linear reward-inaction learner (LRILearner) with ", Crayon(foreground=:cyan), z.n, Crayon(foreground=:default)," actions\n\n")
    print(io, "Reward rates:  ", Crayon(foreground=:light_blue), z.γ, Crayon(foreground=:default))
    print(io, "\n\nAction costs:  ", Crayon(foreground=:light_yellow), z.δ, Crayon(foreground=:default))
    print(io, "\n\nAdvantage matrix:\n")
    print(io, "\n\t[")
    for i in 1:size(z.A, 1)
        if i > 1
            print(io, "\n")
        end
        for j in 1:size(z.A, 2)
            if !(i == 1 && j == 1)
                print(io, "\t ")
            end
            print(io, Crayon(foreground=:cyan), z.A[i,j])
        end
    end
    print(io, Crayon(foreground=:default), "]\n")
    print(io, "\nCurrent action probabilities: ", Crayon(foreground=:green, bold=true), round.(z.W, digits=3), Crayon(foreground=:default, bold=false), "*")
    print(io, "\n\n*) ", Crayon(italics=true), "Rounded. To obtain exact probability vector, call ", Crayon(foreground=:green), "<learner>.W")
end



# CONSTRUCTORS

"""
    LRILearner(n::Int,
               γ::Vector{Float64};
               δ::Vector{Float64} = zeros(n),
               W::Vector{Float64} = ones(n) ./ n,
               A::Matrix{Float64} = zeros(n, n))

Create an `LRILearner` with `n` actions and learning rate vector `γ`.
"""
function LRILearner(n::Int,
           γ::Vector{Float64};
           δ::Vector{Float64} = zeros(n),
           W::Vector{Float64} = ones(n) ./ n,
           A::Matrix{Float64} = zeros(n, n))
    R = Vector{Matrix{Float64}}(undef, n)
    learner = LRILearner(n, γ, δ, W, A, R, copy(R))
    revive_operators!(learner)
    return learner
end


"""
    LRILearner(n::Int,
               γ::Float64;
               δ::Vector{Float64} = zeros(n),
               W::Vector{Float64} = ones(n) ./ n,
               A::Matrix{Float64} = zeros(n, n))

Create an `LRILearner` with `n` actions and learning rate `γ`.
"""
function LRILearner(n::Int,
           γ::Float64;
           δ::Vector{Float64} = zeros(n),
           W::Vector{Float64} = ones(n) ./ n,
           A::Matrix{Float64} = zeros(n, n))
    R = Vector{Matrix{Float64}}(undef, n)
    learner = LRILearner(n, γ .* ones(n), δ, W, A, R, copy(R))
    revive_operators!(learner)
    return learner
end


# UTILITY FUNCTIONS

function revive_operators!(x::AbstractLRILearner)
    for i in 1:x.n
        x.R[i] = (1 - x.γ[i] - x.δ[i]) * LinearAlgebra.I(x.n) + x.γ[i] * matrixunit(x.n, i) * ones(x.n, x.n) + (x.δ[i]/(x.n - 1)) * (ones(x.n, x.n) - matrixunit(x.n, i) * ones(x.n, x.n))
        x.P[i] = LinearAlgebra.I(x.n)
    end
end


