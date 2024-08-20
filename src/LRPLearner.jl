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
Kauhanen FIXME
Narendra and Thathachar FIXME
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

function Base.show(io::IO, z::LRPLearner)
    print(io, "Linear reward-penalty learner (LRPLearner) with ", Crayon(foreground=:cyan), z.n, Crayon(foreground=:default)," actions\n\n")
    print(io, "Reward rates:  ", Crayon(foreground=:light_blue), z.a, Crayon(foreground=:default))
    print(io, "\nPenalty rates: ", Crayon(foreground=:light_magenta), z.b, Crayon(foreground=:default))
    print(io, "\n\nAction costs:  ", Crayon(foreground=:light_yellow), z.c, Crayon(foreground=:default))
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
    print(io, "\n\n*) ", Crayon(italics=true), "Rounded. To obtain exact probability vector, call ", Crayon(foreground=:green), "get_probs(learner)", Crayon(foreground=:default), " or ", Crayon(foreground=:green), "learner.W")
end



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
    learner = LRPLearner(n, a, b, c, W, A, R, copy(R))
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
    learner = LRPLearner(n, a .* ones(n), b, c, W, A, R, copy(R))
    revive_operators!(learner)
    return learner
end


# UTILITY FUNCTIONS

function revive_operators!(x::LRPLearner)
    for i in 1:x.n
        x.R[i] = (1 - x.a[i] - x.c[i]) * LinearAlgebra.I + x.a[i] * matrixunit(x.n, i) * ones(x.n, x.n) + (x.c[i]/(x.n - 1)) * (ones(x.n, x.n) - matrixunit(x.n, i) * ones(x.n, x.n))
        x.P[i] = (1 - x.b[i] - x.c[i]) * LinearAlgebra.I + ((x.b[i] + x.c[i])/(x.n - 1)) * (ones(x.n, x.n) - matrixunit(x.n, i)* ones(x.n, x.n))
    end
end


