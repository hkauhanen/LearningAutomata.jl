"""
    SRE(n::Int,
        c::Vector{Float64}) <: AbstractSRE

    A stationary random environment (SRE), a vector of constant penalty probabilities.
"""
struct SRE <: AbstractSRE
    c::Vector{Float64}
end


SRE(c...) = SRE(collect(convert.(Float64, c)))


# PRETTY-PRINTING

function Base.show(io::IO, z::AbstractSRE)
    print(io, "Stationary random environment (SRE) for ", Crayon(foreground=:cyan), length(z.c), Crayon(foreground=:default)," actions\n\n")
    print(io, "Penalty probabilities: ", Crayon(foreground=:light_red), z.c, Crayon(foreground=:default))
end


