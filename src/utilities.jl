"Matrix unit, i.e. the n x n square matrix with a 1.0 in position [i,i] and 0.0 elsewhere."
function matrixunit(n::Int, i::Int)
    E = zeros(n, n)
    E[i,i] = 1.0
    return E
end

