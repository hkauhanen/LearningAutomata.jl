function matrixunit(n::Int, i::Int)
  E = zeros(n, n)
  E[i,i] = 1.0
  return E
end

