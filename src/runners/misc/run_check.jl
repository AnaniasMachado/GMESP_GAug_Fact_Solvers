using MAT
using LinearAlgebra
using Printf

data_n = 63
k = 63

matfile = matopen("data/data$data_n.mat")
C = data_n == 63 ? read(matfile, "A") : read(matfile, "C")
close(matfile)

C = Matrix{Float64}(C)
C = C[1:k, 1:k]
C = Symmetric(C)

Cmat = Matrix(C)

tol = 1e-12

min_entry = minimum(Cmat)
n_negative = count(x -> x < -tol, Cmat)
n_small_negative = count(x -> x < 0.0 && x >= -tol, Cmat)

println("="^80)
println("Entrywise nonnegativity check")
println("="^80)
println("data_n:                 ", data_n)
println("n:                      ", size(Cmat, 1))
@printf("minimum entry:          %.16e\n", min_entry)
println("tolerance:              ", tol)
println("entries < -tol:         ", n_negative)
println("entries in [-tol, 0):   ", n_small_negative)
println("entrywise nonnegative:  ", n_negative == 0)
println("="^80)

if n_negative > 0
    neg_idx = findall(x -> x < -tol, Cmat)

    println()
    println("Most negative entries:")
    vals = [(I[1], I[2], Cmat[I]) for I in neg_idx]
    sort!(vals, by = x -> x[3])

    for (r, c, v) in vals[1:min(20, length(vals))]
        @printf("C[%2d,%2d] = %.16e\n", r, c, v)
    end
end