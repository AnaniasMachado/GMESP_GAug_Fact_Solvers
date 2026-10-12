
using CSV
using DataFrames
using MAT
using LinearAlgebra
using Random
using Statistics

df = CSV.read("data/generated/instance_statistics.csv", DataFrame)

# Keep instances that improve over GLINX on average,
# then rank by the fraction of nodes satisfying the
# DDGFact+ closed-form condition.
ranking = combine(
    groupby(df, :id),
    :gap_difference => mean => :improvement,
    :frac_holds => mean => :fraction,
)

filter!(
    r -> isfinite(r.improvement) &&
         r.improvement > 0 &&
         isfinite(r.fraction),
    ranking,
)

sort!(ranking, :fraction, rev = true)

output_dir = "data/generated/best"
mkpath(output_dir)

for (rank, x) in enumerate(eachrow(first(ranking, 10)))
    row = df[findfirst(==(x.id), df.id), :]

    n = Int(row.n)
    t = Int(row.t)

    source = matread(String(row.source_file))
    C0 = n == 63 ? source["A"] : source["C"]
    C0 = Symmetric((C0 + C0') / 2)

    F = eigen(C0)
    b = reverse(F.values)
    Q = F.vectors[:, end:-1:1]

    r = t + Int(row.delta)
    m = Int(row.m)
    nmid = n - r - m

    mid_orig = b[r+1:r+nmid]
    lo = minimum(b)
    hi = nmid > 0 ? maximum(mid_orig) : 1.0

    v = hi > lo ? (mid_orig .- lo) ./ (hi - lo) : zeros(nmid)

    rng = MersenneTwister(Int(row.generation_seed))

    top = row.psi0 .+
          row.sigma .* (1 .+ row.jit_top .* rand(rng, r))

    mid = row.psi0 .+
          row.jit_bot .+
          row.epsilon .* v

    bot = row.psi0 .+
          row.jit_bot .* rand(rng, m)

    m > 0 && (bot[argmin(bot)] = row.psi0)

    λ = vcat(
        sort(top; rev = true),
        sort(mid; rev = true),
        sort(bot; rev = true),
    )

    C = Q * Diagonal(λ) * Q'
    C = (C + C') / 2

    filepath = joinpath(
        output_dir,
        "data$(n)_inst$(rank).mat",
    )

    matwrite(filepath, Dict(
        "C" => C,
        "lambda" => λ,
        "n" => n,
        "t" => t,
        "original_id" => Int(x.id),
        "mean_improvement" => x.improvement,
        "closed_form_fraction" => x.fraction,
    ))

    println(
        "Saved $filepath | " *
        "ID $(x.id) | " *
        "improvement = $(round(x.improvement; digits = 6)) | " *
        "closed-form fraction = $(round(x.fraction; digits = 6))"
    )
end
