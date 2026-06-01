using JuMP
using KNITRO

println("=== Knitro / Julia diagnostic test ===")
println("Julia executable: ", Base.julia_cmd())
println("Julia version: ", VERSION)

println("\n--- Environment variables ---")
for var in ["ARTELYS_LICENSE", "KNITRODIR", "PATH"]
    val = get(ENV, var, nothing)
    if val === nothing
        println(var, " = <not set>")
    else
        println(var, " = ", val)
    end
end

println("\n--- Package load test ---")
println("JuMP loaded successfully.")
println("KNITRO.jl loaded successfully.")

println("\n--- Solver test ---")

model = Model(KNITRO.Optimizer)

# Optional: reduce Knitro output noise
set_silent(model)

@variable(model, x)
@variable(model, y)

# Simple nonlinear convex problem:
# minimize (x - 1)^2 + (y - 2)^2
# subject to x + y >= 1
@objective(model, Min, (x - 1)^2 + (y - 2)^2)
@constraint(model, x + y >= 1)

optimize!(model)

status = termination_status(model)
primal_status_val = primal_status(model)

println("Termination status: ", status)
println("Primal status: ", primal_status_val)

if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
    println("\n Knitro appears to be working.")
    println("Objective value = ", objective_value(model))
    println("x = ", value(x))
    println("y = ", value(y))
else
    println("\n Knitro ran, but did not report a successful solve.")
end