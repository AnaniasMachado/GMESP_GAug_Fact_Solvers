# src/check_knitro_license_full.jl

println("============================================================")
println("Knitro full license diagnostic")
println("============================================================")

println()
println("Julia version:")
println(VERSION)

println()
println("Working directory:")
println(pwd())

println()
println("Sys information:")
println("Sys.KERNEL = ", Sys.KERNEL)
println("Sys.MACHINE = ", Sys.MACHINE)
println("Sys.iswindows() = ", Sys.iswindows())
println("Sys.islinux() = ", Sys.islinux())
println("Sys.isapple() = ", Sys.isapple())

println()
println("Important environment variables:")
env_keys = [
    "ARTELYS_LICENSE",
    "ARTELYS_LICENSE_DEBUG",
    "KNITRODIR",
    "KNITRO_JL_USE_KNITRO_JLL",
    "PATH",
    "LD_LIBRARY_PATH",
]

for key in env_keys
    println(key, " = ", get(ENV, key, "<not set>"))
end

println()
println("============================================================")
println("License file checks")
println("============================================================")

license_path = get(ENV, "ARTELYS_LICENSE", "")

println("ARTELYS_LICENSE raw value:")
println(license_path)

if isempty(license_path)
    println("ARTELYS_LICENSE is empty or not set.")
else
    println("isfile(ARTELYS_LICENSE) = ", isfile(license_path))
    println("isdir(ARTELYS_LICENSE)  = ", isdir(license_path))

    if isfile(license_path)
        println("License file exists.")
        println("License file size in bytes = ", filesize(license_path))

        println()
        println("First 20 lines of license file, with possible long tokens redacted:")
        try
            for (i, line) in enumerate(eachline(license_path))
                if i > 20
                    break
                end

                # Redact long alphanumeric chunks to avoid exposing license secrets.
                safe_line = replace(line, r"[A-Za-z0-9+/=]{24,}" => "<REDACTED_LONG_TOKEN>")
                println(lpad(i, 3), ": ", safe_line)
            end
        catch err
            println("Could not read license file:")
            showerror(stdout, err)
            println()
        end
    else
        println("License file does not exist at ARTELYS_LICENSE path.")

        parent_dir = dirname(license_path)
        println("Parent directory = ", parent_dir)
        println("isdir(parent_dir) = ", isdir(parent_dir))

        if isdir(parent_dir)
            println()
            println("Files in parent directory:")
            try
                for name in readdir(parent_dir)
                    println("  ", name)
                end
            catch err
                println("Could not list parent directory:")
                showerror(stdout, err)
                println()
            end
        end
    end
end

println()
println("============================================================")
println("Knitro installation directory checks")
println("============================================================")

knitrodir = get(ENV, "KNITRODIR", "")

println("KNITRODIR raw value:")
println(knitrodir)

if isempty(knitrodir)
    println("KNITRODIR is empty or not set.")
else
    println("isdir(KNITRODIR) = ", isdir(knitrodir))

    if isdir(knitrodir)
        println()
        println("Top-level entries in KNITRODIR:")
        try
            for name in readdir(knitrodir)
                println("  ", name)
            end
        catch err
            println("Could not list KNITRODIR:")
            showerror(stdout, err)
            println()
        end

        possible_paths = String[]

        if Sys.iswindows()
            push!(possible_paths, joinpath(knitrodir, "knitroampl", "knitroampl.exe"))
            push!(possible_paths, joinpath(knitrodir, "bin", "knitroampl.exe"))
            push!(possible_paths, joinpath(knitrodir, "lib", "knitro.dll"))
            push!(possible_paths, joinpath(knitrodir, "bin", "knitro.dll"))
            push!(possible_paths, joinpath(knitrodir, "get_machine_ID.exe"))
            push!(possible_paths, joinpath(knitrodir, "bin", "get_machine_ID.exe"))
        else
            push!(possible_paths, joinpath(knitrodir, "knitroampl", "knitroampl"))
            push!(possible_paths, joinpath(knitrodir, "bin", "knitroampl"))
            push!(possible_paths, joinpath(knitrodir, "lib", "libknitro.so"))
            push!(possible_paths, joinpath(knitrodir, "get_machine_ID"))
            push!(possible_paths, joinpath(knitrodir, "bin", "get_machine_ID"))
        end

        println()
        println("Checking common Knitro executable/library paths:")
        for p in possible_paths
            println("  ", p, " | exists = ", isfile(p))
        end
    end
end

println()
println("============================================================")
println("Trying to locate Knitro executables in PATH")
println("============================================================")

function run_command_safely(cmd)
    println()
    println("Running command:")
    println(cmd)

    try
        output = read(cmd, String)
        println("Command output:")
        println(output)
    catch err
        println("Command failed:")
        showerror(stdout, err)
        println()
    end
end

if Sys.iswindows()
    run_command_safely(`where knitroampl`)
    run_command_safely(`where get_machine_ID`)
else
    run_command_safely(`which knitroampl`)
    run_command_safely(`which get_machine_ID`)
end

println()
println("============================================================")
println("Trying standalone Knitro / machine ID commands")
println("============================================================")

# Try get_machine_ID if found in KNITRODIR.
if !isempty(knitrodir) && isdir(knitrodir)
    machine_id_candidates = Sys.iswindows() ? [
        joinpath(knitrodir, "get_machine_ID.exe"),
        joinpath(knitrodir, "bin", "get_machine_ID.exe"),
    ] : [
        joinpath(knitrodir, "get_machine_ID"),
        joinpath(knitrodir, "bin", "get_machine_ID"),
    ]

    for exe in machine_id_candidates
        if isfile(exe)
            run_command_safely(`$exe -v`)
        end
    end
end

# Try knitroampl if found in KNITRODIR.
if !isempty(knitrodir) && isdir(knitrodir)
    knitroampl_candidates = Sys.iswindows() ? [
        joinpath(knitrodir, "knitroampl", "knitroampl.exe"),
        joinpath(knitrodir, "bin", "knitroampl.exe"),
    ] : [
        joinpath(knitrodir, "knitroampl", "knitroampl"),
        joinpath(knitrodir, "bin", "knitroampl"),
    ]

    for exe in knitroampl_candidates
        if isfile(exe)
            run_command_safely(`$exe`)
        end
    end
end

println()
println("============================================================")
println("Trying JuMP + KNITRO with ARTELYS_LICENSE_DEBUG = 1")
println("============================================================")

try
    ENV["ARTELYS_LICENSE_DEBUG"] = "1"

    using JuMP
    using KNITRO

    println("Loaded JuMP and KNITRO.")
    println("KNITRO.has_knitro() = ", KNITRO.has_knitro())

    println()
    println("Creating Model(KNITRO.Optimizer)...")

    model = Model(KNITRO.Optimizer)

    println("Model created successfully.")

    @variable(model, x >= 0)
    @objective(model, Min, (x - 1.0)^2)

    println("Optimizing tiny test problem...")
    optimize!(model)

    println("termination_status = ", termination_status(model))
    println("primal_status      = ", primal_status(model))
    println("objective_value    = ", objective_value(model))
    println("x                  = ", value(x))

catch err
    println()
    println("ERROR in JuMP + KNITRO test:")
    showerror(stdout, err)
    println()

    println()
    println("Stacktrace:")
    for frame in stacktrace(catch_backtrace())
        println(frame)
    end
end

println()
println("============================================================")
println("Trying to print KNITRO.jl package location")
println("============================================================")

try
    using KNITRO
    println("KNITRO module path candidates:")
    println(pathof(KNITRO))
catch err
    println("Could not get KNITRO path:")
    showerror(stdout, err)
    println()
end

println()
println("============================================================")
println("Diagnostic finished")
println("============================================================")