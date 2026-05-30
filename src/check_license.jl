# src/check_knitro_license_version.jl

println("============================================================")
println("Knitro license/version check")
println("============================================================")

license_path = get(ENV, "ARTELYS_LICENSE", "")
knitrodir = get(ENV, "KNITRODIR", "")

println("ARTELYS_LICENSE = ", isempty(license_path) ? "<not set>" : license_path)
println("KNITRODIR        = ", isempty(knitrodir) ? "<not set>" : knitrodir)

println()
println("License file info:")
if isempty(license_path)
    println("ARTELYS_LICENSE is not set.")
elseif !isfile(license_path)
    println("License file does not exist.")
else
    println("exists = true")
    println("size   = ", filesize(license_path), " bytes")

    println()
    println("Raw license file lines, with long tokens redacted:")
    for (i, line) in enumerate(eachline(license_path))
        safe_line = replace(line, r"[A-Za-z0-9+/=_.:-]{16,}" => "<REDACTED_TOKEN>")
        println(lpad(i, 3), ": ", safe_line)
        if i >= 30
            break
        end
    end

    println()
    println("Searching license file for readable version/date keywords:")
    txt = read(license_path, String)
    for pattern in [
        r"(?i)version",
        r"(?i)knitro",
        r"(?i)expire",
        r"(?i)expiration",
        r"(?i)maintenance",
        r"(?i)202[0-9]",
        r"(?i)14\.0",
        r"(?i)13\.",
        r"(?i)12\.",
    ]
        m = match(pattern, txt)
        println(pattern, " => ", m === nothing ? "not found" : "found")
    end
end

println()
println("Knitro installation info:")
if isempty(knitrodir) || !isdir(knitrodir)
    println("KNITRODIR missing or not a directory.")
else
    for file in [
        joinpath(knitrodir, "INSTALL.txt"),
        joinpath(knitrodir, "README.txt"),
        joinpath(knitrodir, "Knitro_14_0_ReleaseNotes.txt"),
        joinpath(knitrodir, "LICENSE.txt"),
    ]
        println()
        println("File: ", file)
        if isfile(file)
            println("exists = true")
            println("First 15 lines:")
            for (i, line) in enumerate(eachline(file))
                println(lpad(i, 3), ": ", line)
                if i >= 15
                    break
                end
            end
        else
            println("exists = false")
        end
    end
end

println()
println("Trying Knitro command-line version/help output:")
function run_command_safely(cmd)
    println()
    println("Running: ", cmd)
    try
        output = read(cmd, String)
        println(output)
    catch err
        println("Command failed:")
        showerror(stdout, err)
        println()
    end
end

if Sys.iswindows()
    if !isempty(knitrodir)
        knitroampl = joinpath(knitrodir, "knitroampl", "knitroampl.exe")
        if isfile(knitroampl)
            run_command_safely(`$knitroampl -v`)
            run_command_safely(`$knitroampl -h`)
        else
            println("Could not find knitroampl.exe at ", knitroampl)
        end
    end
else
    if !isempty(knitrodir)
        knitroampl = joinpath(knitrodir, "knitroampl", "knitroampl")
        if isfile(knitroampl)
            run_command_safely(`$knitroampl -v`)
            run_command_safely(`$knitroampl -h`)
        else
            println("Could not find knitroampl at ", knitroampl)
        end
    end
end

println()
println("Trying JuMP/KNITRO with license debug:")
try
    ENV["ARTELYS_LICENSE_DEBUG"] = "1"

    using JuMP
    using KNITRO

    println("KNITRO.has_knitro() = ", KNITRO.has_knitro())
    println("KNITRO.jl path      = ", pathof(KNITRO))

    model = Model(KNITRO.Optimizer)

    @variable(model, x >= 0)
    @objective(model, Min, (x - 1.0)^2)
    optimize!(model)

    println("termination_status = ", termination_status(model))
    println("x = ", value(x))
catch err
    println("JuMP/KNITRO failed:")
    showerror(stdout, err)
    println()
end

println()
println("============================================================")
println("Done")
println("============================================================")