using LinearAlgebra

function prod_eigs(Csub,t::Int)
    λ = reverse(eigvals(Symmetric(Csub)));
    return sum(log, @view λ[1:t])
end

function init_heur_soln(C::Symmetric{<:Real,<:AbstractMatrix},s::Int,t::Int, type_init::Symbol)
    n = size(C,1);
    if type_init == :Cont
        λ,U = eigen(C);
        perm = length(λ):-1:1          
        U = U[:, perm];
        x_save = vec(sum(abs2, @view U[:, 1:t]; dims=2))
        S = partialsortperm(x_save, 1:s;rev=true)
    elseif type_init == :Greedy
        S = Int[]
        remaining = collect(1:n)
        for _ in 1:s
            best_j = 0
            best_val = -Inf
            for j in remaining
                cand = vcat(S, j)
                Csub = @view C[cand,cand];
                t_eff = min(t, length(cand))
                val = prod_eigs(Csub,t_eff)
                if val > best_val
                    best_val = val
                    best_j = j
                end
            end
            push!(S, best_j)
            deleteat!(remaining, findfirst(==(best_j), remaining))
        end
    elseif type_init == :ReverseGreedy
        S = collect(1:n)
        while length(S) > s
            best_pos = 0
            best_val = -Inf
            for (pos,_) in enumerate(S)
                cand = vcat(S[1:pos-1], S[pos+1:end]) # S ∖ j
                Csub = @view C[cand,cand];
                val = prod_eigs(Csub,t);
                if val > best_val
                    best_val = val
                    best_pos = pos
                end
            end
            deleteat!(S, best_pos)
        end        
    else
        error("Unknown type_init = $(type_init)")
    end
    arr_init = Vector{Int64}(S)
    return arr_init
end

function runLS(C::Symmetric{<:Real,<:AbstractMatrix}, n::Int, s::Int, t::Int, type_LS::Symbol; atol=1e-5, arr_init = [])
    if length(arr_init) == s
        arr_rows = copy(arr_init);
    else    
        arr_rows = collect(1:s);
    end

    in_set = falses(n);
    in_set[arr_rows] .= true;
    Csub = @view C[arr_rows, arr_rows]
    best_val = prod_eigs(Csub, t)

    while true
        improved = false
        leave_ind = 0
        enter_ind = 0

        for i in 1:s
            old = arr_rows[i]
            for j in 1:n
                if !in_set[j]
                    # try swap i -> j
                    arr_rows[i] = j
                    Csub = @view C[arr_rows, arr_rows]
                    new_val = prod_eigs(Csub, t)
                    arr_rows[i] = old

                    if new_val >= best_val + atol
                        best_val = new_val
                        leave_ind = i
                        enter_ind = j
                        improved = true
                        if type_LS == :FI
                            break
                        end
                    end
                end
            end
            if improved && (type_LS == :FI || type_LS == :FP)
                break
            end
        end

        if !improved
            break
        end

        # commit the best move found
        old = arr_rows[leave_ind]
        in_set[old] = false
        arr_rows[leave_ind] = enter_ind
        in_set[enter_ind] = true
    end

    x = zeros(n)
    x[arr_rows] .= 1.0
    z = best_val
    return x, z
end

function run_all_LS(C::Symmetric{<:Real,<:AbstractMatrix},s::Int,t::Int)
    n = size(C,1);

    x_ls = zeros(n); z_ls = -Inf;
    for init_strategy in [:Cont,:Greedy,:ReverseGreedy]
        arr_init = init_heur_soln(C,s,t, init_strategy)
        for ls_strategy in [:FI,:FP,:BI]
            x_temp,z_temp =  runLS(C,n,s,t,ls_strategy;arr_init = copy(arr_init));
            if z_temp > z_ls
                x_ls = x_temp; z_ls = z_temp; 
            end
        end
    end
    return x_ls,z_ls
end