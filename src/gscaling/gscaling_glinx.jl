function lower_triangular_solver(A)
    return [A[i, j] for i = 1:size(A, 1) for j = 1:i]
end

function build_linx_gmesp(C::Symmetric{<:Real,<:AbstractMatrix},s::Integer,t::Integer;optimizer = Hypatia.Optimizer, gamma = 1.0)
    n = size(C,1);
    model = Model(optimizer)
    set_silent(model)   
    @variable(model, x[1:n])
    @variable(model, X[1:n,1:n], Symmetric)
    @variable(model, t_obj)
    @constraint(model, con_psd_X, X in PSDCone())
    @constraint(model,tr(X) == t)
    @constraint(model,x .>= 0.0)
    @constraint(model,x .<= 1.0)
    @constraint(model, con_psd_Y, X <= diagm(x), PSDCone())  
    @constraint(model,sum(x) == s)
    E = gamma*C*X*C + I - X;
    @constraint(model, [t_obj; 1; lower_triangular_solver(E)] in MOI.LogDetConeTriangle(n))
    @objective(model,Max,t_obj)
    return model
end

function linx_gmesp(C::Symmetric{<:Real,<:AbstractMatrix},s::Integer,t::Integer;optimizer = Hypatia.Optimizer, gamma = 1.0, return_model = false)
    model = build_linx_gmesp(C,s,t;optimizer=optimizer,gamma=gamma)
    optimize!(model)
    has_values(model) || error("GLINX has no primal solution: $(termination_status(model))")
    X = value.(model[:X]); x = value.(model[:x]);
    L = gamma*C*X*C + I - X
    z_rel = 0.5*logdet(Symmetric(0.5*(L+L'))) - 0.5*t*log(gamma)

    return return_model ? (x,X,z_rel,model) : (x,X,z_rel)
end

function info_subprob_gamma_linx(X,C,t,gamma; verbose=true)
    n = size(C, 1)
    CXC = C*X*C;
    F = Symmetric(gamma*CXC - X + Matrix(1.0I,n,n));
    Finv = inv(F);
    Finv = Symmetric((Finv + Finv')/2);
    res = n - t - tr(Finv*(I-X)); 
    if verbose
        @show res,gamma
    end
    return CXC,Finv,res
end

function compute_gamma_linx(C,s,t;tol = 1e-2,gamma = 0)
    global n = size(C,1); iter = 0; max_iter = 20; 
    # Tolerance for residual and dualtity gap  
    # Select initial gamma
    c_sort=sort(diag(C),rev=true);
    if gamma == 0
        gamma=(c_sort[t])^(-1.5);
    end
    # solve the linx relaxation for gamma and obtain x
    x,X,bound = linx_gmesp(C,s,t;gamma=gamma);
    GC.gc()
    CXC,Finv,res = info_subprob_gamma_linx(X,C,t,gamma);
    nx = x; nX = X;
    # wolfe parameters
    c1=1e-4; c2=0.9;
    while abs(res) > tol
        # Compute MAT= H_G(gamma:=exp(psi))
        Finv2 = sqrt(Finv)
        tmp = gamma*Finv2*CXC*Finv2; tmp = Symmetric(0.5*(tmp+tmp'));
        Mat = tr(tmp) -tr(tmp^2)
        Mat = max(Mat,1e-12); # avoid issues in newton step
        # Compute the search direction for Newton's method (dir=-res/Mat)
        dir = -res/Mat; edir=exp(dir);
        if !isfinite(edir); edir=sign(edir)*1e10; end;
        dir=log(edir);
        # check if alpha=1 satisfies the Strong Wolfe Conditions
        alpha=1;
        if dir <= - 20; dir = 20; elseif dir >= 20; dir = 20; end;
        ngamma=gamma*exp(alpha*dir);
        GC.gc()
        nx,nX,nbound = linx_gmesp(C,s,t;gamma=ngamma);
        nCXC,nFinv,nres = info_subprob_gamma_linx(nX,C,t,ngamma)
        if ((nbound-bound) > c1*alpha*dir*res) || abs(dir*nres) > c2*abs(dir*res)
            judge = true;
        else
            judge = false;
        end
        # line search
        a=0; b=1;
        while judge
            alpha=(a+b)/2;
            ngamma=gamma*exp(alpha*dir);
            GC.gc()
            nx,nX,nbound = linx_gmesp(C,s,t;gamma=ngamma);
            nCXC,nFinv,nres = info_subprob_gamma_linx(nX,C,t,ngamma)
            if (nbound-bound) > c1*alpha*dir*res
                b = alpha;
            elseif abs(dir*nres)>c2*abs(dir*res)
                a = alpha;
            else
                break
            end
            if abs(b-a) < 1e-2
                break
            end   
        end
        gamma=ngamma;res=nres;Finv=nFinv;CXC=nCXC;X=nX;bound=nbound;
        iter += 1;
        if iter >= max_iter;break;end
    end
    return gamma
end

function compute_gamma_linx_armijo(C,s,t;tol = 1e-2,gamma = 0)
    global n = size(C,1); iter = 0; max_iter = 20; 
    # Tolerance for residual and dualtity gap  
    # Select initial gamma
    c_sort=sort(diag(C),rev=true);
    if gamma == 0
        gamma=(c_sort[t])^(-1.5);
    end
    # solve the linx relaxation for gamma and obtain x
    x,X,bound = linx_gmesp(C,s,t;gamma=gamma);
    GC.gc()
    CXC,Finv,res = info_subprob_gamma_linx(X,C,t,gamma);
    nx = x; nX = X;
    # Armijo parameter
    c1=1e-4;
    while abs(res) > tol
        # Compute MAT= H_G(gamma:=exp(psi))
        Finv2 = sqrt(Finv)
        tmp = gamma*Finv2*CXC*Finv2; tmp = Symmetric(0.5*(tmp+tmp'));
        Mat = tr(tmp) -tr(tmp^2)
        Mat = max(Mat,1e-12); # avoid issues in newton step
        # Compute the search direction for Newton's method (dir=-res/Mat)
        dir = -res/Mat; edir=exp(dir);
        if !isfinite(edir); edir=sign(edir)*1e10; end;
        dir=log(edir);
        # truncate the search direction
        alpha=1;
        if dir <= -20; dir = -20; elseif dir >= 20; dir = 20; end;
        # line search using the Armijo condition
        max_ls_iter = 20;
        ls_iter = 0;
        ngamma = gamma;
        nres = res;
        nFinv = Finv;
        nCXC = CXC;
        nX = X;
        nbound = bound;
        while true
            ngamma=gamma*exp(alpha*dir);
            GC.gc()
            nx,nX,nbound = linx_gmesp(C,s,t;gamma=ngamma);
            nCXC,nFinv,nres = info_subprob_gamma_linx(nX,C,t,ngamma)
            # check if alpha satisfies the Armijo condition
            if (nbound-bound) <= c1*alpha*dir*res
                break
            end
            # backtracking
            alpha=alpha/2;
            ls_iter += 1;
            # maximum number of line search iterations
            if ls_iter >= max_ls_iter
                break
            end
        end
        gamma=ngamma;res=nres;Finv=nFinv;CXC=nCXC;X=nX;bound=nbound;
        iter += 1;
        if iter >= max_iter;break;end
    end
    return gamma
end

function glinx(C,s,t;tol=1e-2,gamma=0)
    gamma_opt = compute_gamma_linx_armijo(C,s,t;tol=tol,gamma=gamma);
    GC.gc()
    x, X, objval = linx_gmesp(C,s,t;gamma = gamma_opt);
    return x, X, objval,gamma_opt
end

