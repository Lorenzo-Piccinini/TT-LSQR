function[X, Res, iter] = TT_Tensorized_LSQR(values, Rhs, Params, X)


beta = norm(Rhs);

D = Rhs;

U = D; 
U = U/beta;

V = OpL_T(values, U);
V = round(V, Params.tol_tr);

alfa = norm(V);
V = V/alfa;

W = V;

phi_bar = beta;
rho_bar = alfa;

res0 = beta;
Res.real_abs = [res0];
Res.real_rel = [1];

res_old = res0;
totres = res0;

n_terms = length(values);
d = Rhs.d;



Ftot = OpL_T(values, Rhs);
res0_ne = norm(Ftot);

i = 0;
fprintf('iteation  res_ne_true   res_true\n')

while ( i < Params.imax )

    i = i+1;
    
    wrk1 = OpL(values, V);
    wrk1 = round(wrk1, Params.tol_tr, Params.rank_tr);

    U = wrk1 - alfa*U;
    U = round(U, Params.tol_tr, Params.rank_tr);

    beta = norm(U);
    U = U/beta;

    wrk2 = OpL_T(values, U);
    wrk2 = round(wrk2, Params.tol_tr, Params.rank_tr);

    V = wrk2 - beta*V;
    V = round(V, Params.tol_tr, Params.rank_tr);

    alfa = norm(V);
    V = V/alfa;

    rho = sqrt(rho_bar^2 + beta^2);
    c = rho_bar/rho;
    s = beta/rho;
    theta = s*alfa;
    rho_bar = -c*alfa;
    phi = c*phi_bar;
    phi_bar = s*phi_bar;

    res_est = phi_bar;
    res_ne_est = phi_bar*alfa*abs(c);

    X = X + (phi/rho)*W;
    X = round(X, Params.tol_tr, Params.rank_tr);

    LX = OpL(values, X);
    LTLX = OpL_T(values, LX);
    res11 = norm(Rhs-LX);
    res_true = norm(Ftot- LTLX);
    totres = [totres; res_true];

    if res_true <= Params.tol || abs(res_true - res_old) < Params.tol/10
        break, end

    res_old = res_true;

    Res.real_abs = [Res.real_abs; res_true];
    Res.real_rel = [Res.real_rel; res_true/res0_ne];

    W = V - (theta/rho)*W;
    W = round(W, Params.tol_tr, Params.rank_tr);

  % fprintf('  %d  %.4e %.4e\n', [i,res_true/res0_ne, res_true])
  fprintf('  %d  %.4e %.4e\n', [i,res11, res_true])

end
iter = i;
end


function[Y] = OpL(values, X)

d = X.d;
n_terms = length(values);

for k = 1:n_terms
    T{k} = X;
    for j = 1:d
        T{k} = ttm(T{k}, j, values{k}{j}');
    end
end
Y = T{1};
for k = 2:n_terms
    Y = Y + T{k};
end
end


function[Y] = OpL_T(values, X)

d = X.d;
n_terms = length(values);

for k = 1:n_terms
    T{k} = X;
    for j = 1:d
        T{k} = ttm(T{k}, j, values{k}{j});
    end
end
Y = T{1};
for k = 2:n_terms
    Y = Y + T{k};
end
end





