function[X, Res] = RandTT_Tensorized_LSQR(terms, Rhs, Params, X)
l = length(terms); N = length(X);
termsT = cell(l,1);

for i = 1:l
    termsT{i} = cell(N,1);
    for j = 1:N
        
        termsT{i}{j} = terms{i}{j}';
    end
end

beta = TTnorm(Rhs);

D = Rhs;

U = D; 
%U = U/beta;
U = TTscale(U, 1/beta);

V = OpL_T(termsT, U);
%V = round(V, Params.tol_tr);

alfa = TTnorm(V);
V = TTscale(V, 1/alfa);

W = V;

phi_bar = beta;
rho_bar = alfa;

res0 = beta;
Res.real_abs = [res0];
Res.real_rel = [1];

res_old = res0;
totres = res0;

n_terms = length(terms);
d = length(Rhs);



Ftot = OpL_T(termsT, Rhs);
res0_ne = TTnorm(Ftot);

i = 0;
fprintf('iteation  res_ne_true   res_true\n')

while ( i < Params.imax )

    i = i+1;
    
    wrk1{1} = OpL(terms, V);
    % wrk1 = round(wrk1, Params.tol_tr);
    wrk1{2} = U;

    U = TTsum_Randomize_then_Orthogonalize(wrk1, [1; -alfa]);

    beta = TTnorm(U);
    U = TTscale(U, 1/beta);

    wrk2{1} = OpL_T(termsT, U);
    wrk2{2} = V;

    V = TTsum_Randomize_then_Orthogonalize(wrk2, [1; -beta]);

    alfa = TTnorm(V);
    V = TTscale(V, 1/alfa);

    rho = sqrt(rho_bar^2 + beta^2);
    c = rho_bar/rho;
    s = beta/rho;
    theta = s*alfa;
    rho_bar = -c*alfa;
    phi = c*phi_bar;
    phi_bar = s*phi_bar;

    res_est = phi_bar;
    res_ne_est = phi_bar*alfa*abs(c);

    wrk3{1} = X;
    wrk3{2} = W;
    X = TTsum_Randomize_then_Orthogonalize(wrk3, [1; phi/rho]);

    LX = OpL(terms, X);
    LTLX = OpL_T(termsT, LX);

    wrk4{1} = LTLX;
    wrk4{2} = Ftot;

    Res_ne = TTsum_Randomize_then_Orthogonalize(wrk4, [1; -1]);
    
    res_true = TTnorm(Res_ne);
    totres = [totres; res_true];

    if res_true <= Params.tol || abs(res_true - res_old) < Params.tol/10
        break, end

    res_old = res_true;

    Res.real_abs = [Res.real_abs; res_true];
    Res.real_rel = [Res.real_rel; res_true/res0_ne];

    wrk5{1} = V; 
    wrk5{2} = W;
    W = TTsum_Randomize_then_Orthogonalize(wrk5, [1; -theta/rho]);

    fprintf('  %d  %.4e %.4e\n', [i,res_true/res0_ne, res_true])

end
end


function[Y] = OpL(terms, X)

l = length(terms);
coeffs = ones(l,1);
temp = cell(l,1);
for i = 1:l
    temp{i} = TTKronOp(terms{i}, X);
end
Y = TTsum_Randomize_then_Orthogonalize(temp, coeffs);
end


function[Y] = OpL_T(termsT, X)

l = length(termsT);
coeffs = ones(l,1);
temp = cell(l,1);
for i = 1:l
    temp{i} = TTKronOp(termsT{i}, X);
end
Y = TTsum_Randomize_then_Orthogonalize(temp, coeffs);

end





