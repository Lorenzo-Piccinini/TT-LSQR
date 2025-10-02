function[X, Res] = TT_Tensorized_LSQR_Trunc_3terms(values1, values2, values3, Rhs, Params, X)
%
% [X, Res] = Tensorized_LSQR(values1, values2, D, Params)
%
% Tensorized version of the LSQR algorithm used to solve the least squares
% Sylvester tensor equation:
%
%    X x_1 A1 x_2 B1 x_3 C1 x_4 ... + X x_1 A2 x_2 B2 x_3 C2 x_4 ... = D,
%
% where X and D are tensors of dimensions compatible with those of the
% matrices and x_n is the n-mode product defined for tensors.
%
% INPUT:
% - values1 : contains the matrices A1, B1, C1, ...
% - values2 : contains the matrices A2, B2, C2, ...
% - Rhs : is the structure containing the right-hand side tensor, stored
%         as a core 1 x 1 x 1 x ... and the vectors of each mode
% - Params : contains the tolernce, truncation tolerance (if requested),
%            maximum number of iterations, initial guessing X0 (that
%            usually is fixed to the null tensor)
% - X : structure representing the initial guessing for the tensor solution
%       (usually it is the zero tensor) store as Rhs
%
% OUTPUT:
% - X : solution of the least squares problem, as a structure containing
%       the core tensor and the matrices to use for each mode
% - Res : structure containing all the residuals (both estimated and real
%         ones)

% TO DO:
% - new way to compute norms thanks to the structures ( DONE )
% - implement the truncation ( DONE )
% - theoretical support on why the new norm is working


% global sigmadrop

% beta = norm(D);
% beta = norm( ttm(Rhs.F, {Rhs.a, Rhs.b, Rhs.c}) )
beta = norm(Rhs);
% build RHS useful for later
D = Rhs;
% U = D; U = U/beta;
U = D; U = U/beta;

V = OpL_T(values1, values2, values3, U);
% V = round(V, Params.tol_tr, Params.r);
V = round(V, Params.tol_tr);
% alfa = norm(V); V= V/alfa;
% V
% pause
% alfa = norm( ttm(V.F, {V.U1, V.U2, V.U3}) ); 
alfa = norm(V); V = V/alfa;
% W = V;
W = V;
% X = Params.X0;

phi_bar = beta;
rho_bar = alfa;

res0 = beta;
res_ne_est = res0;
Res.est_abs = [res0];
Res.est_rel = [1];
Res.real_abs = [res0];
Res.real_rel = [1];
res_old = res0;
totres = res0;

F1 = Rhs; F2 = Rhs; F3 = Rhs;
    for k = 1:3 
    
        F1 = ttm(F1, k, values1{k});
        F2 = ttm(F2, k, values2{k});
        F3 = ttm(F3, k, values3{k});
    
    end

res0 = norm(F1+F2+F3);

i = 0;
fprintf('iteation   res_est   res_ne_est   res_true\n')

while ( i < Params.imax )

    i = i+1;

    uold = U; 
    % nu = U.n; pu = 1;
    % for i=1:length(nu)
    %     pu = pu*nu(i);
    % end
    % uold = reshape(uold, pu);
    
    
    % U = OpL(values1, values2, V) - alfa*U;
    wrk1 = OpL(values1, values2, values3, V);
    %wrk1 = trunc_HOSVD(wrk1, Params.tol_tr, Params.r);
    wrk1 = round(wrk1, Params.tol_tr, Params.r);
    
    %wrk1 = round(wrk1, Params.tol_tr);

    % can't control the orthogonality because they have different
    % dimensions.
    % tem1 = wrk1.U1; tem1 = tem1(:);
    % tem2 = U.U1; tem2 = tem2(:);
    % temorth = tem1'*tem2,

    % N = [size(values1{1},1), size(values1{2},1), size(values1{3},1)];
    
    U = wrk1 - alfa*U;
    % size(U.U1), size(U.U2), size(U.U3),
    % svd(U.U1), svd(U.U2), svd(U.U3),

    %U = round(U, Params.tol_tr, Params.r);
    
    U = round(U, Params.tol_tr);
    
    % sigmatot = [sigmatot; sigmadrop];

    % beta = norm(U); U = U/beta;
    % beta = norm( ttm(U.F, {U.U1, U.U2, U.U3}) ); 
    beta = norm(U); U = U/beta;

    unew = U;
    % nu = U.n; pu = 1;
    % for i = 1:length(nu)
    %     pu = pu*nu(i);
    % end
    % unew = reshape(U, pu);
    %size(uold), size(unew)
    %uorth(i) = unew'*uold;
    uorth(i) = dot(uold, unew);

    vold = V;
    % nv = V.n; pv = 1;
    % for i = 1:length(nv)
    %     pv = pv*nv(i);
    % end
    % vold = reshape(V,pv);

    % V = OpL_T(values1, values2, U) - beta*V;
    % values1, values2, U, pause
    wrk2 = OpL_T(values1, values2, values3, U);
    %wrk2 = round(wrk2, Params.tol_tr, Params.r);
    wrk2 = round(wrk2, Params.tol_tr);

    % N = [size(values1{1},2), size(values1{2},2), size(values1{3},2)];

    V = wrk2 - beta*V;
    %V = round(V, Params.tol_tr, Params.r);
    V = round(V, Params.tol_tr);

    % sigmatot1 = [sigmatot1; sigmadrop];

    % alfa = norm(V); V = V/alfa;
    % alfa = norm( ttm(V.F, {V.U1, V.U2, V.U3}) );
    alfa = norm(V); V = V/alfa;

    vnew = V;
    % nv = V.n; pv = 1;
    % for i = 1:length(nv)
    %     pv = pv*nv(i);
    % end
    % vnew = reshape(V,pv);
    % vorth(i) = vnew'*vold;
    vorth(i) = dot(vnew, vold);

    rho = sqrt(rho_bar^2 + beta^2);
    c = rho_bar/rho;
    s = beta/rho;
    theta = s*alfa;
    rho_bar = -c*alfa;
    phi = c*phi_bar;
    phi_bar = s*phi_bar;

    res_est = phi_bar;
    res_ne_est = phi_bar*alfa*abs(c);

    Res.est_rel = [Res.est_rel; res_est/res0];
    Res.est_abs = [Res.est_abs; res_est];

    % X = X + (phi/rho)*W;
    % N = [size(X.U1,1), size(X.U2,1), size(X.U3,1)];
 
    X = X + (phi/rho)*W;
    %X = round(X, Params.tol_tr, Params.r);
    X = round(X, Params.tol_tr);

    % Computing the real residual 
    temp1 = X;
    temp2 = X;
    temp3 = X;

    for k = 1:3
        
        temp1 = ttm(temp1, k, values1{k}');
        temp2 = ttm(temp2, k, values2{k}');
        temp3 = ttm(temp3, k, values3{k}');

    end
    coef1 = temp1 + temp2 + temp3;
    %size(coef1)
    wrk1 = coef1;
    wrk2 = coef1;
    wrk3 = coef1;
    for k = 1:3 
    
        wrk1 = ttm(wrk1, k, values1{k});
        wrk2 = ttm(wrk2, k, values2{k});
        wrk3 = ttm(wrk3, k, values3{k});
    
    end
 

    %r_trunc_tensor = norm(F-temp1-temp2-temp3)/norm(F);
    res_true = norm((F1+F2+F3)-wrk1-wrk2-wrk3);
    
    %res_true = norm(D-temp1-temp2-temp3);
    totres = [totres; res_true];

    if res_true <= Params.tol || abs(res_true - res_old) < Params.tol/10
        break, end

    res_old = res_true;
    % res_true = 0;
    % Find a way to compute norms that uses the structure of the tensors.
    Res.real_abs = [Res.real_abs; res_true];
    Res.real_rel = [Res.real_rel; res_true/res0];
    
    % W = V - (theta/rho)*W;
    % N = [size(V.U1,1), size(V.U2,1), size(V.U3,1)];


    W = V - (theta/rho)*W;
    %W = round(W, Params.tol_tr, Params.r);
    W = round(W, Params.tol_tr);
    
    % disp([i, res_est, res_ne_est, res_true])
    fprintf(' %d  %.4e %.4e %.4e\n', [i, res_est/res0, res_ne_est/res0, res_true/res0])

end

%%{
figure(502)
semilogy(totres/totres(1),'d-','linewidth',4)
%semilogy(r_res,'linewidth',4)
hold on
semilogy(abs(uorth),'o','linewidth',4)
semilogy(abs(vorth),'x','linewidth',4)
%semilogy(totrank/totrank(end),'x','linewidth',4)
%rf=find(totrank/totrank(end)==1,1,'first');
%semilogy([rf,rf],[1e-15,10],'linewidth',4)
hold off
legend('true res normal eqn','orth U','orth V','max rank')
xlabel('number of iterations')
ylabel('loss of optimality')
 axis([0,150,1e-16,10]);
hold off
% figure(203)
% semilogy(sigmatot,'o','linewidth',4)
% hold on
% semilogy(sigmatot1,'x','linewidth',4)
% semilogy([rf,rf],[1e-15,10],'linewidth',4)
% hold off 
%  axis([0,150,1e-16,10]);
% legend('U','V')
% xlabel('number of iterations')
% ylabel('magnitude of dropped sing.values')
%}
end 
% end of main function




function[Y] = OpL(values1, values2, values3, X)

% wrk1 = ttm(X,Coef1.A,1) + ttm(X,Coef1.B,2) + ttm(X,Coef1.C,3);
% wrk1 = ttm(X, values1);
% wrk2 = ttm(X,Coef2.A,1) + ttm(X,Coef2.B,2) + ttm(X,Coef2.C,3);
% wrk2 = ttm(X, values2);
% Y = wrk1 + wrk2;

% N = [size(values1{1},1), size(values1{2},1), size(values1{3},1)];
d = X.d;
T1 = X; T2 = X; T3 = X;

for k = 1:d
    T1 = ttm(T1, k, values1{k}');
    T2 = ttm(T2, k, values2{k}');
    T3 = ttm(T3, k, values3{k}');
end

Y = T1+T2+T3;

end

function[Z] = OpL_T(values1, values2, values3, Y)

% wrk1 = ttm(Y,Coef1.A',1) + ttm(Y,Coef1.B',2) + ttm(Y,Coef1.C',3);
% wrk1 = ttm(Y, values1, 't');
% wrk2 = ttm(Y,Coef2.A',1) + ttm(Y,Coef2.B',2) + ttm(Y,Coef2.C',3);
% wrk2 = ttm(Y, values2, 't');
% Z = wrk1 + wrk2;

% N = [size(values1{1},2), size(values1{2},2), size(values1{3},2)];

d = Y.d;
T1 = Y; T2 = Y; T3 = Y;

for k = 1:d
    T1 = ttm(T1, k, values1{k});
    T2 = ttm(T2, k, values2{k});
    T3 = ttm(T3, k, values3{k});
end

Z = T1+T2+T3;

end