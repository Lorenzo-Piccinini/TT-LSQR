function[X, Res] = Tensorized_LSQR_Trunc_3terms(values1, values2, values3, Rhs, Params, X)
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
beta = norm(Rhs.F);
% build RHS useful for later
D = ttm(Rhs.F, {Rhs.a, Rhs.b, Rhs.c});
% U = D; U = U/beta;
U.F = Rhs.F; U.F = U.F/beta;
U.U1 = Rhs.a;
U.U2 = Rhs.b;
U.U3 = Rhs.c;

V = OpL_T(values1, values2, values3, U);
V = trunc_HOSVD(V, Params.tol_tr, Params.r);
% alfa = norm(V); V= V/alfa;
% V
% pause
% alfa = norm( ttm(V.F, {V.U1, V.U2, V.U3}) ); 
alfa = norm(V.F); V.F = V.F/alfa;
% W = V;
W.F = V.F;
W.U1 = V.U1;
W.U2 = V.U2;
W.U3 = V.U3;
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

i = 0;
fprintf('iteation   res_est   res_ne_est   res_true\n')

while ( i < Params.imax )

    i = i+1;

    uold = ttm(U.F, {U.U1, U.U2, U.U3}); 
    uold = uold(:);
    
    % U = OpL(values1, values2, V) - alfa*U;
    wrk1 = OpL(values1, values2, values3, V);
    wrk1 = trunc_HOSVD(wrk1, Params.tol_tr, Params.r);

    % can't control the orthogonality because they have different
    % dimensions.
    % tem1 = wrk1.U1; tem1 = tem1(:);
    % tem2 = U.U1; tem2 = tem2(:);
    % temorth = tem1'*tem2,

    % N = [size(values1{1},1), size(values1{2},1), size(values1{3},1)];
    N = size(wrk1.F); M = size(U.F);
    SU = tensor(zeros(N+M));
    SU(1:N(1), 1:N(2), 1:N(3)) = wrk1.F;
    SU(N(1)+1:end, N(2)+1:end, N(3)+1:end) = -alfa*U.F;
    U.F = SU;
    U.U1 = [wrk1.U1, U.U1];
    U.U2 = [wrk1.U2, U.U2];
    U.U3 = [wrk1.U3, U.U3];
    % size(U.U1), size(U.U2), size(U.U3),
    % svd(U.U1), svd(U.U2), svd(U.U3),

    U = trunc_HOSVD(U, Params.tol_tr, Params.r);
    
    % sigmatot = [sigmatot; sigmadrop];

    % beta = norm(U); U = U/beta;
    % beta = norm( ttm(U.F, {U.U1, U.U2, U.U3}) ); 
    beta = norm(U.F); U.F = U.F/beta;

    unew = ttm(U.F, {U.U1, U.U2, U.U3});
    unew = unew(:);
    uorth(i) = unew'*uold;

    vold = ttm(V.F, {V.U1, V.U2, V.U3});
    vold = vold(:);

    % V = OpL_T(values1, values2, U) - beta*V;
    % values1, values2, U, pause
    wrk2 = OpL_T(values1, values2, values3, U);
    wrk2 = trunc_HOSVD(wrk2, Params.tol_tr, Params.r);

    % N = [size(values1{1},2), size(values1{2},2), size(values1{3},2)];
    N = size(wrk2.F); M = size(V.F);
    SV = tensor(zeros(N+M));
    SV(1:N(1), 1:N(2), 1:N(3)) = wrk2.F;
    SV(N(1)+1:end, N(2)+1:end, N(3)+1:end) = -beta*V.F;
    V.F = SV;
    V.U1 = [wrk2.U1, V.U1];
    V.U2 = [wrk2.U2, V.U2];
    V.U3 = [wrk2.U3, V.U3];
    V = trunc_HOSVD(V, Params.tol_tr, Params.r);

    % sigmatot1 = [sigmatot1; sigmadrop];

    % alfa = norm(V); V = V/alfa;
    % alfa = norm( ttm(V.F, {V.U1, V.U2, V.U3}) );
    alfa = norm(V.F); V.F = V.F/alfa;

    vnew = ttm(V.F, {V.U1, V.U2, V.U3});
    vnew = vnew(:);
    vorth(i) = vnew'*vold;

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
    N = size(X.F); M = size(W.F);
    SX = tensor(zeros(N+M));
    SX(1:N(1), 1:N(2), 1:N(3)) = X.F;
    SX(N(1)+1:end, N(2)+1:end, N(3)+1:end) = (phi/rho)*W.F;

    X.F = SX;
    % X, W
    X.U1 = [X.U1, W.U1];
    X.U2 = [X.U2, W.U2];
    X.U3 = [X.U3, W.U3];
   X = trunc_HOSVD(X, Params.tol_tr, Params.r);

    % Computing the real residual 
    % 
    % wrk1 = ttm(X,Coef1.A,1) + ttm(X,Coef1.B,2) + ttm(X,Coef1.C,3);
    % wrk2 = ttm(X,Coef2.A,1) + ttm(X,Coef2.B,2) + ttm(X,Coef2.C,3);
    wrk1 = ttm(X.F, {values1{1}*X.U1, values1{2}*X.U2, values1{3}*X.U3});
    wrk2 = ttm(X.F, {values2{1}*X.U1, values2{2}*X.U2, values2{3}*X.U3});
    wrk3 = ttm(X.F, {values3{1}*X.U1, values3{2}*X.U2, values3{3}*X.U3});
    
    res_true = norm(D-wrk1-wrk2-wrk3);
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
    N = size(V.F); M = size(W.F);
    SW = tensor(zeros(N+M));
    SW(1:N(1), 1:N(2), 1:N(3)) = V.F;
    SW(N(1)+1:end, N(2)+1:end, N(3)+1:end) = -(theta/rho)*W.F;

    W.F = SW;
    W.U1 = [V.U1, W.U1];
    W.U2 = [V.U2, W.U2];
    W.U3 = [V.U3, W.U3];
    W = trunc_HOSVD(W, Params.tol_tr, Params.r);
    
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
N = size(X.F);
G = tensor(zeros(3*N));
G(1:N(1), 1:N(2), 1:N(3)) = X.F;
G(N(1)+1:2*N(1), N(2)+1:2*N(2), N(3)+1:2*N(3)) = X.F;
G(2*N(1)+1:end, 2*N(2)+1:end, 2*N(3)+1:end) = X.F;

Y.F = G;
Y.U1 = [values1{1}*X.U1, values2{1}*X.U1, values3{1}*X.U1];
Y.U2 = [values1{2}*X.U2, values2{2}*X.U2, values3{2}*X.U2];
Y.U3 = [values1{3}*X.U3, values2{3}*X.U3, values3{3}*X.U3];

end

function[Z] = OpL_T(values1, values2, values3, Y)

% wrk1 = ttm(Y,Coef1.A',1) + ttm(Y,Coef1.B',2) + ttm(Y,Coef1.C',3);
% wrk1 = ttm(Y, values1, 't');
% wrk2 = ttm(Y,Coef2.A',1) + ttm(Y,Coef2.B',2) + ttm(Y,Coef2.C',3);
% wrk2 = ttm(Y, values2, 't');
% Z = wrk1 + wrk2;

% N = [size(values1{1},2), size(values1{2},2), size(values1{3},2)];
N = size(Y.F);
G = tensor(zeros(3*N));
G(1:N(1), 1:N(2), 1:N(3)) = Y.F;
G(N(1)+1:2*N(1), N(2)+1:2*N(2), N(3)+1:2*N(3)) = Y.F;
G(2*N(1)+1:end, 2*N(2)+1:end, 2*N(3)+1:end) = Y.F;

Z.F = G;
Z.U1 = [values1{1}'*Y.U1, values2{1}'*Y.U1, values3{1}'*Y.U1];
Z.U2 = [values1{2}'*Y.U2, values2{2}'*Y.U2, values3{2}'*Y.U2];
Z.U3 = [values1{3}'*Y.U3, values2{3}'*Y.U3, values3{3}'*Y.U3];

end