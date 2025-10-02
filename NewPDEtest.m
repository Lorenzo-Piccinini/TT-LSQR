%clear all
close all
clc

format shorte
format compact

N = []; M = [];

n1 = 50; N = [N, n1]; m1 = 50; M = [M, m1];
n2 = 50; N = [N, n2]; m2 = 50; M = [M, m2];
n3 = 50; N = [N, n3]; m3 = 50; M = [M, m3];

d = 3;


h1 = 1/n1;
h2 = 1/n2;
h3 = 1/n3;

x_nodes = linspace(0,1,n1+1)';
y_nodes = linspace(0,1,n2+1)';
z_nodes = linspace(0,1,n3+1)';
addpath('./Data')
load mnist_all
addpath('./oseledets_TT-Toolbox')

% e = ones(n1+1,1);
% T1 = spdiags([e -2*e e],-1:1,m1+1,m1+1); 
% e = ones(n2+1,1);
% T2 = spdiags([e -2*e e],-1:1,m2+1,m2+1); 
% e = ones(n3+1,1);
% T3 = spdiags([e -2*e e],-1:1,m3+1,m3+1); 

a = ones(n1+1,1); a = a/norm(a);
b = ones(n2+1,1); b = b/norm(b);
c = ones(n3+1,1); c = c/norm(c);

e = ones(m1+1,1);
A1 = -spdiags([e -2*e e],-1:1,m1+1,m1+1); 
e = ones(m2+1,1);
A2 = -spdiags([e -2*e e],-1:1,m2+1,m2+1); 
e = ones(m3+1,1);
A3 = -spdiags([e -2*e e],-1:1,m3+1,m3+1); 


% Phi1 = diag(2*ones(n1+1,1)) + diag(2+sin(2*pi*x_nodes)); 
% Psi1 = diag(2*ones(n1+1,1)) +diag(sin(2*pi*y_nodes));
% Ni1 = diag(2*ones(n1+1,1)) + diag(sin(2*pi*z_nodes));
% 
% Phi2 =diag(2*ones(n1+1,1)) + diag(sin(2*pi*x_nodes));
% Psi2 =diag(2*ones(n1+1,1)) + diag(sin(2*pi*y_nodes));
% Ni2 = diag(2*ones(n1+1,1)) +diag(sin(2*pi*z_nodes));
% 
% Phi3 =diag(2*ones(n1+1,1)) + diag(sin(2*pi*x_nodes));
% Psi3 =diag(2*ones(n1+1,1)) + diag(sin(2*pi*y_nodes));
% Ni3 =diag(2*ones(n1+1,1)) + diag(sin(2*pi*z_nodes));

Phi1 = diag(2*ones(n1+1,1)); 
Psi1 = diag(2*ones(n1+1,1));
Ni1 = diag(2*ones(n1+1,1));

Phi2 =diag(2*ones(n1+1,1));
Psi2 =diag(2*ones(n1+1,1));
Ni2 = diag(2*ones(n1+1,1));

Phi3 =diag(2*ones(n1+1,1));
Psi3 =diag(2*ones(n1+1,1));
Ni3 =diag(2*ones(n1+1,1));

values1{1} = Phi1 * A1; values1{2} = Ni1; values1{3} = Psi1;
values2{1} = Phi2; values2{2} = Ni2 * A2; values2{3} = Psi2;
values3{1} = Phi3; values3{2} = Ni3; values3{3} = Psi3 * A3;

fprintf('Data ...\n')
m = 10;
[V1,H1] = arnoldi(values1{1},m,a);
[V2,H2] = arnoldi(values2{2},m,b);
[V3,H3] = arnoldi(values3{3},m,c);
n = m+1;
fprintf('Problems size: %d, %d\n', n, m)

% 1st factor
values1{1} = V1' * values1{1} * V1(:,1:end-1);
values1{2} = V2' * values1{2} * V2(:,1:end-1);
values1{3} = V3' * values1{3} * V3(:,1:end-1);
% 2nd factor
values2{1} = V1' * values2{1} * V1(:,1:end-1);
values2{2} = V2' * values2{2} * V2(:,1:end-1);
values2{3} = V3' * values2{3} * V3(:,1:end-1);
% 3rd factor
values3{1} = V1' * values3{1} * V1(:,1:end-1);
values3{2} = V2' * values3{2} * V2(:,1:end-1);
values3{3} = V3' * values3{3} * V3(:,1:end-1);


fprintf('Preconditioning ... \n')
values{1} = values1;
values{2} = values2;
values{3} = values3;
for i = 1:3
    [~, R1{i}] = qr(values1{i},0);
    [~, R2{i}] = qr(values2{i},0);
    [~, R3{i}] = qr(values3{i},0);
    %size(R1{i}),size(R2{i}),size(R3{i})

    % [cond(R1{i}), cond(R2{i}), cond(R3{i})],
    % pause,

    [~, ind] = min([cond(R1{i}), cond(R2{i}), cond(R3{i})]);
    rr = eval(['R',num2str(ind)]);
    %rr{i} = eye(size(R1{i}));
    new_values1{i} = values1{i}/rr{i};
    new_values2{i} = values2{i}/rr{i};
    new_values3{i} = values3{i}/rr{i};
end

new_values{1} = new_values1;
new_values{2} = new_values2;
new_values{3} = new_values3;

p =1;
addpath('tensor_toolbox-v3.6')
% Tucker RHS
[Qa,Ra] = qr(a,0);
[Qb,Rb] = qr(b,0);
[Qc,Rc] = qr(c,0);

K(:,:,1) = 1;
K = tensor(K);
K(:,:,1) = 1;

Rhs.a = Qa;
Rhs.b = Qb;
Rhs.c = Qc;
Rhs.F = ttm(K,{Ra, Rb, Rc});

% Creating the TT-format RHS
a = V1' * a;
b = V2' * b;
c = V3' * c;
rhs_vec = {a,b,c};
F = tt_tensor(rhs_vec);
%F = F/norm(F);


Params.tol = 1e-9;
Params.imax = 200;
Params.X0 = 0;
Params.tol_tr = 1e-8;
Params.r = 1000;

X = tt_zeros([m,m,m],3);

fprintf('TT-LSQR ... \n')
tic;
[X2, Res2] = TT_Tensorized_LSQR(values, F, Params, X);
t_tt = toc;

LX2 = OpL(values, X2);
final_res = norm(LX2-F);

fprintf('Final Relative Residual: %.4e\n', final_res / norm(F))
fprintf('Time: %d, Iterations: %d\n', t_tt, length(Res2.real_rel))

fprintf('PC-TT-LSQR ... \n')
tic;
[X3, Res3] = TT_Tensorized_LSQR(new_values, F, Params, X);
t_tt_pc = toc;
 
% for j = 1:X3.d
%     X3 = ttm(X3, j, inv(rr{j})');
% end

LX3 = OpL(new_values, X3);
final_res_pc = norm(LX3-F);

fprintf('Final Relative Residual: %.4e\n', final_res_pc / norm(F) )
fprintf('Time PC: %d, Iterations PC: %d\n', t_tt_pc, length(Res3.real_rel))


addpath('./ompbox10')

fprintf('OMP ...\n')

% building matrix D
t1 = kron(kron(values1{3}, values1{2}), values1{1});
t2 = kron(kron(values2{3}, values2{2}), values2{1});
t3 = kron(kron(values3{3}, values3{2}), values3{1});
DD = t1 + t2 + t3; %DD = DD/norm(DD);
ff = kron(kron(c,b),a);
x_esatta = DD\ff;
maxit = 50;
tol = 1e-5;
figure(1),
tic;
[x_omp,Lset] = omp_brandoni(ff,DD,maxit,tol);
t_omp = toc;


figure(321)
semilogy(Res2.real_rel,'-or'), hold on
semilogy(Res3.real_rel,'-*b')
legend('TT-LSQR', 'PC-TT-LSQR')
title('Relative Residuals')




%tic;
%[Y2, Res2pc, iter3] = TT_Tensorized_LSQR(new_values, F, Params, X);
%t_tt_prec = toc;

%{

[factors,core] = tt_tuck(X2,1e-9);

completeX2 = core;
for j = 1:X2.d
    completeX2 = ttm(completeX2, j, factors{j}');
end


%TX2 = ttm(X2.F, {X2.U1, X2.U2, X2.U3});
%wrk4 = ttm(TX2, values1); wrk5 = ttm(TX2, values2); wrk6 = ttm(TX2, values3);

% temp1 = X2; temp2 = X2; temp3 = X2;
% for k = 1:3
%     temp1 = ttm(temp1, k, values1{k}');
%     temp2 = ttm(temp2, k, values2{k}');
%     temp3 = ttm(temp3, k, values3{k}');
% end
% 
% coef1 = temp1 + temp2 + temp3;
% 
% for k = 1:3 
% 
%     wrk1 = ttm(coef1, k, values1{k});
%     wrk2 = ttm(coef1, k, values2{k});
%     wrk3 = ttm(coef1, k, values3{k});
%     F1 = ttm(F, k, values1{k});
%     F2 = ttm(F, k, values2{k});
%     F3 = ttm(F, k, values3{k});
% 
% end

%r_trunc_tensor = norm(F-temp1-temp2-temp3)/norm(F);
%r_trunc_tensor = norm((F1+F2+F3)-wrk1-wrk2-wrk3)/norm(F1+F2+F3);

% cores = X2.core;
% c1 = cores(1:m1^2);
% c2 = cores(m1^2+1:m1^2+m1^3);
% c3 = cores(m1^2+m1^3+1:end);
% c1 = reshape(c1,m1,m1);
% c2 = reshape(c2,m1,m1,m1);
% c3 = reshape(c3,m1,m1);
% tc2 = tt_tensor(c2);
% 
% norm(c1), norm(tc2), norm(c3)

cores = X2.core;
ps = X2.ps;
ranks = X2.r;
for k = 1:d
    C{k} = cores(ps(k):ps(k+1)-1);
    C{k} = reshape(C{k}, [ranks(k), M(k), ranks(k+1)]);
    fprintf('norm of core %d: %.4e\n', [k, norm(C{k},'fro')])
end


XX2 = reshape(full(X2), M);
p = n_sample;
if d == 3
    CoreX2{1} = full(XX2(1:p,1:p,1:p));
    for k = 2:d
        CoreX2{k} = full(XX2(p*(k-1)+1:p*k,p*(k-1)+1:p*k,p*(k-1)+1:p*k));
    end
elseif d == 4
    CoreX2{1} = XX2(1:p,1:p,1:p,1:p);
    for k = 2:d
        CoreX2{k} = XX2(p*(k-1)+1:p*k,p*(k-1)+1:p*k,p*(k-1)+1:p*k,p*(k-1)+1:p*k);
    end
end
    
for k = 1:length(CoreX2)
    fprintf('Norm of diagonal block relative to digit %d: %.4e\n',[k-1, norm(CoreX2{k},'fro')])
end

% c1 = XX2(1:P(1),1:P(2),1:P(3),1:P(4));
% c2 = XX2(P(1)+1:2*P(1),P(2)+1:2*P(2),P(3)+1:2*P(3),P(4)+1:2*P(4));
% c3 = XX2(2*P(1)+1:3*P(1),2*P(2)+1:3*P(2),2*P(3)+1:3*P(3),2*P(4)+1:3*P(4));
% norm(c1,'fro'),norm(c2,'fro'),norm(c3,'fro'),

%pause,
%
% J(:,:,1) = 1;
% J = tensor(J);
% J(:,:,1) = 1;
% 
% Y0.F = J;
% Y0.U1 = zeros(m1,p);
% Y0.U2 = zeros(m2,p);
% Y0.U3 = zeros(m3,p);
% 
% Params.tol_tr = 1e-10;
% 
% tic;
% %[X3, Res3] = Tensorized_LSQR_Trunc_3terms(values1, values2, values3, Rhs, Params, Y0);
% % [X3, Res3] = TT_Tensorized_LSQR_Trunc_3terms(values1, values2, values3, F, Params, X);
% t_tucker = toc;
% X3 = Y0;
% Res3 = Res2;
% completeX3 = ttm(X3.F, {X3.U1, X3.U2, X3.U3});
% wrk4 = ttm(completeX3, values1); wrk5 = ttm(completeX3, values2); wrk6 = ttm(completeX3, values3);
% r_trunc_tucker = norm(D-wrk4-wrk5-wrk6)/norm(D);

%{
addpath('./TTcore')
addpath('./TTrandomized')

% Creating a "rank 1" right-hand side

p = 1;

% a = ones(n1+1,1);
% b = ones(n2+1,1);
% c = ones(n3+1,1);

F_rand = cell(d,1);
F_rand{1} = a;
F_rand{2} = b;
F_rand{3} = c;


tol = 1e-6;
imax = 1000;
tol_tr = 1e-6;
r = 1000;

% X0 = tt_zeros([m1,m2,m3],3);
X0 = 0;
% defining the parameters
% Params.tol = 1e-7;
% Params.imax = 100;
% Params.X0 = X0;
% Params.tol_tr = 1e-6;
% Params.r = 1000;

% X = tt_zeros([m1,m2,m3],3);
X = cell(d,1);
for i = 1:d
    X{i} = zeros(1*M(i),1);
end

disp('rand')
terms = values;
tic;
[X3, Res3] = RandTT_Tensorized_LSQR(terms, F_rand, Params, X);
t_tensor_trunc5 = toc;

l = length(values); Nd = d; 
termsT = cell(l,1);

for i = 1:l
    termsT{i} = cell(Nd,1);
    for j = 1:Nd
        termsT{i}{j} = terms{i}{j}';
    end
end

LX = OperatorL(terms, X3);
LTLX = OperatorLT(termsT, LX);
LTF = OperatorLT(termsT, F_rand);
wrk{1} = LTLX; wrk{2} = LTF;
R_ne = TTsum_Randomize_then_Orthogonalize(wrk, [1; -1]);
r_trunc_tensor5 = TTnorm(R_ne) / TTnorm(LTF);

% figure(1)
% semilogy(Res2.real_ne_rel)
% 
% im1 = double(train0(79,:)');
% rhs2 = reshape(im1,28*28,1);
% 
% im1 = double(train1(79,:)');
% rhs3 = reshape(im1,28*28,1);
% 
% a = rhs2;
% b = rhs2;
% c = rhs2;
% G = cell(d,1);
% G{1} = a;
% G{2} = b;
% G{3} = c;
% 
% LTG = OperatorLT(termsT, G);
% wrk{1} = LTLX; wrk{2} = LTG;
% % R_ne = TTsum_Randomize_then_Orthogonalize(wrk, [1; -1]);
% R_ne2 = TTsum(wrk, [1; -1]);
% r_trunc_tensor2 = TTnorm(R_ne2) / TTnorm(LTG)
% 
% a = rhs3;
% b = rhs3;
% c = rhs3;
% H = cell(d,1);
% H{1} = a;
% H{2} = b;
% H{3} = c;
% 
% LTH = OperatorLT(termsT, H);
% wrk{1} = LTLX; wrk{2} = LTH;
% R_ne = TTsum_Randomize_then_Orthogonalize(wrk, [1; -1]);
% r_trunc_tensor3 = TTnorm(R_ne3) / TTnorm(LTH);



%}
%}