% test for 3D problem

% BORIS KhorOMSKij

clear all
close all


format shorte
format compact


% CAMBIA LE FUNZIONI

addpath('./Data')
load mnist_all
addpath('./TTcore')
addpath('./TTrandomized')

d = 3;

% set number of sample per digit
m1 = 20;
m2 = m1;
m3 = m1;

M(1) = m1; M(2) = m2; M(3) = m3;


im0 = double(train2(58,:)');
rhs1 = reshape(im0,28*28,1);

for k = 1:m1

    im0 = double(train0(1*k,:)');
    Phi1(:,k) = reshape(im0,28^2,1);
    im1 = double(train1(1*k,:)');
    Psi1(:,k) = reshape(im1,28^2,1);
    im2 = double(train2(1*k,:)');
    Ni1(:,k) = reshape(im2,28^2,1);

end

for k = 1:m2

    im0 = double(train0(50*k,:)');
    Phi2(:,k) = reshape(im0,28^2,1);
    im1 = double(train1(50*k,:)');
    Psi2(:,k) = reshape(im1,28^2,1);
    im2 = double(train2(50*k,:)');
    Ni2(:,k) = reshape(im2,28^2,1);

end

for k = 1:m2

    im0 = double(train0(20*k,:)');
    Phi3(:,k) = reshape(im0,28^2,1);
    im1 = double(train1(20*k,:)');
    Psi3(:,k) = reshape(im1,28^2,1);
    im2 = double(train2(20*k,:)');
    Ni3(:,k) = reshape(im2,28^2,1);

end



n1 = size(Phi1,1); n2 = size(Phi2,1); n3 = size(Phi3,1);
n = n1; m = m1;
N(1) = n; N(2) = n; N(3) = n;
M(1) = m; M(2) = m; M(3) = m;

% values1{1} = Phi1(1:n1+1,1:m1+1); values1{2} = Ni1(1:n1+1,1:m1+1); values1{3} = Psi1(1:n1+1,1:m1+1);
% values2{1} = Phi2(1:n2+1,1:m2+1); values2{2} = Ni2(1:n2+1,1:m2+1); values2{3} = Psi2(1:n2+1,1:m2+1);
% values3{1} = Phi3(1:n3+1,1:m3+1); values3{2} = Ni3(1:n3+1,1:m3+1); values3{3} = Psi3(1:n3+1,1:m3+1);

values1{1} = Phi1; values1{2} = Ni1; values1{3} = Psi1;
values2{1} = Phi2; values2{2} = Ni2; values2{3} = Psi2;
values3{1} = Phi3; values3{2} = Ni3; values3{3} = Psi3;
terms = cell(d,1);
terms{1} = values1;
terms{2} = values2;
terms{3} = values3;

% Creating a "rank 1" right-hand side

p = 1;

% a = ones(n1+1,1);
% b = ones(n2+1,1);
% c = ones(n3+1,1);

a = rhs1;
b = rhs1;
c = rhs1;
F = cell(d,1);
F{1} = a;
F{2} = b;
F{3} = c;

% Se i=6 allora risolutore diretto non funziona più
% gli altri risolutori vanno meglio invece

% Tucker RHS
[Qa,Ra] = qr(a,0);
[Qb,Rb] = qr(b,0);
[Qc,Rc] = qr(c,0);

K(:,:,1) = 1;
K = tensor(K);
K(:,:,1) = 1;

% Rhs.a = Qa;
% Rhs.b = Qb;
% Rhs.c = Qc;
% Rhs.F = ttm(K,{Ra, Rb, Rc});

% Creating the TT-format RHS
%F = tt_tensor({a,b,c});

for k = 1:n3

    D(:,:,k) = a*b'*c(k);

end

D = sptensor(D);

tol = 1e-6;
imax = 1000;
tol_tr = 1e-6;
r = 1000;

% X0 = tt_zeros([m1,m2,m3],3);
X0 = 0;
% defining the parameters
Params.tol = 1e-7;
Params.imax = 1000;
Params.X0 = X0;
Params.tol_tr = 1e-6;
Params.r = 100;

% X = tt_zeros([m1,m2,m3],3);
X = cell(d,1);
for i = 1:d
    X{i} = zeros(1*M(i),1);
end

%flag = 'random';
flag = 'classi';

tic;
[X2, Res2] = RandTTlsqr(terms, F, Params, X, flag);
t_tensor_trunc = toc;

l = length(terms); Nd = length(X2); 
termsT = cell(l,1);

for i = 1:l
    termsT{i} = cell(Nd,1);
    for j = 1:Nd
        termsT{i}{j} = terms{i}{j}';
    end
end

LX = OperatorL(terms, X2, flag);
LTLX = OperatorLT(termsT, LX, flag);
LTF = OperatorLT(termsT, F, flag);
wrk{1} = LTLX; wrk{2} = LTF;
if flag == 'random'
    R_ne = TTsum_Randomize_then_Orthogonalize(wrk, [1; -1]);
else
    R_ne = TTsum(wrk, [1; -1]);
end
r_trunc_tensor = TTnorm(R_ne) / TTnorm(LTF);

figure(1)
semilogy(Res2.real_ne_rel)

im1 = double(train0(58,:)');
rhs2 = reshape(im1,28*28,1);

im1 = double(train1(58,:)');
rhs3 = reshape(im1,28*28,1);

a = rhs2;
b = rhs2;
c = rhs2;
G = cell(d,1);
G{1} = a;
G{2} = b;
G{3} = c;

LTG = OperatorLT(termsT, G, flag);
wrk{1} = LTLX; wrk{2} = LTG;
if flag == 'random'
    R_ne2 = TTsum_Randomize_then_Orthogonalize(wrk, [1; -1]);
else
    R_ne2 = TTsum(wrk, [1; -1]);
end
r_trunc_tensor2 = TTnorm(R_ne2) / TTnorm(LTG)

a = rhs3;
b = rhs3;
c = rhs3;
H = cell(d,1);
H{1} = a;
H{2} = b;
H{3} = c;

LTH = OperatorLT(termsT, H, flag);
wrk{1} = LTLX; wrk{2} = LTH;
if flag == 'random'
    R_ne3 = TTsum_Randomize_then_Orthogonalize(wrk, [1; -1]);
else
    R_ne3 = TTsum(wrk, [1; -1]);
end
r_trunc_tensor3 = TTnorm(R_ne3) / TTnorm(LTH)

%TX2 = ttm(X2.F, {X2.U1, X2.U2, X2.U3});
%wrk4 = ttm(TX2, values1); wrk5 = ttm(TX2, values2); wrk6 = ttm(TX2, values3);
% 
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
% c1 = cores(1:m^2);
% c2 = cores(m^2+1:m^2+m^3);
% c3 = cores(m^2+m^3+1:end);
% c1 = reshape(c1,m,m);
% c2 = reshape(c2,m,m,m);
% c3 = reshape(c3,m,m);
% tc2 = tt_tensor(c2);
% 
% norm(c1), norm(tc2), norm(c3)


% J(:,:,1) = 1;
% J = tensor(J);
% J(:,:,1) = 1;
% 
% Y0.F = J;
% Y0.U1 = zeros(m1+1,p);
% Y0.U2 = zeros(m2+1,p);
% Y0.U3 = zeros(m3+1,p);
% 
% Params.tol_tr = 1e-12;
% 
% tic;
% [X3, Res3] = Tensorized_LSQR_Trunc_3terms(values1, values2, values3, Rhs, Params, Y0);
% [X3, Res3] = TT_Tensorized_LSQR_Trunc_3terms(values1, values2, values3, F, Params, X);
% t_tucker = toc;
% completeX3 = ttm(X3.F, {X3.U1, X3.U2, X3.U3});
% wrk4 = ttm(completeX3, values1); wrk5 = ttm(completeX3, values2); wrk6 = ttm(completeX3, values3);
% r_trunc_tucker = norm(D-wrk4-wrk5-wrk6)/norm(D);

% temp4 = X3; temp5 = X3; temp6 = X3;
% for k = 1:3
%     temp4 = ttm(temp4, k, values1{k}');
%     temp5 = ttm(temp5, k, values2{k}');
%     temp6 = ttm(temp6, k, values3{k}');
% end
% r_trunc_tucker = norm(F-temp4-temp5-temp6)/norm(F);

% % solver 
% % tic;
% AA = kron( kron(sparse(Psi1),sparse(Ni1)), sparse(T1)) + kron( kron(sparse(T2),sparse(Ni2)), sparse(Phi2))...
%    + kron( kron(sparse(Psi3),sparse(T3)), sparse(Phi3));
% ff = sparse(kron(kron(c,b), a));
% % xx = AA\ff;
% % %AA = zeros(n1,m1); ff = zeros(n1,1); xx = zeros(m1,1);
% % t_dir_solv = toc
% % n_dir_solv = norm(AA*xx-ff)
% [xx2, flag, relres, iter, resvec] = pcg(AA, ff, 1e-6, 200);


