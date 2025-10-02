    % test for 3D problem

% BORIS KhorOMSKij

clear all
%close all
%clc

format shorte
format compact

N = []; M = [];

n1 = 30; N = [N, n1]; %m1 = 20; M = [M, m1];
n2 = 30; N = [N, n2]; %m2 = 20; M = [M, m2];
n3 = 30; N = [N, n3]; %m3 = 20; M = [M, m3];
n4 = 30; N = [N, n4]; %m4 = 20; M = [M, m4];

d = 3;
n_sample = 4;
for k = 1:d
    M(k) = n_sample*d;
    M(k) = 20;
end


%N_sample = 6 per 2 termini (m = 18) Risultati: 
% 285 secondi ( no prec )
% 76 secondi ( prec )

%N_sample = 4 per 3 termini (m = 12) Risultati:
% 195 secondi ( no prec )
% 29 secondi ( prec )


% n-sample = 12 2 termini (m = 36)
% 1592 secondi ( no prec )
% neanche converge


% n-sample = 8 3 termini (m = 24)
% 1422 secondi ( no prec )
% 765 secondi ( no prec )


h1 = 1/n1;
h2 = 1/n2;
h3 = 1/n3;

x_nodes = linspace(0,1,n1)';
y_nodes = linspace(0,1,n2)';
z_nodes = linspace(0,1,n3)';

% building the matrices (n1+1) x (n1+1), (n2+1) x (n2+1), (n3+1) x (n3+1)

% T1 = (-1/h1^2)*( diag([-1; -2*ones(n1-1,1); -1])+diag([0; ones(n1-1,1)],1)+diag([ones(n1-1,1); 0],-1) );        
% T2 = (-1/h2^2)*( diag([-1; -2*ones(n2-1,1); -1])+diag([0; ones(n2-1,1)],1)+diag([ones(n2-1,1); 0],-1) );
% T3 = (-1/h3^2)*( diag([-1; -2*ones(n3-1,1); -1])+diag([0; ones(n3-1,1)],1)+diag([ones(n3-1,1); 0],-1) );
% 
% % addpath('./Data')
% load mnist_all
% 
% n_sample = 20;
% n_pix = 2;
% n_pix_rhs = 1;
% dim_patch = 2;
% digit = 2;
% [H1,H2,I1,I2,rhs1,rhs2] = dict_setup(n_sample,dim_patch,digit);
% rhs = rhs1*rhs2';
% T1 = H1;
% T2 = H2;
% [H1,H2,I1,I2,rhs1,rhs2] = dict_setup(n_sample,dim_patch,digit);
% T3 = H1;
% 
% n1 = size(H1,1); N = [N, n1+1]; m1 = size(H1,2); M = [M, m1+1];
% n2 = size(H1,1); N = [N, n2+1]; m2 = size(H1,2); M = [M, m2+1];
% n3 = size(H1,1); N = [N, n3+1]; m3 = size(H1,2); M = [M, m3+1];
% 
% x_nodes = linspace(0,1,n1)';
% y_nodes = linspace(0,1,n2)';
% z_nodes = linspace(0,1,n3)';
% n1

% CAMBIA LE FUNZIONI

addpath('./Data')
load mnist_all
addpath('./oseledets_TT-Toolbox')


% e = ones(n1+1,1);
% T1 = spdiags([e -2*e e],-1:1,n1+1,n1+1);
% e = ones(n2+1,1);
% T2 = spdiags([e -2*e e],-1:1,n2+1,n2+1);
% e = ones(n3+1,1);
% T3 = spdiags([e -2*e e],-1:1,n3+1,n3+1);

%T1 = full(T1); T2 = full(T2); T3= full(T3);
% CAMBIA LE FUNZIONI
% Phi1 = diag(2*ones(n1+1,1)) + diag(sin(2*pi*x_nodes));
% Psi1 = diag(2*ones(n1+1,1)) +diag(sin(2*pi*y_nodes));
% Ni1 = diag(2*ones(n1+1,1)) + diag(sin(2*pi*z_nodes));

% im0 = double(train0(79,:)');
% rhs1 = reshape(im0,28*28,1);
% 
% for k = 1:m1
% 
%     im0 = double(train0(1*k,:)');
%     Phi1(:,k) = reshape(im0,28^2,1);
%     im1 = double(train1(1*k,:)');
%     Psi1(:,k) = reshape(im1,28^2,1);
%     im2 = double(train2(1*k,:)');
%     Ni1(:,k) = reshape(im2,28^2,1);
% 
% end
% 
% % Phi2 =diag(2*ones(n1+1,1)) + diag(sin(2*pi*x_nodes));
% % Psi2 =diag(2*ones(n1+1,1)) + diag(sin(2*pi*y_nodes));
% % Ni2 = diag(2*ones(n1+1,1)) +diag(sin(2*pi*z_nodes));
% 
% for k = 1:m2
% 
%     im0 = double(train0(100*k,:)');
%     Phi2(:,k) = reshape(im0,28^2,1);
%     im1 = double(train1(100*k,:)');
%     Psi2(:,k) = reshape(im1,28^2,1);
%     im2 = double(train2(100*k,:)');
%     Ni2(:,k) = reshape(im2,28^2,1);
% 
% end
% 
% % Phi3 =diag(2*ones(n1+1,1)) + diag(sin(2*pi*x_nodes));
% % Psi3 =diag(2*ones(n1+1,1)) + diag(sin(2*pi*y_nodes));
% % Ni3 =diag(2*ones(n1+1,1)) + diag(sin(2*pi*z_nodes));
% 
% for k = 1:m2
% 
%     im0 = double(train0(200*k,:)');
%     Phi3(:,k) = reshape(im0,28^2,1);
%     im1 = double(train1(200*k,:)');
%     Psi3(:,k) = reshape(im1,28^2,1);
%     im2 = double(train2(200*k,:)');
%     Ni3(:,k) = reshape(im2,28^2,1);
% 
% end


im0 = double(test1(randi(1000),:)');
rhs1 = reshape(im0,28*28,1);
rhs1 = rhs1/norm(rhs1);


% Scomment for particular case
%{
for k = 1:M(1)

    im0 = double(train0(1*k,:)');
    Phi1(:,k) = reshape(im0,28^2,1);
    im1 = double(train1(1*k,:)');
    Psi1(:,k) = reshape(im1,28^2,1);
    im2 = double(train8(1*k,:)');
    Ni1(:,k) = reshape(im2,28^2,1);
    im3 = double(train3(1*k,:)');
    Gamma1(:,k) = reshape(im3,28^2,1);

    Phi1(:,k) = Phi1(:,k)/norm(Phi1(:,k));
    Ni1(:,k) = Ni1(:,k)/norm(Ni1(:,k));
    Psi1(:,k) = Psi1(:,k)/norm(Psi1(:,k));
    Gamma1(:,k) = Gamma1(:,k)/norm(Gamma1(:,k));

end

% 50 buoni risultati
for k = 1:M(2)

    im0 = double(train0(47*k,:)');
    Phi2(:,k) = reshape(im0,28^2,1);
    im1 = double(train1(47*k,:)');
    Psi2(:,k) = reshape(im1,28^2,1);
    im2 = double(train8(47*k,:)');
    Ni2(:,k) = reshape(im2,28^2,1);
    im3 = double(train3(47*k,:)');
    Gamma2(:,k) = reshape(im3, 28^2, 1);

    Phi2(:,k) = Phi2(:,k)/norm(Phi2(:,k));
    Ni2(:,k) = Ni2(:,k)/norm(Ni2(:,k));
    Psi2(:,k) = Psi2(:,k)/norm(Psi2(:,k));
    Gamma2(:,k) = Gamma2(:,k)/norm(Gamma2(:,k));

end

for k = 1:M(3)

    im0 = double(train0(20*k,:)');
    Phi3(:,k) = reshape(im0,28^2,1);
    im1 = double(train1(20*k,:)');
    Psi3(:,k) = reshape(im1,28^2,1);
    im2 = double(train8(20*k,:)');
    Ni3(:,k) = reshape(im2,28^2,1);
    im3 = double(train3(20*k,:)');
    Gamma3(:,k) = reshape(im3,28^2,1);

    Phi3(:,k) = Phi3(:,k)/norm(Phi3(:,k));
    Ni3(:,k) = Ni3(:,k)/norm(Ni3(:,k));
    Psi3(:,k) = Psi3(:,k)/norm(Psi3(:,k));
    Gamma3(:,k) = Gamma3(:,k)/norm(Gamma3(:,k));

end
%}

% Random Choice
%%{
for k = 1:M(1)

    im0 = double(train0(randi([1,2000]),:)');
    Phi1(:,k) = reshape(im0,28^2,1);
    im1 = double(train1(randi([1,2000]),:)');
    Psi1(:,k) = reshape(im1,28^2,1);
    im2 = double(train8(randi([1,2000]),:)');
    Ni1(:,k) = reshape(im2,28^2,1);
    im3 = double(train3(randi([1,2000]),:)');
    Gamma1(:,k) = reshape(im3,28^2,1);

    Phi1(:,k) = Phi1(:,k)/norm(Phi1(:,k));
    Ni1(:,k) = Ni1(:,k)/norm(Ni1(:,k));
    Psi1(:,k) = Psi1(:,k)/norm(Psi1(:,k));
    Gamma1(:,k) = Gamma1(:,k)/norm(Gamma1(:,k));

end

for k = 1:M(2)

    im0 = double(train0(randi([1,2000]),:)');
    Phi2(:,k) = reshape(im0,28^2,1);
    im1 = double(train1(randi([1,2000]),:)');
    Psi2(:,k) = reshape(im1,28^2,1);
    im2 = double(train8(randi([1,2000]),:)');
    Ni2(:,k) = reshape(im2,28^2,1);
    im3 = double(train3(randi([1,2000]),:)');
    Gamma2(:,k) = reshape(im3, 28^2, 1);

    Phi2(:,k) = Phi2(:,k)/norm(Phi2(:,k));
    Ni2(:,k) = Ni2(:,k)/norm(Ni2(:,k));
    Psi2(:,k) = Psi2(:,k)/norm(Psi2(:,k));
    Gamma2(:,k) = Gamma2(:,k)/norm(Gamma2(:,k));

end

for k = 1:M(3)

    im0 = double(train0(randi([1,2000]),:)');
    Phi3(:,k) = reshape(im0,28^2,1);
    im1 = double(train1(randi([1,2000]),:)');
    Psi3(:,k) = reshape(im1,28^2,1);
    im2 = double(train8(randi([1,2000]),:)');
    Ni3(:,k) = reshape(im2,28^2,1);
    im3 = double(train3(randi([1,2000]),:)');
    Gamma3(:,k) = reshape(im3,28^2,1);

    Phi3(:,k) = Phi3(:,k)/norm(Phi3(:,k));
    Ni3(:,k) = Ni3(:,k)/norm(Ni3(:,k));
    Psi3(:,k) = Psi3(:,k)/norm(Psi3(:,k));
    Gamma3(:,k) = Gamma3(:,k)/norm(Gamma3(:,k));

end
%}

n1 = size(Phi1,1); n2 = size(Phi2,1); n3 = size(Phi3,1); n4 = n3;


% e = ones(n1+1,1);
% T1 = spdiags([e -2*e e],-1:1,n1+1,n1+1);
% e = ones(n2+1,1);
% T2 = spdiags([e -2*e e],-1:1,n2+1,n2+1);
% e = ones(n3+1,1);
% T3 = spdiags([e -2*e e],-1:1,n3+1,n3+1);
% 
% T1 = full(T1); T2 = full(T2); T3= full(T3);
% 
% x_nodes = linspace(0,1,n1+1)';
% y_nodes = linspace(0,1,n2+1)';
% z_nodes = linspace(0,1,n3+1)';




values1{1} = Phi1; values1{2} = Psi1; values1{3} = Ni1;
values2{1} = Phi2; values2{2} = Psi2; values2{3} = Ni2;
values3{1} = Phi3; values3{2} = Psi3; values3{3} = Ni3;

%{
P = M/d;

if d == 3
    Phi1_2 = [Phi1(:,1:P(1)), Ni1(:,1:P(2)), Psi1(:,1:P(3))];
    Ni1_2 = [Phi1(:,P(1)+1:2*P(1)), Ni1(:,P(2)+1:2*P(2)), Psi1(:,P(3)+1:2*P(3))];
    Psi1_2 = [Phi1(:,2*P(1)+1:3*P(1)), Ni1(:,2*P(2)+1:3*P(2)), Psi1(:,2*P(3)+1:3*P(3))];
elseif d == 4
    Phi1_2 = [Phi1(:,1:P(1)), Ni1(:,1:P(2)), Psi1(:,1:P(3)), Gamma1(:,1:P(4))];
    Ni1_2 = [Phi1(:,P(1)+1:2*P(1)), Ni1(:,P(2)+1:2*P(2)), Psi1(:,P(3)+1:2*P(3)), Gamma1(:,P(4)+1:2*P(4))];
    Psi1_2 = [Phi1(:,2*P(1)+1:3*P(1)), Ni1(:,2*P(2)+1:3*P(2)), Psi1(:,2*P(3)+1:3*P(3)), Gamma1(:,2*P(4)+1:3*P(4))];
    Gamma1_2 = [Phi1(:,3*P(1)+1:4*P(1)), Ni1(:,3*P(2)+1:4*P(2)), Psi1(:,3*P(3)+1:4*P(3)), Gamma1(:,3*P(4)+1:4*P(4))];
end

if d == 3
    Phi2_2 = [Phi2(:,1:P(1)), Ni2(:,1:P(2)), Psi2(:,1:P(3))];
    Ni2_2 = [Phi2(:,P(1)+1:2*P(1)), Ni2(:,P(2)+1:2*P(2)), Psi2(:,P(3)+1:2*P(3))];
    Psi2_2 = [Phi2(:,2*P(1)+1:3*P(1)), Ni2(:,2*P(2)+1:3*P(2)), Psi2(:,2*P(3)+1:3*P(3))];
elseif d == 4
    Phi2_2 = [Phi2(:,1:P(1)), Ni2(:,1:P(2)), Psi2(:,1:P(3)), Gamma2(:,1:P(4))];
    Ni2_2 = [Phi2(:,P(1)+1:2*P(1)), Ni2(:,P(2)+1:2*P(2)), Psi2(:,P(3)+1:2*P(3)), Gamma2(:,P(4)+1:2*P(4))];
    Psi2_2 = [Phi2(:,2*P(1)+1:3*P(1)), Ni2(:,2*P(2)+1:3*P(2)), Psi2(:,2*P(3)+1:3*P(3)), Gamma2(:,2*P(4)+1:3*P(4))];
    Gamma2_2 = [Phi2(:,3*P(1)+1:4*P(1)), Ni2(:,3*P(2)+1:4*P(2)), Psi2(:,3*P(3)+1:4*P(3)), Gamma2(:,3*P(4)+1:4*P(4))];
end

if d == 3
    Phi3_2 = [Phi3(:,1:P(1)), Ni3(:,1:P(2)), Psi3(:,1:P(3))];
    Ni3_2 = [Phi3(:,P(1)+1:2*P(1)), Ni3(:,P(2)+1:2*P(2)), Psi3(:,P(3)+1:2*P(3))];
    Psi3_2 = [Phi3(:,2*P(1)+1:3*P(1)), Ni3(:,2*P(2)+1:3*P(2)), Psi3(:,2*P(3)+1:3*P(3))];
elseif d == 4
    Phi3_2 = [Phi3(:,1:P(1)), Ni3(:,1:P(2)), Psi3(:,1:P(3)), Gamma3(:,1:P(4))];
    Ni3_2 = [Phi3(:,P(1)+1:2*P(1)), Ni3(:,P(2)+1:2*P(2)), Psi3(:,P(3)+1:2*P(3)), Gamma3(:,P(4)+1:2*P(4))];
    Psi3_2 = [Phi3(:,2*P(1)+1:3*P(1)), Ni3(:,2*P(2)+1:3*P(2)), Psi3(:,2*P(3)+1:3*P(3)), Gamma3(:,2*P(4)+1:3*P(4))];
    Gamma3_2 = [Phi3(:,3*P(1)+1:4*P(1)), Ni3(:,3*P(2)+1:4*P(2)), Psi3(:,3*P(3)+1:4*P(3)), Gamma3(:,3*P(4)+1:4*P(4))];
end
%}


% values1{1} = Phi1_2; values1{2} = Ni1_2; values1{3} = Psi1_2; 
% values2{1} = Phi2_2; values2{2} = Ni2_2; values2{3} = Psi2_2; 
% values3{1} = Phi3_2; values3{2} = Ni3_2; values3{3} = Psi3_2;


% valuesi{j} = [00000 11111 88888] = QR
% D = values1 + values2 + values3
% values2 = Q R  
s = 4*M(1);
tic;
% Omega = randn(s,n1)/sqrt(s);
% Omega = sparse_sign_backup(s,n1,8);
for i = 1:3
    [~, R1{i}] = qr(values1{i},0);
    [~, R2{i}] = qr(values2{i},0);
    [~, R3{i}] = qr(values3{i},0);


    % [cond(R1{i}), cond(R2{i}), cond(R3{i})],
    % pause,

    [m, ind] = min([cond(R1{i}), cond(R2{i}), cond(R3{i})]);
    %[m, ind] = min([cond(R1{i}), cond(R2{i})]);
    rr = eval(['R',num2str(ind)]);
    
    % rr = R2;
    new_values1{i} = values1{i}/rr{i};
    new_values2{i} = values2{i}/rr{i};
    new_values3{i} = values3{i}/rr{i};
end
t_creat_prec = toc;



new_values{1} = new_values1;
new_values{2} = new_values2;
new_values{3} = new_values3; 



if d == 4
    values1{4} = Gamma1_2;
    values2{4} = Gamma2_2;
    values3{4} = Gamma3_2;
end

values{1} = values1;
values{2} = values2;
values{3} = values3;

% Creating a "rank 1" right-hand side

p = 1;

% a = rand(n1+1,p); % a = a/norm(a);
% b = rand(n2+1,p); % b = b/norm(b);
% c = rand(n3+1,p); % c = c/norm(c);

% a = sin(2*pi*x_nodes);
% b = sin(2*pi*y_nodes);
% c = sin(2*pi*z_nodes);

% a = 10 + x_nodes.^2;
% b = 10 + y_nodes.^2;
% c = 10 + sin(2*pi*z_nodes);

% a = ones(n1+1,1);
% b = ones(n2+1,1);
% c = ones(n3+1,1);

a = rhs1;
b = rhs1;
c = rhs1;


% for i=1:p
% 
%     i = 3;
%     a = 10^i + x_nodes.^i;
%     b = 10^i + y_nodes.^i;
%     c = 10^i + sin(2*pi*z_nodes);
% 
% end

% Se i=6 allora risolutore diretto non funziona più
% gli altri risolutori vanno meglio invece


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
if d == 3
    rhs_vec = {a,b,c};
elseif d == 4
    d = a;
    rhs_vec = {a,b,c,d};
end

F = tt_tensor(rhs_vec);
%F = F/norm(F);

for k = 1:n3

    D(:,:,k) = a*b'*c(k);

end



D = sptensor(D);
X0 = tensor(zeros(M));
Params.tol = 1e-12;
Params.imax = 1000;
Params.X0 = X0;
Params.tol_tr = 1e-8;
Params.r = 1000;
X_tuck = tensor(zeros(M));

tic;
% [X1, Res1] = Tensorized_LSQR(values1, values2, D, Params);
%[X1, Res1] = Tensorized_LSQR_Trunc_3terms(values1, values2, values3, Rhs, Params, X_tuck);
% X1 = sptensor(zeros(m1+1,m2+1,m3+1));
Res1.real_abs=0;
Res1.est_abs=0;
X1 = tensor(zeros(M));
t_tensor = toc;

% Clsqr.A = kron(Ni1(1:n1+1,1:m1+1), T1(1:n1+1,1:m1+1));
% Clsqr.F = Psi1(1:n1+1,1:m1+1)';
% Clsqr.E = kron(Ni2(1:n2+1,1:m2+1), Phi2(1:n2+1,1:m2+1));
% Clsqr.B = (T2(1:n1+1,1:m2+1)');
% Clsqr.J = kron(T3(1:n3+1,1:m3+1), Phi3(1:n3+1,1:m3+1));
% Clsqr.K = Psi3(1:n3+1,1:m3+1)';
% Clsqr.C1 = kron(b, a);
% Clsqr.C2 = c;

% tol = 1e-6;
% imax = 1000;
% tol_tr = 1e-9;
% r = 1000;

% tic;
% %[X_1,X_2,r_res,a_res,rks,DD] = lsqr_gen_trunc_3terms(Clsqr,tol,imax,tol_tr,r);
% % r_res residuo relativo vero
% t_lsqr_matrix = toc;
%  X_1 = zeros(m1,1); X_2 = X_1;
%  r_res = zeros(imax,1);
%  a_res= zeros(imax,1);

% wrk1 = [Clsqr.A*X_1, Clsqr.E*X_1, Clsqr.J*X_1, -Clsqr.C1];
% wrk2 = [Clsqr.F'*X_2, Clsqr.B'*X_2, Clsqr.K'*X_2, Clsqr.C2];
% r_matrix_lsqr = norm(wrk1*wrk2','fro')/norm(Clsqr.C1*Clsqr.C2','fro');
%r_matrix_lsqr=1;



% defining the parameters
% Params.tol = 1e-9;
% Params.imax = 1000;
% Params.X0 = X0;
% Params.tol_tr = 1e-8;
% Params.r = 1000;
% Creating initial guess in Tucker form: ( for the truncated version )
% F(:,:,1) = 1;
% F = tensor(F);
% F(:,:,1) = 1;

X = tt_zeros(M,d);


tic;
%[X2, Res2, iter2] = TT_Tensorized_LSQR(values, F, Params, X);
t_tt_noprec = toc;
X2 = X;

tic;
[Y2, Res2pcQr, iter3] = TT_Tensorized_LSQR(new_values, F, Params, X);
t_tt_prec = toc;

for j = 1:Y2.d
    Y2 = ttm(Y2, j, inv(rr{j})');
end

% Y2   || a   -   D mode_j Y2 ||
%     n x n x n     m x n x m


LY = OpL(values, Y2);
ly2 = reshape(full(LY),[784,784,784]);
ty = tensor(ly2);

cores = LY.core;
rks = LY.r;
d = LY.d;
pos = LY.ps;
m_y = LY.n;
K1 = cores(pos(1):pos(2)-1);
K2 = cores(pos(2):pos(3)-1);
K3 = cores(pos(3):pos(4)-1);
K1 = reshape(K1,m_y(1),rks(2));
K2 = reshape(K2,rks(2),m_y(2),rks(3));
K3 = reshape(K3,rks(3),m_y(3));
[u1,s1,v1] = svd(K1,0);
[u3,s3,v3]=svd(K3',0);
U3 = u3; U1 = u1; 
U2= nvecs(tensor(K2),2,54);
U1=U1/norm(U1);U2=U2/norm(U2);U3=U3/norm(U3);
plot(abs(a'*U1)), hold on, plot(abs(a'*U2)), plot(abs(a'*U3))
svd(a'*orth(U1)), svd(a'*orth(U2)), svd(a'*orth(U3))

% y2 = reshape(full(Y2),[M(1),M(2),M(3)]);
% ty = tensor(y2);
% Ricorda che valusa ha immagini come colonne
% ty = tensor(G2);
U1 = nvecs(ty,1,1); %<-- Mode 1
U2 = nvecs(ty,2,1); %<-- Mode 2
U3 = nvecs(ty,3,1); %<-- Mode 3

% angles
angle_1 = abs(a' * U1);
angle_2 = abs(a' * U2);
angle_3 = abs(a' * U3);
fprintf('Digit 0: %e, Digit 1: %e, Digit 8: %e\n', angle_1,...
    angle_2, angle_3)
% [a'*U1, a'*U2, a'*U3],

% disp('class 1st mode')
% [svd(U1' * orth(values1{1})),...
% svd(U1' * orth(values1{2})),...
% svd(U1' * orth(values1{3}))]
% 
% disp('class 2nd mode')
% [svd(U2' * orth(values1{1})),...
% svd(U2' * orth(values1{2})),...
% svd(U2' * orth(values1{3}))]
% 
% disp('class 3rd mode')
% [svd(U3' * orth(values1{1})),...
% svd(U3' * orth(values1{2})),...
% svd(U3' * orth(values1{3}))]


[factors,core] = tt_tuck(X2,1e-9);

completeX2 = core;
for j = 1:X2.d
    completeX2 = ttm(completeX2, j, factors{j}');
end

% Y2 = OpL(values, Y2);
cores = Y2.core;
rks = Y2.r;
d = Y2.d;
pos = Y2.ps;
m_y = Y2.n;
G1 = cores(pos(1):pos(2)-1);
G2 = cores(pos(2):pos(3)-1);
G3 = cores(pos(3):pos(4)-1);
% size(G1),size(G2),size(G3)
G1 = reshape(G1,m_y(1),rks(2));
G2 = reshape(G2,rks(2),m_y(2),rks(3));
G3 = reshape(G3,rks(3),m_y(3));
% G1 = reshape(G1, 784, rks(2));
% G2 = reshape(G2, rks(2), 784, rks(3));
% G3 = reshape(G3, rks(3), 784);
[u3,s3,v3] = svd(G3');
[u1,s1,v1] = svd(G1);
ty = tensor(G2);
%U1 = nvecs(ty,1,1); %<-- Mode 1
U2 = nvecs(ty,2,1); %<-- Mode 2
%U3 = nvecs(ty,3,1); %<-- Mode 3
%u1 = u1(:,1); u3 = u3(:,1);
U1 = u1(:,1); U3 = u3(:,1);
UU1 = U1/norm(U1);
UU2 = U2/norm(U2);
UU3 = U3/norm(U3);
UU = tt_tensor({UU1,UU2,UU3});


luu = OpL(values, UU);

cores_luu = luu.core;
rks_luu = luu.r;
d_luu = luu.d;
pos_luu = luu.ps;
P1 = cores_luu(pos_luu(1):pos_luu(2)-1);
P2 = cores_luu(pos_luu(2):pos_luu(3)-1);
P3 = cores_luu(pos_luu(3):pos_luu(4)-1);
P1 = reshape(P1,784,rks_luu(2));
P2 = reshape(P2, rks_luu(2), 784, rks_luu(3));
P3 = reshape(P3, rks_luu(3), 784);
[u1,s1,v1] = svd(P1,0); UU1 = u1;%UU1 = u1(:,1); %UU1 = U1/norm(U1);
[u3,s3,v3] = svd(P3',0); UU3 = u3;%UU3 = u3(:,1); %UU3 = U3/norm(U3);
UU2 = nvecs(tensor(P2),2,3); %UU2 = U2/norm(U2);

% questo non va benissimo :(((
% UU1 = values1{1}*U1+values2{1}*U1+values3{1}*U1;
% UU2 = values1{2}*U2+values2{2}*U2+values3{2}*U2;
% UU3 = values1{3}*U3+values2{3}*U3+values3{3}*U3;
UU1 = UU1/norm(UU1);UU2 = UU2/norm(UU2);UU3 = UU3/norm(UU3);


% abs(a'*UU1),abs(a'*UU2),abs(a'*UU3)
dist_1 = svd(a' * orth(UU1));
dist_2 = svd(a' * orth(UU2));
dist_3 = svd(a' * orth(UU3));
fprintf('Dist 0: %e, Dist 1: %e, Dist 8: %e\n', dist_1,...
    dist_2, dist_3)
% svd(a'*orth(UU1)), svd(a'*orth(UU2)), svd(a'*orth(UU3))


addpath('./ompbox10')

fprintf('OMP ...\n')

% building matrix D
%t1 = kron(kron(values1{3}, values1{2}), values1{1});
%t2 = kron(kron(values2{3}, values2{2}), values2{1});
%t3 = kron(kron(values3{3}, values3{2}), values3{1});
%DD = t1 + t2 + t3; DD = DD/norm(DD);
%ff = kron(kron(c,b),a);
%x_esatta = DD\ff;
%maxit = 5;
%tol = 1e-3;
%figure(1),
%[x_omp,Lset] = omp_brandoni(ff,DD,maxit,tol);



%{

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
    % fprintf('norm of core %d: %.4e\n', [k, norm(C{k},'fro')])
end

N = [784,784,784];

XX2 = reshape(full(X2), M);
% pause,
% 
% lx = full(LX,N);
% xval=tensor(lx);
% U1=nvecs(xval,1,M(1));
% U2=nvecs(xval,2,M(2));
% U3=nvecs(xval,3,M(3));
% S=ttm(xval,{U1',U2',U3'});
% sval=reshape(full(S),M);
% [U1, S1, ~] = svd(unfold(xval,1),'econ');
% [U2, S2, ~] = svd(unfold(xval,2),'econ');
% [U3, S3, ~] = svd(unfold(xval,3),'econ');
% diag(S1), 
% diag(S2),
% diag(S3),

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
    
% for k = 1:length(CoreX2)
%     fprintf('Norm of diagonal block relative to digit %d: %.4e\n',[k-1, norm(CoreX2{k},'fro')])
% end


YY2 = reshape(full(Y2), M);

p = n_sample;
if d == 3
    CoreY2{1} = full(YY2(1:p,1:p,1:p));
    for k = 2:d
        CoreY2{k} = full(YY2(p*(k-1)+1:p*k,p*(k-1)+1:p*k,p*(k-1)+1:p*k));
    end
elseif d == 4
    CoreY2{1} = YY2(1:p,1:p,1:p,1:p);
    for k = 2:d
        CoreY2{k} = YY2(p*(k-1)+1:p*k,p*(k-1)+1:p*k,p*(k-1)+1:p*k,p*(k-1)+1:p*k);
    end
end
    
% for k = 1:length(CoreY2)
%     fprintf('Norm of diagonal block relative to digit %d: %.4e\n',[k-1, norm(CoreY2{k},'fro')])
% end
fprintf('Tempo No-Prec: %d\n', t_tt_noprec)
fprintf('Tempo Prec: %d\n', t_tt_prec)

semilogy(Res2.real_rel)
hold on
semilogy(Res2pcQr.real_rel)
legend('no-pc','QR-Prec','CHOL-Prec')

%}

%{

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

% 
% 
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


tol = 1e-12;
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
% terms = values;
new_terms = new_values;
tic;
[X3, Res3] = RandTT_Tensorized_LSQR(new_terms, F_rand, Params, X);
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

fprintf('Tempo TT-LSQR: %d\n', t_tensor_trunc)
fprintf('Tempo Rand-TT-LSQR: %d\n', t_tensor_trunc5)

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

% 
% 
% % temp4 = X3; temp5 = X3; temp6 = X3;
% % for k = 1:3
% %     temp4 = ttm(temp4, k, values1{k}');
% %     temp5 = ttm(temp5, k, values2{k}');
% %     temp6 = ttm(temp6, k, values3{k}');
% % end
% % r_trunc_tucker = norm(F-temp4-temp5-temp6)/norm(F);
% 
% 
% % solver 
% % tic;
% % AA = kron( kron(sparse(Psi1),sparse(Ni1)), sparse(T1)) + kron( kron(sparse(T2),sparse(Ni2)), sparse(Phi2))...
% %    + kron( kron(sparse(Psi3),sparse(T3)), sparse(Phi3));
% % ff = sparse(kron(kron(c,b), a));
% % xx = AA\ff;
% % %AA = zeros(n1,m1); ff = zeros(n1,1); xx = zeros(m1,1);
% % t_dir_solv = toc
% % n_dir_solv = norm(AA*xx-ff)
% % [xx2, flag, relres, iter, resvec] = pcg(AA, ff, 1e-6, 200);
% 
% % TX2 = ttm(X2.F, {X2.U1, X2.U2, X2.U3});
% % error = norm(X1-TX2);
% % wrk1 = ttm(X1, values1); wrk2 = ttm(X1, values2); wrk3 = ttm(X1, values3);
% % wrk4 = ttm(TX2, values1); wrk5 = ttm(TX2, values2); wrk6 = ttm(TX2, values3);
% % fprintf('Real Residual Matrix-LSQR: %e, Real Residual TT-LSQR: %e \n', r_matrix_lsqr, r_trunc_tensor)
% % fprintf('Time Matrix-LSQR: %e, Time TT-LSQR: %e\n', t_lsqr_matrix, t_tensor_trunc)
% % fprintf('Error between the solutions: %e\n', error)
% 
% % figure(1)
% % subplot(2,1,1)
% % semilogy(Res1.real_abs,'r-o')
% % hold on
% % semilogy(Res1.est_abs, 'b-*')
% % hold off
% % title('Plot of Absolute Residuals')
% % xlabel('iterations')
% % ylabel('Absolute Residual')
% % legend('Real', 'Estimated')
% % subplot(2,1,2)
% % % figure(3)
% % semilogy(Res1.real_rel,'r-o')
% % hold on
% % semilogy(Res1.est_rel,'b-*')
% % hold off
% % title('Plot of Relative Residuals')
% % xlabel('iterations')
% % ylabel('Relative Residual')
% % legend('Real', 'Estimated')
% 
% % % figure(2)
% % % subplot(2,1,1)
% % % semilogy(Res2.real_abs,'r-o')
% % % hold on
% % % semilogy(Res2.est_abs, 'b-*')
% % % hold off
% % % title('Plot of Absolute Residuals (trunc version)')
% % % xlabel('iterations')
% % % ylabel('Absolute Residual')
% % % legend('Real', 'Estimated')
% % % subplot(2,1,2)
% % figure(4)
% % semilogy(Res2.real_rel,'r-o')
% % hold on
% % %semilogy(Res2.est_rel,'b-*')
% % % pause
% % semilogy(r_res.nrml_res, 'k-d')
% % %semilogy(Res3.real_rel,'g-+')
% % title('Plot of Relative Residuals (trunc version)')
% % xlabel('iterations')
% % ylabel('Relative Residual')
% % legend('TT-LSQR','Matricized-LSQR')
% % % legend('Real', 'Estimated','Matricized')
% 


%}

