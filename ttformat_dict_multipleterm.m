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
n_sample = 6;
for k = 1:d
    M(k) = n_sample*d;
    M(k) = 14;
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

addpath('./Data')
load mnist_all
addpath('./oseledets_TT-Toolbox')

fprintf('Recognising a 1: \n')
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
    im2 = double(train2(randi([1,2000]),:)');
    Ni1(:,k) = reshape(im2,28^2,1);
    im3 = double(train3(randi([1,2000]),:)');
    Gamma1(:,k) = reshape(im3,28^2,1);
    im4 = double(train4(randi([1,2000]),:)');
    Theta1(:,k) = reshape(im4,28^2,1);

    Phi1(:,k) = Phi1(:,k)/norm(Phi1(:,k));
    Ni1(:,k) = Ni1(:,k)/norm(Ni1(:,k));
    Psi1(:,k) = Psi1(:,k)/norm(Psi1(:,k));
    Gamma1(:,k) = Gamma1(:,k)/norm(Gamma1(:,k));
    Theta1(:,k) = Theta1(:,k)/norm(Theta1(:,k));

end

for k = 1:M(2)

    im0 = double(train0(randi([1,2000]),:)');
    Phi2(:,k) = reshape(im0,28^2,1);
    im1 = double(train1(randi([1,2000]),:)');
    Psi2(:,k) = reshape(im1,28^2,1);
    im2 = double(train2(randi([1,2000]),:)');
    Ni2(:,k) = reshape(im2,28^2,1);
    im3 = double(train3(randi([1,2000]),:)');
    Gamma2(:,k) = reshape(im3, 28^2, 1);
    im4 = double(train4(randi([1,2000]),:)');
    Theta2(:,k) = reshape(im4,28^2,1);

    Phi2(:,k) = Phi2(:,k)/norm(Phi2(:,k));
    Ni2(:,k) = Ni2(:,k)/norm(Ni2(:,k));
    Psi2(:,k) = Psi2(:,k)/norm(Psi2(:,k));
    Gamma2(:,k) = Gamma2(:,k)/norm(Gamma2(:,k));
    Theta2(:,k) = Theta2(:,k)/norm(Theta2(:,k));

end

for k = 1:M(3)

    im0 = double(train0(randi([1,2000]),:)');
    Phi3(:,k) = reshape(im0,28^2,1);
    im1 = double(train1(randi([1,2000]),:)');
    Psi3(:,k) = reshape(im1,28^2,1);
    im2 = double(train2(randi([1,2000]),:)');
    Ni3(:,k) = reshape(im2,28^2,1);
    im3 = double(train3(randi([1,2000]),:)');
    Gamma3(:,k) = reshape(im3,28^2,1);
    im4 = double(train4(randi([1,2000]),:)');
    Theta3(:,k) = reshape(im4,28^2,1);

    Phi3(:,k) = Phi3(:,k)/norm(Phi3(:,k));
    Ni3(:,k) = Ni3(:,k)/norm(Ni3(:,k));
    Psi3(:,k) = Psi3(:,k)/norm(Psi3(:,k));
    Gamma3(:,k) = Gamma3(:,k)/norm(Gamma3(:,k));
    Theta3(:,k) = Theta3(:,k)/norm(Theta3(:,k));

end

for k = 1:M(3)

    im0 = double(train0(randi([1,2000]),:)');
    Phi4(:,k) = reshape(im0,28^2,1);
    im1 = double(train1(randi([1,2000]),:)');
    Psi4(:,k) = reshape(im1,28^2,1);
    im2 = double(train2(randi([1,2000]),:)');
    Ni4(:,k) = reshape(im2,28^2,1);
    im3 = double(train3(randi([1,2000]),:)');
    Gamma4(:,k) = reshape(im3,28^2,1);
    im4 = double(train4(randi([1,2000]),:)');
    Theta4(:,k) = reshape(im4,28^2,1);

    Phi4(:,k) = Phi4(:,k)/norm(Phi4(:,k));
    Ni4(:,k) = Ni4(:,k)/norm(Ni4(:,k));
    Psi4(:,k) = Psi4(:,k)/norm(Psi4(:,k));
    Gamma4(:,k) = Gamma4(:,k)/norm(Gamma4(:,k));
    Theta4(:,k) = Theta4(:,k)/norm(Theta4(:,k));

end

for k = 1:M(3)

    im0 = double(train0(randi([1,2000]),:)');
    Phi5(:,k) = reshape(im0,28^2,1);
    im1 = double(train1(randi([1,2000]),:)');
    Psi5(:,k) = reshape(im1,28^2,1);
    im2 = double(train2(randi([1,2000]),:)');
    Ni5(:,k) = reshape(im2,28^2,1);
    im3 = double(train3(randi([1,2000]),:)');
    Gamma5(:,k) = reshape(im3,28^2,1);
    im4 = double(train4(randi([1,2000]),:)');
    Theta5(:,k) = reshape(im4,28^2,1);

    Phi5(:,k) = Phi5(:,k)/norm(Phi5(:,k));
    Ni5(:,k) = Ni5(:,k)/norm(Ni5(:,k));
    Psi5(:,k) = Psi5(:,k)/norm(Psi5(:,k));
    Gamma5(:,k) = Gamma5(:,k)/norm(Gamma5(:,k));
    Theta5(:,k) = Theta5(:,k)/norm(Theta5(:,k));

end
%}

n1 = size(Phi1,1); n2 = size(Phi2,1); n3 = size(Phi3,1); n4 = n3; n5 = n4;


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
%values1{4} = Gamma1; values1{5} = Theta1;
values2{1} = Phi2; values2{2} = Psi2; values2{3} = Ni2;
%values2{4} = Gamma2; values2{5} = Theta2;
values3{1} = Phi3; values3{2} = Psi3; values3{3} = Ni3;
%values3{4} = Gamma3; values3{5} = Theta3;
values4{1} = Phi4; values4{2} = Psi4; values4{3} = Ni4;
%values4{4} = Gamma4; values4{5} = Theta4;
values5{1} = Phi5; values5{2} = Psi5; values5{3} = Ni5;
%values5{4} = Gamma5; values5{5} = Theta5;

% mi = M(1)/2;
% values1{1} = Phi1(:,1:mi); values1{2} = Psi1(:,1:mi); values1{3} = Ni1(:,1:mi);
% values2{1} = Phi2(:,1:mi); values2{2} = Psi2(:,1:mi); values2{3} = Ni2(:,1:mi);
% values3{1} = Phi3(:,1:mi); values3{2} = Psi3(:,1:mi); values3{3} = Ni3(:,1:mi);
% values4{1} = Phi4(:,1:mi); values4{2} = Psi4(:,1:mi); values4{3} = Ni4(:,1:mi);
% values5{1} = Phi5(:,1:mi); values5{2} = Psi5(:,1:mi); values5{3} = Ni5(:,1:mi);
% values6{1} = Phi1(:,mi+1:end); values6{2} = Psi1(:,mi+1:end); values6{3} = Ni1(:,mi+1:end);
% values7{1} = Phi2(:,mi+1:end); values7{2} = Psi2(:,mi+1:end); values7{3} = Ni2(:,mi+1:end);
% values8{1} = Phi3(:,mi+1:end); values8{2} = Psi3(:,mi+1:end); values8{3} = Ni3(:,mi+1:end);
% values9{1} = Phi4(:,mi+1:end); values9{2} = Psi4(:,mi+1:end); values9{3} = Ni4(:,mi+1:end);
% values10{1} = Phi5(:,mi+1:end); values10{2} = Psi5(:,mi+1:end); values10{3} = Ni5(:,mi+1:end);



%values4{1} = zeros(size(Phi1)); values4{2} = zeros(size(Psi1)); values4{3} = zeros(size(Ni1));
%values5{1} = zeros(size(Phi1)); values5{2} = zeros(size(Psi1)); values5{3} = zeros(size(Ni1));


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
for i = 1:d
    [~, R1{i}] = qr(values1{i},0);
    [~, R2{i}] = qr(values2{i},0);
    [~, R3{i}] = qr(values3{i},0);
    [~, R4{i}] = qr(values4{i},0);
    [~, R5{i}] = qr(values5{i},0);
    [~, R6{i}] = qr(values6{i},0);
    [~, R7{i}] = qr(values7{i},0);
    [~, R8{i}] = qr(values8{i},0);
    [~, R9{i}] = qr(values9{i},0);
    [~, R10{i}] = qr(values10{i},0);


    % [cond(R1{i}), cond(R2{i}), cond(R3{i})],
    % pause,

    %[m, ind] = min([cond(R1{i}), cond(R2{i}), cond(R3{i}),...
     %   cond(R4{i}), cond(R5{i})]);
    [m, ind] = min([cond(R1{i}), cond(R2{i}), cond(R3{i}), ...
        cond(R4{i}), cond(R5{i}), cond(R6{i}), cond(R7{i}), ...
        cond(R8{i}), cond(R9{i}), cond(R10{i})]);
    %[m, ind] = min([cond(R1{i}), cond(R2{i})]);
    rr = eval(['R',num2str(ind)]);
    
    % rr = R2;
    new_values1{i} = values1{i}/rr{i};
    new_values2{i} = values2{i}/rr{i};
    new_values3{i} = values3{i}/rr{i};
    new_values4{i} = values4{i}/rr{i};
    new_values5{i} = values5{i}/rr{i};
    new_values6{i} = values6{i}/rr{i};
    new_values7{i} = values7{i}/rr{i};
    new_values8{i} = values8{i}/rr{i};
    new_values9{i} = values9{i}/rr{i};
    new_values10{i} = values10{i}/rr{i};
end
t_creat_prec = toc;



new_values{1} = new_values1;
new_values{2} = new_values2;
new_values{3} = new_values3; 
new_values{4} = new_values4;
new_values{5} = new_values5;
new_values{6} = new_values6;
new_values{7} = new_values7;
new_values{8} = new_values8; 
new_values{9} = new_values9;
new_values{10} = new_values10;



if d == 4
    values1{4} = Gamma1_2;
    values2{4} = Gamma2_2;
    values3{4} = Gamma3_2;
end

values{1} = values1;
values{2} = values2;
values{3} = values3;
values{4} = values4;
values{5} = values5;
values{6} = values6;
values{7} = values7;
values{8} = values8;
values{9} = values9;
values{10} = values10;

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

rhs_vec = {rhs1, rhs1, rhs1, rhs1, rhs1};
F = tt_tensor(rhs_vec);
%F = F/norm(F);

for k = 1:n3

    D(:,:,k) = a*b'*c(k);

end



D = sptensor(D);
X0 = tensor(zeros(M));
Params.tol = 1e-6;
Params.imax = 100;
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


punteggio_angoli = [0, 0, 0];
punteggio_sottospazi = [0, 0, 0];
% punteggio_sottospazi_sketch = [0, 0, 0];
t_tot = 0;
M = M/2;

for k = 1:10

clearvars -except punteggio_angoli punteggio_sottospazi d values new_values M Params rr t_tot


addpath('./Data')
load mnist_all
addpath('./oseledets_TT-Toolbox')

fprintf('Recognising a 0: \n')
im0 = double(test0(randi(500),:)');
rhs1 = reshape(im0,28*28,1);
rhs1 = rhs1/norm(rhs1);

% rhs_vec = {rhs1, rhs1, rhs1, rhs1, rhs1};
rhs_vec = {rhs1, rhs1, rhs1};
F = tt_tensor(rhs_vec);
a = rhs1;

X = tt_zeros(M,d);


tic;
%[X2, Res2, iter2] = TT_Tensorized_LSQR(values, F, Params, X);
t_tt_noprec = toc;
X2 = X;

tic;
[Y2, Res2pcQr, iter3] = TT_Tensorized_LSQR(new_values, F, Params, X);
% [Y2, Res2pcQr, iter3] = Sketch_TT_Tensorized_LSQR(new_values, F, Params, X);
t_tt_prec = toc;
t_tot = t_tot + t_tt_prec;

for j = 1:Y2.d
    Y2 = ttm(Y2, j, inv(rr{j})');
end

% Y2   || a   -   D mode_j Y2 ||
%     n x n x n     m x n x m


% LY = OpL(values, Y2);
% ly2 = reshape(full(LY),[784,784,784,784,784]);
% ty = tensor(ly2);
% 
% cores = LY.core;
% rks = LY.r;
% d = LY.d;
% pos = LY.ps;
% m_y = LY.n;
% K1 = cores(pos(1):pos(2)-1);
% K2 = cores(pos(2):pos(3)-1);
% K3 = cores(pos(3):pos(4)-1);
% K1 = reshape(K1,m_y(1),rks(2));
% K2 = reshape(K2,rks(2),m_y(2),rks(3));
% K3 = reshape(K3,rks(3),m_y(3));
% [u1,s1,v1] = svd(K1,0);
% [u3,s3,v3]=svd(K3',0);
% U3 = u3; U1 = u1; 
% U2= nvecs(tensor(K2),2,54);
% U1=U1/norm(U1);U2=U2/norm(U2);U3=U3/norm(U3);
% %plot(abs(a'*U1)), hold on, plot(abs(a'*U2)), plot(abs(a'*U3))
% svd(a'*orth(U1)), svd(a'*orth(U2)), svd(a'*orth(U3))
% 
% % y2 = reshape(full(Y2),[M(1),M(2),M(3)]);
% % ty = tensor(y2);
% % Ricorda che valusa ha immagini come colonne
% % ty = tensor(G2);
% U1 = nvecs(ty,1,1); %<-- Mode 1
% U2 = nvecs(ty,2,1); %<-- Mode 2
% U3 = nvecs(ty,3,1); %<-- Mode 3
% 
% % angles
% angle_1 = abs(a' * U1);
% angle_2 = abs(a' * U2);
% angle_3 = abs(a' * U3);
% fprintf('Digit 0: %e, Digit 1: %e, Digit 8: %e\n', angle_1,...
%     angle_2, angle_3)
% [mm, ii] = max([angle_1, angle_2, angle_3]);
% punteggio_angoli(ii) = punteggio_angoli(ii) + 1;

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


% [factors,core] = tt_tuck(X2,1e-9);
% 
% completeX2 = core;
% for j = 1:X2.d
%     completeX2 = ttm(completeX2, j, factors{j}');
% end

% Y2 = OpL(values, Y2);
cores = Y2.core;
rks = Y2.r;
d = Y2.d;
pos = Y2.ps;
m_y = Y2.n;
G1 = cores(pos(1):pos(2)-1);
G2 = cores(pos(2):pos(3)-1);
G3 = cores(pos(3):pos(4)-1);
% G4 = cores(pos(4):pos(5)-1);
% G5 = cores(pos(5):pos(6)-1);
% size(G1),size(G2),size(G3)
G1 = reshape(G1,m_y(1),rks(2));
G2 = reshape(G2,rks(2),m_y(2),rks(3));
G3 = reshape(G3,rks(3),m_y(3));
%G3 = reshape(G3,rks(3),m_y(3),rks(4));
%G4 = reshape(G4,rks(4),m_y(4),rks(5));
%G5 = reshape(G5,rks(5),m_y(5));
% G1 = reshape(G1, 784, rks(2));
% G2 = reshape(G2, rks(2), 784, rks(3));
% G3 = reshape(G3, rks(3), 784);
% [u5,s5,v5] = svd(G5');
[u3, s3, v3] = svd(G3');
[u1,s1,v1] = svd(G1);
ty2 = tensor(G2);
%ty3 = tensor(G3);
%ty4 = tensor(G4);
%U1 = nvecs(ty,1,1); %<-- Mode 1
U2 = nvecs(ty2,2,1); %<-- Mode 2
%U3 = nvecs(ty3,2,1);
%U4 = nvecs(ty4,2,1);
%U3 = nvecs(ty,3,1); %<-- Mode 3
U1 = u1(:,1); U3 = u3(:,1);
%U1 = u1(:,1); U5 = u5(:,1);
UU1 = U1/norm(U1);
UU2 = U2/norm(U2);
UU3 = U3/norm(U3);
%UU4 = U4/norm(U4);
%UU5 = U5/norm(U5);
%UU = tt_tensor({UU1,UU2,UU3,UU4,UU5});
UU = tt_tensor({UU1,UU2,UU3});

luu = OpL(values, UU);

cores_luu = luu.core;
rks_luu = luu.r;
d_luu = luu.d;
pos_luu = luu.ps;
P1 = cores_luu(pos_luu(1):pos_luu(2)-1);
P2 = cores_luu(pos_luu(2):pos_luu(3)-1);
P3 = cores_luu(pos_luu(3):pos_luu(4)-1);
% P4 = cores_luu(pos_luu(4):pos_luu(5)-1);
% P5 = cores_luu(pos_luu(5):pos_luu(6)-1);
P1 = reshape(P1,784,rks_luu(2));
P2 = reshape(P2, rks_luu(2), 784, rks_luu(3));
P3 = reshape(P3, rks_luu(3), 784);
% P3 = reshape(P3, rks_luu(3), 784, rks_luu(4));
% P4 = reshape(P4, rks_luu(4), 784, rks_luu(5));
% P5 = reshape(P5, rks_luu(5), 784);
[u1,s1,v1] = svd(P1,0); UU1 = u1;%UU1 = u1(:,1); %UU1 = U1/norm(U1);
% [u5,s5,v5] = svd(P5',0); UU5 = u5;%UU3 = u3(:,1); %UU3 = U3/norm(U3);
[u3,s3,v3] = svd(P3',0); UU3 = u3;
UU2 = nvecs(tensor(P2),2,3); %UU2 = U2/norm(U2);
%UU3 = nvecs(tensor(P3),2,3);
%UU4 = nvecs(tensor(P4),2,3);

% questo non va benissimo :(((
% UU1 = values1{1}*U1+values2{1}*U1+values3{1}*U1;
% UU2 = values1{2}*U2+values2{2}*U2+values3{2}*U2;
% UU3 = values1{3}*U3+values2{3}*U3+values3{3}*U3;
UU1 = UU1/norm(UU1);UU2 = UU2/norm(UU2);UU3 = UU3/norm(UU3);
%UU4 = UU4/norm(UU4); UU5 = UU5/norm(UU5);


% abs(a'*UU1),abs(a'*UU2),abs(a'*UU3)
dist_0 = svd(a' * orth(UU1));
dist_1 = svd(a' * orth(UU2));
dist_2 = svd(a' * orth(UU3));
%dist_3 = svd(a' * orth(UU4));
%dist_4 = svd(a' * orth(UU5));
% fprintf('Dist 0: %e, Dist 1: %e, Dist 2: %e, Dist 3: %2, Dist 4: %e\n', dist_0, dist_1,...
%     dist_2, dist_3, dist_4)
fprintf('Dist 0: %e, Dist 1: %e, Dist 2: %e\n', dist_0, dist_1,...
    dist_2)

% svd(a'*orth(UU1)), svd(a'*orth(UU2)), svd(a'*orth(UU3))
% [m2, jj] = max([dist_0, dist_1, dist_2, dist_3, dist_4]);
[m2, jj] = max([dist_0, dist_1, dist_2]);
punteggio_sottospazi(jj) = punteggio_sottospazi(jj) +1;


fprintf('time: %d, residual: %e\n', t_tt_prec, Res2pcQr.real_rel(end))


end
punteggio_sottospazi,
t_tot,



%addpath('./ompbox10')

%fprintf('OMP ...\n')

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