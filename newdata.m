%clear all
close all
%clc

format shorte
format compact

N = []; M = [];


d = 3;

addpath('./tensor_toolbox-v3.6');
addpath('./oseledets_TT-Toolbox');
%tensor_toolbox-v3.6')

addpath('./Data')
load Reuters21578
gnd(end)
% idxR index pointer to first component of the group
for k=1:gnd(end),idxR(k)=find(gnd==k,1,'first');end

%test data
%  fea           8293x18933            6382752  double    sparse    
%  gnd           8293x1                  66344  double              
%  testIdx       2347x1                  18776  double              
%  trainIdx      5946x1    

% test is  first mode
%a = fea(idxR(2)-1,:)'; a = a/norm(a); fprintf('is test  mode 1?\n')
% test is  second mode
%a = fea(idxR(3)-2,:)'; a = a/norm(a); fprintf('is test  mode 2?\n')
%a = fea(idxR(3)-3,:)';  fprintf('is test  mode 2?\n')
% test is  third mode
a = fea(idxR(4)-4,:)'; a = a/norm(a); fprintf('is test  mode 3?\n')
b = a; c=a;

kk=4;
kk=9;
%kk=19;

values1{1} = fea(idxR(1):idxR(1)+kk,:)';
values1{2} = fea(idxR(2):idxR(2)+kk,:)';
values1{3} = fea(idxR(3):idxR(3)+kk,:)';

values2{1} = fea(idxR(1)+10:idxR(1)+10+kk,:)';
values2{2} = fea(idxR(2)+10:idxR(2)+10+kk,:)';
values2{3} = fea(idxR(3)+10:idxR(3)+10+kk,:)';

values3{1} = fea(idxR(1)+20:idxR(1)+20+kk,:)';
values3{2} = fea(idxR(2)+20:idxR(2)+20+kk,:)';
values3{3} = fea(idxR(3)+20:idxR(3)+20+kk,:)';

m=kk+1;
n=18933;

n1 = n; N = [N, n1]; m1 = m; M = [M, m1];
n2 = n; N = [N, n2]; m2 = m; M = [M, m2];
n3 = n; N = [N, n3]; m3 = m; M = [M, m3];


values{1} = values1;
values{2} = values2;
values{3} = values3;
for i = 1:3
    [~, R1{i}] = qr(full(values1{i}),0);
    [~, R2{i}] = qr(full(values2{i}),0);
    [~, R3{i}] = qr(full(values3{i}),0);

    [~, ind] = min([cond(R1{i}), cond(R2{i}), cond(R3{i})]);
    rr = eval(['R',num2str(ind)]);
    rr{i} = eye(size(R1{i}));
    new_values1{i} = values1{i}/rr{i};
    new_values2{i} = values2{i}/rr{i};
    new_values3{i} = values3{i}/rr{i};
end

new_values{1} = new_values1;
new_values{2} = new_values2;
new_values{3} = new_values3;

p =1;
% Tucker RHS
% [Qa,Ra] = qr(a,0);
% [Qb,Rb] = qr(b,0);
% [Qc,Rc] = qr(c,0);
% 
% K(:,:,1) = 1; K = tensor(K); K(:,:,1) = 1;
% 
% Rhs.a = Qa;
% Rhs.b = Qb;
% Rhs.c = Qc;
% Rhs.F = ttm(K,{Ra, Rb, Rc});

% Creating the TT-format RHS
rhs_vec = {a,b,c};
F = tt_tensor(rhs_vec);

X0 = tensor(zeros(M));
Params.tol = 1e-9;
Params.imax = 80;
Params.imax = 10;
Params.X0 = X0;
Params.tol_tr = 1e-8;
Params.r = 1000;

X = tt_zeros([m,m,m],3);

score = 0;
t_tot = 0;
sample = 10;

for ll = 1:sample

ii = randi([1,50],1);

% test is  first mode
%a = fea(idxR(2)-1,:)'; a = a/norm(a); fprintf('is test  mode 1?\n')
% test is  second mode
a = fea(idxR(3)-ii,:)'; a = a/norm(a); fprintf('is test  mode 2?\n')
%a = fea(idxR(3)-3,:)';  fprintf('is test  mode 2?\n')
% test is  third mode
%a = fea(idxR(4)-4,:)'; a = a/norm(a); fprintf('is test  mode 3?\n')
b = a; c=a;

tic;
[X2, Res2] = TT_Tensorized_LSQR(new_values, F, Params, X);
t_tensor_trunc = toc;
t_tot = t_tot + t_tensor_trunc;

% y=opl_val(new_values,X2);
y = OpL(new_values, X2);
[mival,ival]=max([norm(ttm(y,1,a)), norm(ttm(y,2,a)), norm(ttm(y,3,a))]);
fprintf('test is mode %d\n',ival)

if ival == 2
    score = score + 1;
end

end

fprintf('time: %d\n', t_tot)
fprintf('Score is %d out of %d\n', score, sample)

