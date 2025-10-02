format shorte
clear all

load A_cran.mat; A = A_cran;
load dict_cran.mat; dic = dict_cran;

ng = 4;
[idx, c] = kmeans(A', ng, 'Distance', 'correlation', 'Replicates', 10);

g1 = find(idx == 1);    s1 = sum(A(:,g1),2);
g2 = find(idx == 2);    s2 = sum(A(:,g2),2);
g3 = find(idx == 3);    s3 = sum(A(:,g3),2);
g4 = find(idx == 4);    s4 = sum(A(:,g4),2);
g5 = 0; % g5 = find(idx == 5);  s5 = sum(A(:,g5),2);
g6 = 0; % g6 = find(idx == 6);  s6 = sum(A(:,g6),2);

[m1, i1] = sort(s1, 'descend');
[m2, i2] = sort(s2, 'descend');
[m3, i3] = sort(s3, 'descend');
[m4, i4] = sort(s4, 'descend');
% [m5, i5] = sort(s5, 'descend');  %dic(i5(1:5),:)
% [m6, i6] = sort(s6, 'descend');  %dic(i6(1:5),:)

[vord,iord] = sort([length(g1),length(g2),length(g3),length(g4),length(g5),length(g6)],'descend');

gx1 = eval(['g', num2str(iord(1))]);
gx2 = eval(['g', num2str(iord(2))]);
gx3 = eval(['g', num2str(iord(3))]);
gx4 = eval(['g', num2str(iord(4))]);
%gx5=eval(['g',num2str(iord(5))]);
%gx6=eval(['g',num2str(iord(6))]);

m = 40;
% X = [A(:, gx1(1:m)), A(:, gx2(1:m)), A(:, gx3(1:m))];
% X = [X, A(:, gx4(1:m))];
% This X is the dictionary

A1 = A(:, gx1(1:m)); 
A2 = A(:, gx2(1:m));
A3 = A(:, gx3(1:m));
A4 = A(:, gx4(1:m));

l = 4;
mm = m/l;

% 4 termini a 4 dimensioni
values1{1} = A1(:, 1:mm); values1{2} = A2(:, 1:mm); values1{3} = A3(:, 1:mm); %values1{4} = A4(:, 1:mm);
values2{1} = A1(:, mm+1:2*mm); values2{2} = A2(:, mm+1:2*mm); values2{3} = A3(:, mm+1:2*mm); %values2{4} = A4(:, mm+1:2*mm);
values3{1} = A1(:, 2*mm+1:3*mm); values3{2} = A2(:, 2*mm+1:3*mm); values3{3} = A3(:, 2*mm+1:3*mm); %values3{4} = A4(:, 2*mm+1:3*mm);
values4{1} = A1(:, 3*mm+1:end); values4{2} = A2(:, 3*mm+1:end); values4{3} = A3(:, 3*mm+1:end); %values4{4} = A4(:, 3*mm+1:end);

values{1} = values1;
values{2} = values2;
values{3} = values3;
values{4} = values4;

% f = A(:, gx1(m+10));
f = A(:, gx2(m+13));
% f = A(:, gx3(m+10));
% f = A(:, gx4(m+10));

%F = tt_tensor({f,f,f,f});
F = tt_tensor({f,f,f});

%X0 = tt_zeros([mm,mm,mm,mm], 4);
X0 = tt_zeros([mm,mm,mm], 3);
Params.tol = 1e-8;
Params.imax = 200;
Params.tol_tr = 1e-8;
Params.r = 1000;

tic;
[X, Res, iter] = TT_Tensorized_LSQR(values, F, Params, X0);
t_tt = toc;

y = OpL(values, X);
[mival,ival]=max([norm(ttm(y,1,f)), norm(ttm(y,2,f)), norm(ttm(y,3,f))]);
%[mival,ival]=max([norm(ttm(y,1,f)), norm(ttm(y,2,f)), norm(ttm(y,3,f)), norm(ttm(y,4,f))]);
fprintf('test is mode %d\n',ival)


