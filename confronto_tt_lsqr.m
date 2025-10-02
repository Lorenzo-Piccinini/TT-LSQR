clear all
close all

format shorte
format compact

n = 1000;
mtot = 90;
d = 3;
% l = 3;

for l = [3,6,10]
    l,
m = mtot/l;
t_tot = 0; res_tot = 0; iter_tot = 0;
n_es = 10;
for es = 1:n_es

for i = 1:l
    for j = 1:d
        terms{i}{j} = randn(n,m)/sqrt(m);
    end
end
M=[];
for ii = 1:d, M = [M, m]; end

f = randn(n,1); f = f/norm(f);
F = tt_tensor({f, f, f});

Params.tol = 1e-6;
Params.imax = 50;
Params.tol_tr = 1e-6;
Params.rank_tr = m;
X = tt_zeros(M,d);

tic;
[X_st, Res_st] = TT_Tensorized_LSQR2(terms, F, Params, X);
t_std = toc;
t_tot = t_tot + t_std;
%iter_tot = iter_tot + iter_st;

% s = 4*m;
% for i = 1:d
%     Omega{i} = randn(s,n)/sqrt(s);
% end
% for ii = 1:l
%     for jj = 1:d
%         Sterms{ii}{jj} = Omega{jj}*terms{ii}{jj};
%     end
% end
% 
% for i = 1:d
%     [Q1{i}, R1{i}] = qr(Sterms{1}{i},0);
%     [Q2{i}, R2{i}] = qr(Sterms{2}{i},0);
%     [Q3{i}, R3{i}] = qr(Sterms{3}{i},0);
%     [Q4{i}, R4{i}] = qr(Sterms{4}{i},0);
%     [Q5{i}, R5{i}] = qr(Sterms{5}{i},0);
%     [Q6{i}, R6{i}] = qr(Sterms{6}{i},0);
% 
%     [m, ind] = min([cond(R1{i}), cond(R2{i}), cond(R3{i}), ...
%         cond(R4{i}), cond(R5{i}), cond(R6{i})]);
% 
%     rr = eval(['R',num2str(ind)]);
%     qq = eval(['Q',num2str(ind)]);
% 
%     new_terms{1}{i} = Sterms{2}{i}/rr{i};
%     new_terms{2}{i} = Sterms{2}{i}/rr{i};
%     new_terms{3}{i} = Sterms{3}{i}/rr{i};
%     new_terms{4}{i} = Sterms{4}{i}/rr{i};
%     new_terms{5}{i} = Sterms{5}{i}/rr{i};
%     new_terms{6}{i} = Sterms{6}{i}/rr{i};
% 
% end
% 
% SF = F; 
% for i = 1:d, SF = ttm(SF, i, Omega{i}'); end
% X = SF;
% for i = 1:d, X = ttm(X, i, qq{i}); end
% 
% tic;
% [X_sk, Res_sk, iter_sk] = TT_Tensorized_LSQR(new_terms, SF, Params, X);
% t_sketch = toc;
% 
% for j = 1:X_sk.d
%     X_sk = ttm(X_sk, j, inv(rr{j}));
% end
t_sketch = 0;

fprintf('Time STANDARD: %d, Time SKETCH: %d\n', t_std, t_sketch)

LX_st = OpL(terms, X_st); LTLX_st = OpL_T(terms, LX_st);
%LX_sk = OpL(terms, X_sk); LTLX_sk = OpL_T(terms, LX_sk);
LTF = OpL_T(terms, F);

res_st_ne = norm(LTF - LTLX_st)/norm(LTF);
res_tot = res_tot + res_st_ne;
%res_sk_ne = norm(LTF - LTLX_sk)/norm(LTF);
res_sk_ne = 0;
fprintf('Res Normal Equation STANDARD: %e, SKETCH: %e\n', res_st_ne, res_sk_ne)
end

t_tot/n_es,
res_tot/n_es,
%iter_tot/n_es,

end
% figure(1)
% semilogy(Res_st.real_rel, '--r')
% hold on
% semilogy(Res_sk.real_rel, '--b')




