function[X, Res] = TT_Tensorized_LSQR2(values, Rhs, Params, X)


%beta = norm(Rhs);

lx = OpL(values, X);
D = Rhs - lx;
beta = norm(D);

U = D; 
U = U/beta;
uold = full(U);

V = OpL_T(values, U);
V = round(V, Params.tol_tr,Params.rank_tr);

alfa = norm(V);
V = V/alfa;
vold = full(V);

W = V;

phi_bar = beta;
rho_bar = alfa;


res0 = beta;
% res0 = norm(Rhs);
Res.real_abs = [res0];
Res.real_rel = [1];
% Res.real_rel = [norm(D)/res0];

res_old = res0;
totres = res0;
totresnormal = alfa;

n_terms = length(values);
d = Rhs.d;



Ftot = OpL_T(values, Rhs);
res0_ne = norm(Ftot);
totrank = [];
i = 0;
fprintf('iteation  res_ne_true   res_true\n')

while ( i < Params.imax )

    i = i+1;
    
    wrk1 = OpL(values, V);
    wrk1 = round(wrk1, Params.tol_tr,Params.rank_tr);

    U = wrk1 - alfa*U;
    U = round(U, Params.tol_tr,Params.rank_tr);
clear wrk1

    beta = norm(U);
    U = U/beta;
    unew = full(U);
    uorth(i) = unew'*uold;
    uold = unew;

    wrk2 = OpL_T(values, U);
    wrk2 = round(wrk2, Params.tol_tr,Params.rank_tr);

    V = wrk2 - beta*V;
    V = round(V, Params.tol_tr,Params.rank_tr);
clear wrk2

    alfa = norm(V);
    V = V/alfa;
    vnew = full(V);
    vorth(i) = vnew'*vold;
    vold = vnew;

    rho = sqrt(rho_bar^2 + beta^2);
    c = rho_bar/rho;
    s = beta/rho;
    theta = s*alfa;
    rho_bar = -c*alfa;
    phi = c*phi_bar;
    phi_bar = s*phi_bar;

    res_est = phi_bar;
    res_ne_est = phi_bar*alfa*abs(c);

    X = X + (phi/rho)*W;
    X = round(X, Params.tol_tr,Params.rank_tr);
    totrank = [totrank; max(X.r)];
    totrank(end)

    LX = OpL(values, X);
    LTLX = OpL_T(values, LX);
    res_true = norm(Rhs- LX);
clear LX
    
    res_truenormal = norm(Ftot- LTLX);
    totres = [totres; res_true];
    totresnormal = [totresnormal; res_truenormal];
clear LTLX

   %if res_true/res0_ne <= Params.tol || abs(res_true - res_old) < Params.tol/10
    %if res_true/res0_ne <= Params.tol
        % if res_true/res0 <= Params.tol
        % fprintf('  %d  %.4e %.4e\n', [i,res_true/res0_ne, res_true])
        % break, end
    if res_truenormal/res0_ne <= Params.tol
        fprintf('  %d  %.4e %.4e\n', [i,res_true/res0_ne, res_true])
        break, end

    res_old = res_true;

    Res.real_abs = [Res.real_abs; res_true];
    % Res.real_rel = [Res.real_rel; res_true/res0_ne];
    Res.real_rel = [Res.real_rel; res_true];

    W = V - (theta/rho)*W;
    W = round(W, Params.tol_tr,Params.rank_tr);

    fprintf('  %d  %.4e %.4e\n', [i,res_true, res_truenormal/res0_ne])
   % X.r

end
rf=find(totrank/totrank(end)==1,1,'first');
figure(201)
semilogy(abs(uorth), 'ok')
hold on
semilogy(abs(vorth), 'xr')
semilogy([rf,rf],[1e-16,1e1],'linewidth', 4)
legend('orth U','orth V', 'max rank')
xlabel('Number of ietarations')
ylabel('Loss of orthogonality')
axis([0,250,1e-16,10]);
hold off
end




function[Y] = OpL(values, X)

d = X.d;
n_terms = length(values);

for k = 1:n_terms
    wrk = X;
    for j = 1:d
        wrk = ttm(wrk, j, values{k}{j}');
    end
    if k==1, Y=wrk; else, Y = Y + wrk;end
end

end

%function[Y] = OpL(values, X)
%
%d = X.d;
%n_terms = length(values);
%
%for k = 1:n_terms
%    T{k} = X;
%    for j = 1:d
%        T{k} = ttm(T{k}, j, values{k}{j}');
%    end
%end
%Y = T{1};
%for k = 2:n_terms
%    Y = Y + T{k};
%end
%end

function[Y] = OpL_T(values, X)

d = X.d;
n_terms = length(values);

for k = 1:n_terms

    wrk = X;
    for j = 1:d
        wrk = ttm(wrk, j, values{k}{j});
    end
    if k==1, Y=wrk; else, Y = Y + wrk;end
end
end


%function[Y] = OpL_T(values, X)
%
%d = X.d;
%n_terms = length(values);
%
%for k = 1:n_terms
%    T{k} = X;
%    for j = 1:d
%        T{k} = ttm(T{k}, j, values{k}{j});
%    end
%end
%Y = T{1};
%for k = 2:n_terms
%    Y = Y + T{k};
%end
%end





