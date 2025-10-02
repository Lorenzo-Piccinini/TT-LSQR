function[X, Res] = RandTTlsqr(terms, Rhs, Params, X, flag)

l = length(terms); N = length(X); 
termsT = cell(l,1);

for i = 1:l
    termsT{i} = cell(N,1);
    for j = 1:N
        
        termsT{i}{j} = terms{i}{j}';
    end
end

tol_tr = Params.tol_tr;

beta = TTnorm(Rhs);

D = Rhs;
U = D; 
U = TTscale(U, 1/beta);

V = OperatorLT(termsT, U, flag);
% [N,I,R] = TTsizes(V),
% pause

alfa = TTnorm(V); V = TTscale(V, 1/alfa);
W = V;


phi_bar = beta;
rho_bar = alfa;

res0 = beta;
% res_ne_est = res0;
% Res.est_abs = [res0];
% Res.est_rel = [1];

Rhs_ne = OperatorLT(termsT, Rhs, flag);
res0_ne = TTnorm(Rhs_ne);

Res.real_abs = [res0];
Res.real_rel = [1];
Res.real_ne_abs = [res0_ne];
Res.real_ne_rel = [1];
res_old_ne = res0_ne;
totres = res0_ne;

i = 0;
fprintf('iteation   res_est   res_ne_est   res_true\n')

while ( i < Params.imax )

    i = i+1;
    
    % wrk1 = OpL(values1, values2, values3, V);
    wrk{1} = OperatorL(terms, V, flag);
    wrk{2} = U;
    
    if flag == 'random',
        U = TTsum_Randomize_then_Orthogonalize(wrk, [1; -alfa]);
    else
        U = TTsum(wrk, [1; -alfa],tol_tr);
    end
    % disp('U')
    % [N,I,R] = TTsizes(U),
    % pause
    
    % U = wrk1 - alfa*U;

    beta = TTnorm(U); U = TTscale(U, 1/beta);


    % V = OpL_T(values1, values2, U) - beta*V;
    % values1, values2, U, pause
    wrk{1} = OperatorLT(termsT, U, flag);
    wrk{2} = V;
    
    if flag == 'random'
        V = TTsum_Randomize_then_Orthogonalize(wrk, [1; -beta]);
    else
        V = TTsum(wrk, [1; -beta], tol_tr);
    end
    % V = wrk2 - beta*V;
    % disp('V')
    % [N,I,R] = TTsizes(V),
    % pause
    alfa = TTnorm(V); V = TTscale(V, 1/alfa);

    rho = sqrt(rho_bar^2 + beta^2);
    c = rho_bar/rho;
    s = beta/rho;
    theta = s*alfa;
    rho_bar = -c*alfa;
    phi = c*phi_bar;
    phi_bar = s*phi_bar;

    res_est = phi_bar;
    res_ne_est = phi_bar*alfa*abs(c);

    % Res.est_rel = [Res.est_rel; res_est/res0];
    % Res.est_abs = [Res.est_abs; res_est];

    wrk{1} = X;
    wrk{2} = W;
    if flag == 'random'
        X = TTsum_Randomize_then_Orthogonalize(wrk, [1; phi/rho]);
    else
        X = TTsum(wrk, [1; phi/rho], tol_tr);
    end
    % disp('X')
    % [N,I,R] = TTsizes(X),
    % pause
    % X = X + (phi/rho)*W;

    % Computing the real (normal) residual 
    LX = OperatorL(terms, X, flag);
    LTLX = OperatorLT(termsT, LX, flag);

    wrk{1} = LTLX;
    wrk{2} = Rhs_ne;
    if flag == 'random'
        R_ne = TTsum_Randomize_then_Orthogonalize(wrk, [1; -1]);
    else
        R_ne = TTsum(wrk, [1; -1]);
    end
    res_true_ne = TTnorm(R_ne);

    wrk{1} = LX; wrk{2} = Rhs;
    if flag == 'random'
        R = TTsum_Randomize_then_Orthogonalize(wrk, [1; -1]);
    else
        R = TTsum(wrk, [1; -1]);
    end
    res_true = TTnorm(R);

    
    %res_true = norm(D-temp1-temp2-temp3);
    totres = [totres; res_true_ne];

    if res_true_ne <= Params.tol || abs(res_true_ne - res_old_ne) < Params.tol/10
        break, end

    res_old_ne = res_true_ne;
    % res_true = 0;
    % Find a way to compute norms that uses the structure of the tensors.
    Res.real_ne_abs = [Res.real_ne_abs; res_true_ne];
    Res.real_ne_rel = [Res.real_ne_rel; res_true_ne/res0_ne];
    Res.real_abs = [Res.real_abs; res_true];
    Res.real_rel = [Res.real_rel; res_true/res0];

    wrk{1} = V;
    wrk{2} = W;
    if flag == 'random'
        W = TTsum_Randomize_then_Orthogonalize(wrk, [1; -theta/rho]);
    else
        W = TTsum(wrk, [1; -theta/rho], tol_tr);
    end
    % disp('W')
    % [N,I,R] = TTsizes(W),
    % pause
    % W = V - (theta/rho)*W;
    
    % disp([i, res_est, res_ne_est, res_true])
    fprintf(' %d %.4e\n', [i, res_true_ne/res0_ne])

end

%{
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

