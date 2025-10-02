function [A] = trunc_HOSVD(Coef, tol, r)

% function[S,U1,U2,U3] = trunc_HOSVD(A, tol)
%
% Given a tensor A of dimensions n1 x n2 x n3 this functions compute the
% High-Order SVD, that consists in the core tensor S and the matrices U1,
% U2, U3 such that A = S x_1 U1 x_2 U2 x_3 U3.

% Compute the unfoldings
% 
% A1 = unfold(A,1);
% A2 = unfold(A,2);
% A3 = unfold(A,3);

TT = ttm(Coef.F, {Coef.U1, Coef.U2, Coef.U3});


% Compute the (reduced) svd of the unfoldings
% 
% [U1, S1, ~] = svd(Coef.U1,0);
% [U2, S2, ~] = svd(Coef.U2,0);
% [U3, S3, ~] = svd(Coef.U3,0);
[U1, S1, ~] = svd(unfold(TT,1),'econ');
[U2, S2, ~] = svd(unfold(TT,2),'econ');
[U3, S3, ~] = svd(unfold(TT,3),'econ');

% size(Coef.U1), size(Coef.U2), size(Coef.U3),
% SS1 = diag(S1), SS2 = diag(S2), SS3 = diag(S3),
SS1 = diag(S1); SS2 = diag(S2); SS3 = diag(S3);


t1 = find(cumsum(SS1)./sum(SS1) > 1-tol, 1);
t2 = find(cumsum(SS2)./sum(SS2) > 1-tol, 1);
t3 = find(cumsum(SS3)./sum(SS3) > 1-tol, 1);

t1 = max([1 t1]);
t2 = max([1 t2]);
t3 = max([1 t3]);

% trunc = (min([norm(SS1(t1+1:end)), norm(SS2(t2+1:end)), norm(SS3(t3+1:end))]))

t1 = min([t1 r]);
t2 = min([t2 r]);
t3 = min([t3 r]);

% U1 = U1(1:t1, :);
% U2 = U2(1:t2, :);
% U3 = U3(1:t3, :);
U1 = U1(:, 1:t1);
U2 = U2(:, 1:t2);
U3 = U3(:, 1:t3);

% Building the core tensor S as mode product of the orthogonal matrices
% SC = Coef.F;
% SC = SC(1:t1, 1:t2, 1:t3);
S = ttm(Coef.F, {U1'*Coef.U1, U2'*Coef.U2, U3'*Coef.U3});

if ( t1 == 1 & t2 == 1 & t3 == 1)

    S(:,:,1) = S(1:t1, 1:t2, 1:t3);
    S = tensor(S);
    A.F = S;

else

    A.F = S(1:t1, 1:t2, 1:t3);

end
A.U1 = U1;
A.U2 = U2;
A.U3 = U3;

end