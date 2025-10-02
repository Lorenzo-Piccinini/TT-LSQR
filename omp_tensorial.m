function [X, Lset] = omp_tensorial(terms, Y, maxit, tol)

l = length(terms);
d = length(terms{1});
nd = 1; md = 1;
for i = 1:d 
    [n(i), m(i)] = size(terms{1}{i});
    nd = nd * n(i);
    md = md * m(i);
end
mm = m(1);
yy = full(reshape(Y,nd,1));

normy = norm(Y);
% Y is in TT format as y kron y .. kron y

X = tt_zeros(m, d);

Lset = []; % contains multi-index for p that is a tt-tensor of dim d
% p has dimension m x m x m ... d times, m is small
normres2 = normy;

p = OpL_T(terms, Y);
p = p(:);

for k = 1:maxit
    
    p(Lset) = 0;
    
    [ival, ii] = max(abs(p(:)));
    Lset = [Lset, ii];

    % skip the orth step
    
    % assume m(1) = m(2) = ... = m(d)
    j1 = floor(k/mm);
    j2 = mod(k,mm);
    DD(:,ii) = [];
    for kk = 1:d
        DD(:,ii) = [D(:,ii); terms{d}(:,j2)];
    end
    
    

    proj(k,1) = 
    normres2 = normres2



