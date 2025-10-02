function [Q, R] = randomizedQR(A)
[N,M]= size(A);
d = 2*M;
zeta = 10;
Omega = sparse_sign_backup(d,N,zeta);

OmegaA = Omega * A;
[QQ,RR] = qr(OmegaA,0);
R = RR;
Q = QQ*QQ';
end