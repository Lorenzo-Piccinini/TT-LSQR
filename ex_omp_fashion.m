addpath('./Data')

mnist_extract

[n,m]=size(imagestrain);
[n,mt]=size(imagestest);
name{1+0}='T-shirt/top';
name{1+1}='Trouser';
name{1+2}='Pullover';
name{1+3}='Dress';
name{1+4}='Coat';
name{1+5}='Sandal';
name{1+6}='Shirt';
name{1+7}='Sneaker';
name{1+8}='Bag';
name{1+9}='Ankle boot';

n_dict=10000;
mmax=5;
tol=1e-3;

id=randi(m,n_dict,1);
% D=table2array(imagestrain(:,id));
D = imagestrain(:,id);
Dnorm=diag(sqrt(sum(D.*D,1)));
% Dnorm = sqrt(sum(D.*D,1));
D=D/Dnorm;
iY=randi(mt,1,1);
Y=imagestest(:,iY);
name{1+labelstest(iY)}
nh=sqrt(n);
% pause
%figure(1)
%Matching Pursuit
%[Xmp,Lsetmp]=mp(Y,D,mmax,tol);
% Xmp = zeros(1e4,1); Lsetmp{1} = 0;
% Lsetmp'
% name{1+labelstrain(id(Lsetmp{1}'))}
% 
% figure(10)
% subplot(1,3,1)
% imshow(reshape(Y,nh,nh))
% 
% subplot(1,3,2)
% imshow(reshape(D*Xmp,nh,nh))

figure(1)
%Orthogonal Matching Pursuit
%[Xomp,Lsetomp]=omp_basic(Y,D,mmax,tol);
[Xomp,Lsetomp]=omp_brandoni(Y,D,mmax,tol);
Lsetomp'
Xomp,name{1+labelstrain(id(Lsetomp{1}'))}

figure(10)
subplot(1,3,3)
imshow(reshape(D*Xomp,nh,nh))

