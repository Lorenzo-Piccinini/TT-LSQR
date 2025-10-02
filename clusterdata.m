load A_cran; A=A_cran;
load dict_cran; dic=dict_cran;

%load /home/valeria/Corso_Matematica/DataMining/Datasets/Term-Document/A_cisi.mat
%load /home/valeria/Corso_Matematica/DataMining/Datasets/Term-Document/dict_cisi.mat
%A=A_cisi;
%dic=dict_cisi;

%load /home/valeria/Corso_Matematica/DataMining/Datasets/Term-Document/A_med.mat
%load /home/valeria/Corso_Matematica/DataMining/Datasets/Term-Document/dict_med.mat
%A=A_med;
%dic=dict_med;
%[idx,c]=kmeans(A',6,'Distance','correlation','Replicates',5,'Start','cluster');
%[idx,c]=kmeans(A',6,'Distance','sqeuclidean','Replicates',5,'Start','cluster');
ng=4;
[idx,c]=kmeans(A',ng,'Distance','correlation','Replicates',10);  %,'Start','cluster');
%g = groups      s = frequency of words in that group
g1=find(idx==1);      s1=sum(A(:,g1),2);
g2=find(idx==2);      s2=sum(A(:,g2),2);
g3=find(idx==3);      s3=sum(A(:,g3),2);
g4=find(idx==4);      s4=sum(A(:,g4),2);
g5=0; %g5=find(idx==5);      s5=sum(A(:,g5));
g6=0; %g6=find(idx==6);      s6=sum(A(:,g6));
[m1,i1]=sort(s1,'descend');  dic(i1(1:10),:)
[m2,i2]=sort(s2,'descend');  dic(i2(1:10),:)
[m3,i3]=sort(s3,'descend');  dic(i3(1:10),:)
[m4,i4]=sort(s4,'descend');  dic(i4(1:5),:)
%[m5,i5]=sort(s5,'descend');  dic(i5(1:5),:)
%[m6,i6]=sort(s6,'descend');  dic(i6(1:5),:)
[vord,iord]=sort([length(g1),length(g2),length(g3),length(g4),length(g5),length(g6)],'descend');
gx1=eval(['g',num2str(iord(1))]);
gx2=eval(['g',num2str(iord(2))]);
gx3=eval(['g',num2str(iord(3))]);
gx4=eval(['g',num2str(iord(4))]);
%gx5=eval(['g',num2str(iord(5))]);
%gx6=eval(['g',num2str(iord(6))]);
m=200;
X=[A(:,gx1(1:m)),A(:,gx2(1:m)),A(:,gx3(1:m))];
X=[X A(:,gx4(1:m))]; %,A(:,gx5(1:m)),A(:,gx6(1:m))];
idxX=kron((1:ng)',ones(m,1));
%idxX=kron((1:6)',ones(m,1));
%Y=[A_cran(:,gx1),A_cran(:,gx2),A_cran(:,gx3),A_cran(:,gx4),A_cran(:,gx5),A_cran(:,gx6)];
%figure(15)
%spy(abs(corrcoef(zscore(Y)))>0.3)
figure(20)
spy(abs(corrcoef(zscore(X)))>0.3)

stot=0;ktot=0;
for k=m+1:length(gx1)
  x=X\A(:,gx1(k));
 % [vx,ix]=max(abs(x));
  %if idxX(ix)==1, stot=stot+1;end
  xr=reshape(x,m,ng); 
  [vx,ix]=max(sqrt(sum(xr.*xr)));
  if (ix)==1, stot=stot+1;end
  ktot=ktot+1;
end
num2str(1),stot, stot/ktot

stot=0;ktot=0;
for k=m+1:length(gx2)
  x=X\A(:,gx2(k));
  %[vx,ix]=max(abs(x));
  %if idxX(ix)==2, stot=stot+1;end
  xr=reshape(x,m,ng); 
  [vx,ix]=max(sqrt(sum(xr.*xr)));
  if (ix)==2, stot=stot+1;end
  ktot=ktot+1;
end
num2str(2),stot, stot/ktot

stot=0;ktot=0;
for k=m+1:length(gx3)
  x=X\A(:,gx3(k));
  %[vx,ix]=max(abs(x));
  %if idxX(ix)==3, stot=stot+1;end
  xr=reshape(x,m,ng); 
  [vx,ix]=max(sqrt(sum(xr.*xr)));
  if (ix)==3, stot=stot+1;end
  ktot=ktot+1;
end
num2str(3),stot, stot/ktot

stot=0;ktot=0;
for k=m+1:length(gx4)
  x=X\A(:,gx4(k));
  %[vx,ix]=max(abs(x));
  %if idxX(ix)==4, stot=stot+1;end
  xr=reshape(x,m,ng); 
  [vx,ix]=max(sqrt(sum(xr.*xr)));
  if (ix)==4, stot=stot+1;end
  ktot=ktot+1;
end
num2str(4),stot, stot/ktot

%{
stot=0;ktot=0;
for k=m+1:length(gx5)
  x=X\A(:,gx5(k));
  [vx,ix]=max(abs(x));
  if idxX(ix)==5, stot=stot+1;end
  ktot=ktot+1;
end
num2str(5),stot, stot/ktot

stot=0;ktot=0;
for k=m+1:length(gx6)
  x=X\A(:,gx6(k));
  [vx,ix]=max(abs(x));
  if idxX(ix)==6, stot=stot+1;end
  ktot=ktot+1;
end
num2str(6),stot, stot/ktot
%}

