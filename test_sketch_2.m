% m = ?
% ivalf = ?
m=60; ivalf=1;
format short e
%m = 60;

addpath('/home/valeria/Corso_Matematica/DataScienceLM/matlab/tensor_toolbox-v3.2.1');
addpath('/home/valeria/Dropbox/Piccinini/TensorLSQR/oseledets_TT-Toolbox');




load A_cran.mat; A = A_cran; load Acran_idx; fprintf('Acran\n')
%load A_med.mat; A = A_med; load Amed_idx; fprintf('Amed\n')






[n,~]=size(A);
ng = 3;
d=3;
%[idx, c] = kmeans(A', ng, 'Distance', 'correlation', 'Replicates', 10);


g1 = find(idx == 1);    s1 = sum(A(:,g1),2);
g2 = find(idx == 2);    s2 = sum(A(:,g2),2);
g3 = find(idx == 3);    s3 = sum(A(:,g3),2);
g4 = 0; %g4 = find(idx == 4);    s4 = sum(A(:,g4),2);
g5 = 0; % g5 = find(idx == 5);  s5 = sum(A(:,g5),2);
g6 = 0; % g6 = find(idx == 6);  s6 = sum(A(:,g6),2);


[m1, i1] = sort(s1, 'descend');
[m2, i2] = sort(s2, 'descend');
[m3, i3] = sort(s3, 'descend');
% [m4, i4] = sort(s4, 'descend');
% [m5, i5] = sort(s5, 'descend');  %dic(i5(1:5),:)
% [m6, i6] = sort(s6, 'descend');  %dic(i6(1:5),:)


[vord,iord] = sort([length(g1),length(g2),length(g3),length(g4),length(g5),length(g6)],'descend');


gx1 = eval(['g', num2str(iord(1))]);
gx2 = eval(['g', num2str(iord(2))]);
gx3 = eval(['g', num2str(iord(3))]);




A1 = A(:, gx1(1:m)); 
A2 = A(:, gx2(1:m)); 
A3 = A(:, gx3(1:m)); 
A1=A1/norm(A1',1);
A2=A2/norm(A2',1);
A3=A3/norm(A3',1);
 AA=([A1 A2 A3 ]);
 [u1,~,~]=svds((A1),10); 
 [u2,~,~]=svds((A2),10); 
 [u3,~,~]=svds((A3),10); 
% A4 = A(:, gx4(1:m));
%spy(abs(corrcoef(zscore(AA)))>0.3)


l = 6;
mm = m/l;


fprintf('num terms: %d,  num modes: %d, matrix dim per mode: %d x %d\n', l,3,n,mm)


% 4 termini a 4 dimensioni
values1{1} = A1(:, 1:mm); values1{2} = A2(:, 1:mm); values1{3} = A3(:, 1:mm); %values1{4} = A4(:, 1:mm);
values2{1} = A1(:, mm+1:2*mm); values2{2} = A2(:, mm+1:2*mm); values2{3} = A3(:, mm+1:2*mm);% values2{4} = A4(:, mm+1:2*mm);
values3{1} = A1(:, 2*mm+1:3*mm); values3{2} = A2(:, 2*mm+1:3*mm); values3{3} = A3(:, 2*mm+1:3*mm);% values3{4} = A4(:, 2*mm+1:3*mm);
values4{1} = A1(:, 3*mm+1:4*mm); values4{2} = A2(:, 3*mm+1:4*mm); values4{3} = A3(:, 3*mm+1:4*mm); %values4{4} = A4(:, 3*mm+1:end);
values5{1} = A1(:, 4*mm+1:5*mm); values5{2} = A2(:, 4*mm+1:5*mm); values5{3} = A3(:, 4*mm+1:5*mm); %values4{4} = A4(:, 3*mm+1:end);
values6{1} = A1(:, 5*mm+1:6*mm); values6{2} = A2(:, 5*mm+1:6*mm); values6{3} = A3(:, 5*mm+1:6*mm); %values4{4} = A4(:, 3*mm+1:end);


values{1} = values1;
values{2} = values2;
values{3} = values3;
values{4} = values4;
values{5} = values5;
values{6} = values6;



ix_tot=0; ival_tot=0; ivalw_tot=0; oridx_tot=0; ivalten_tot=0;
Six_tot=0; Sival_tot=0; Sivalw_tot=0; Soridx_tot=0; Sivalten_tot=0;
pix_tot=0; pival_tot=0; pivalw_tot=0; poridx_tot=0; pivalten_tot=0;
t_tt_tot=0;
t_sk_tot=0;
t_psk_tot=0;


for kk=1:20
  kk
  switch ivalf


 case 1
    f = A(:, gx1(m+10+kk)); itest=1;
 case 2
    f = A(:, gx2(m+10+kk)); itest=2;
 case 3
    f = A(:, gx3(m+10+kk)); itest=3;
 otherwise
    break
 end
  %f = A(:, gx4(m+10+kk));
  f=f/norm(f,1);


  fprintf('test class is %d\n', itest)
  %F = tt_tensor({f,f,f,f});
  F = tt_tensor({f,f,f});


  X0 = tt_zeros([mm,mm,mm], d);
  Params.tol = 1e-5;
  Params.imax = 20;
  Params.tol_tr = 1e-4;
%  Params.tol_tr = 1e-10;
  Params.rank_tr = 10000;
  Params.r = 1000;
fprintf('Standard LS\n')
   tic;
   [X, Res] = TT_Tensorized_LSQR2(values, F, Params, X0);
%X_st=X;
X=X0;
  t_tt = toc
t_tt_tot=t_tt_tot+t_tt;
  [ivalw,ivalten,oridx]=validate(values,X,f,u1,u2,u3,itest);
   if ivalw==itest, ivalw_tot = ivalw_tot+1;end
   if ivalten==itest, ivalten_tot = ivalten_tot+1;end
   if oridx==itest, oridx_tot = oridx_tot+1;end




% Sketched
 s = 2*d*m;
fprintf('Value of the sketching parameter %d\n',s)
  rng('default')
  E = spdiags(2*round(rand(n,1))-1,0,n,n); % Rademacher random variables
  D = speye(n); D = D(randperm(n,s),:);    % pick s entries at random
  Omega = @(X) D*dct(E*X)/sqrt(s/n);   % discrete cosine transform(base)
  for i = 1:d
       for ii = 1:l
          new_terms{ii}{i} = Omega(full(values{ii}{i}));
       end
      sf{i} = Omega(full(f));
  end
  
  SF = tt_tensor({sf{1}, sf{2}, sf{3}});
 SAA=Omega(full([A1 A2 A3 ]));
 [Su1,~,~]=svds(Omega(full(A1)),10); 
 [Su2,~,~]=svds(Omega(full(A2)),10); 
 [Su3,~,~]=svds(Omega(full(A3)),10); 
  X0 = tt_zeros([mm,mm,mm], d);


fprintf('Sketched LS\n')
tic;
[X_sk, Res_sk] = TT_Tensorized_LSQR2(new_terms, SF, Params, X0);
t_sketch = toc
t_sk_tot=t_sk_tot+t_sketch;


  [ivalw,ivalten,oridx]=validate(new_terms,X_sk,sf{1},Su1,Su2,Su3,itest);
   if ivalw==1, Sivalw_tot = Sivalw_tot+1;end
   if ivalten==1, Sivalten_tot = Sivalten_tot+1;end
   if oridx==1, Soridx_tot = Soridx_tot+1;end


%%%%%%%%%%%%%%%%%%%%
fprintf('Standard LS w/ initial Sketched soln\n')
   lx = opl_val1(values, X_sk);
   D = F - lx;
   %D = F;
   %D = round(D, Params.tol_tr,Params.rank_tr);
   clear lx 
  Params.imax = 5;
tic
    [Xplus, Resplus] = TT_Tensorized_LSQR2(values, D, Params, X0);
   X_plus = X_sk+Xplus;
t_psketch = toc
t_psk_tot=t_psk_tot+t_psketch;
  [ivalw,ivalten,oridx]=validate(values,X_plus,f,u1,u2,u3,itest);
   if ivalw==1, pivalw_tot = pivalw_tot+1;end
   if ivalten==1, pivalten_tot = pivalten_tot+1;end
   if oridx==1, poridx_tot = poridx_tot+1;end
  


end
time_standard=t_tt_tot/20
time_sketch=t_sk_tot/20
time_psketch=t_psk_tot/20


fprintf('Standard percentage of correct classification:  tensorized 2 %d, tensorized2plus %d,  orth %d \n', ...
ivalw_tot/kk*100,ivalten_tot/kk*100,oridx_tot/kk*100);
fprintf('Sketched percentage of correct classification:  tensorized 2 %d, tensorized2plus %d,  orth %d \n', ...
Sivalw_tot/kk*100,Sivalten_tot/kk*100,Soridx_tot/kk*100);
fprintf('Std + Sketched percentage of correct classification:  tensorized 2 %d, tensorized2plus %d,  orth %d \n', ...
pivalw_tot/kk*100,pivalten_tot/kk*100,poridx_tot/kk*100);
