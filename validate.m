function [ivalw_tot,ivalten_tot,oridx_tot]=validate(values,X,f,u1,u2,u3,itest) 


  [n,m]=size(values{1}{1});
  y = opl_val1(values, X);     %y = OpL(values, X);


  %Criterion 2
  q1= ttm(ttm(y,2,ones(n,1)),3,ones(n,1)); %q1=q1/norm(q1);
  w1=norm(ttm( q1,1,f));
  q1= ttm(ttm(y,1,ones(n,1)),3,ones(n,1)); %q1=q1/norm(q1);
  w2=norm(ttm(q1,2,f));
  q1= ttm(ttm(y,1,ones(n,1)),2,ones(n,1)); %q1=q1/norm(q1);
  w3=norm(ttm(q1,3,f));
  %([w1,w2,w3])
  [~,ivalw]=max([w1,w2,w3]);
  if ivalw==itest, ivalw_tot=1;else ivalw_tot=0;end
  fprintf('tensorized check2: test is guessed mode %d\n',ivalw)
clear q1 X Res


lu = y;
cores_lu = lu.core;
rks_lu = lu.r;
d_lu = lu.d;
pos_lu = lu.ps;
P1 = cores_lu(pos_lu(1):pos_lu(2)-1);
P2 = cores_lu(pos_lu(2):pos_lu(3)-1);
P3 = cores_lu(pos_lu(3):pos_lu(4)-1);
P1 = reshape(P1,n,rks_lu(2));
P2 = reshape(P2, rks_lu(2), n, rks_lu(3));
P3 = reshape(P3, rks_lu(3), n);
[UU1,~,~] = svd(P1,0); 
[UU3,~,~] = svd(P3',0); 
UU2 = nvecs(tensor(P2),2,size(UU1,2));
dist_0 = norm(f' * UU1);
dist_1 = norm(f' * UU2);
dist_2 = norm(f' * UU3);
[val2, idx2] = max([dist_0, dist_1, dist_2]);


  if idx2==itest, ivalten_tot=1;else ivalten_tot=0; end
  fprintf('tensorized check2plus: test is guessed mode %d\n',idx2)
clear lu P1 P2 P3 UU1 UU2 UU3


  
% orthodox query matching
  w1=norm(f-u1*(u1'*f)); w2=norm(f-u2*(u2'*f)); w3=norm(f-u3*(u3'*f));
  [~,oridx]=min([w1,w2,w3]);
  if oridx==itest, oridx_tot=1;else oridx_tot=0;end
fprintf('orthodox comparison, guessed mode %d \n',oridx)




