function  [X,Lset] = omp_brandoni(Y,D,maxit,tol)


[nd,md]=size(D);
normy=norm(Y);
X=sparse(size(D,2),size(Y,2));
% in case norms were not normalized
  normDcol=sum(D.*D).^(1/2);

for i=1:size(Y,2)

%Res(:,i)=Y(:,i);
 Lset{i}=[];
 normres2=norm(Y(:,i))^2; %normres2=norm(Res(:,i))^2;
 p = D'*Y(:,i);  %Res(:,i) = Y(:,i);

 for k=1:maxit

  %p = D'*Res(:,i);    % this is updated at the bottom


  p(Lset{i})=0;
  [ival,ii]=max(abs(p(:)./normDcol(:)));

  Lset{i}=[Lset{i},ii];

% Gram Schmidt step
  if k==1

    Q(:,1)=D(:,ii)/norm(D(:,ii));

  else

% (Classical GS step performed twice for stability)
    Q(:,k)=D(:,ii)-Q(:,1:k-1)*(Q(:,1:k-1)'*D(:,ii));
    Q(:,k)=Q(:,k)-Q(:,1:k-1)*(Q(:,1:k-1)'*Q(:,k));

    Q(:,k)=Q(:,k)/norm(Q(:,k));

  end
  proj(k,1)=Q(:,k)'*Y(:,i);
% Res(:,i)=Res(:,i)-Q(:,k)*proj(k);   %This would be the residual update
  normres2=normres2-proj(k)^2;
  p = p - (D'*Q(:,k))*proj(k);
  
% normres(i,k)=norm(Res);
  normres(i,k)=sqrt(normres2);
%disp([i,k,sqrt(normres2),normres(i,k)^2/nd])
 % if normres(i,k)/normy<tol, break,end
  if normres(i,k)^2/nd<tol, break,end

 end
disp([i,k,sqrt(normres2),normres(i,k)^2/nd])

  R=triu(Q'*D(:,Lset{i}));
  X(Lset{i},i)=R\proj;
% checking the final residual, only for debugging
  Res(:,i)=Y(:,i)-D(:,Lset{i})*X(Lset{i},i);
norm(Res(:,i))
%disp(['...' ])
%pause
end

semilogy(normres')


