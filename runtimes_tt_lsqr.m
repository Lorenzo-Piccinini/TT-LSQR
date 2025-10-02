clear all
 
format shorte
format compact

d = 3;
l = 10;
m = 120;
for k = 1:d
    M(k) = m/l;
end


addpath('./Data')
load mnist_all
addpath('./oseledets_TT-Toolbox')

for k = 1:M(1)

    im0 = double(train0(randi([1,2000]),:)');
    Phi1(:,k) = reshape(im0,28^2,1);
    im1 = double(train1(randi([1,2000]),:)');
    Psi1(:,k) = reshape(im1,28^2,1);
    im2 = double(train2(randi([1,2000]),:)');
    Ni1(:,k) = reshape(im2,28^2,1);
    im3 = double(train3(randi([1,2000]),:)');
    Gamma1(:,k) = reshape(im3,28^2,1);
    im4 = double(train4(randi([1,2000]),:)');
    Theta1(:,k) = reshape(im4,28^2,1);

    Phi1(:,k) = Phi1(:,k)/norm(Phi1(:,k));
    Ni1(:,k) = Ni1(:,k)/norm(Ni1(:,k));
    Psi1(:,k) = Psi1(:,k)/norm(Psi1(:,k));
    Gamma1(:,k) = Gamma1(:,k)/norm(Gamma1(:,k));
    Theta1(:,k) = Theta1(:,k)/norm(Theta1(:,k));

end

for k = 1:M(2)

    im0 = double(train0(randi([1,2000]),:)');
    Phi2(:,k) = reshape(im0,28^2,1);
    im1 = double(train1(randi([1,2000]),:)');
    Psi2(:,k) = reshape(im1,28^2,1);
    im2 = double(train2(randi([1,2000]),:)');
    Ni2(:,k) = reshape(im2,28^2,1);
    im3 = double(train3(randi([1,2000]),:)');
    Gamma2(:,k) = reshape(im3, 28^2, 1);
    im4 = double(train4(randi([1,2000]),:)');
    Theta2(:,k) = reshape(im4,28^2,1);

    Phi2(:,k) = Phi2(:,k)/norm(Phi2(:,k));
    Ni2(:,k) = Ni2(:,k)/norm(Ni2(:,k));
    Psi2(:,k) = Psi2(:,k)/norm(Psi2(:,k));
    Gamma2(:,k) = Gamma2(:,k)/norm(Gamma2(:,k));
    Theta2(:,k) = Theta2(:,k)/norm(Theta2(:,k));

end

for k = 1:M(3)

    im0 = double(train0(randi([1,2000]),:)');
    Phi3(:,k) = reshape(im0,28^2,1);
    im1 = double(train1(randi([1,2000]),:)');
    Psi3(:,k) = reshape(im1,28^2,1);
    im2 = double(train2(randi([1,2000]),:)');
    Ni3(:,k) = reshape(im2,28^2,1);
    im3 = double(train3(randi([1,2000]),:)');
    Gamma3(:,k) = reshape(im3,28^2,1);
    im4 = double(train4(randi([1,2000]),:)');
    Theta3(:,k) = reshape(im4,28^2,1);

    Phi3(:,k) = Phi3(:,k)/norm(Phi3(:,k));
    Ni3(:,k) = Ni3(:,k)/norm(Ni3(:,k));
    Psi3(:,k) = Psi3(:,k)/norm(Psi3(:,k));
    Gamma3(:,k) = Gamma3(:,k)/norm(Gamma3(:,k));
    Theta3(:,k) = Theta3(:,k)/norm(Theta3(:,k));

end

for k = 1:M(3)

    im0 = double(train0(randi([1,2000]),:)');
    Phi4(:,k) = reshape(im0,28^2,1);
    im1 = double(train1(randi([1,2000]),:)');
    Psi4(:,k) = reshape(im1,28^2,1);
    im2 = double(train2(randi([1,2000]),:)');
    Ni4(:,k) = reshape(im2,28^2,1);
    im3 = double(train3(randi([1,2000]),:)');
    Gamma4(:,k) = reshape(im3,28^2,1);
    im4 = double(train4(randi([1,2000]),:)');
    Theta4(:,k) = reshape(im4,28^2,1);

    Phi4(:,k) = Phi4(:,k)/norm(Phi4(:,k));
    Ni4(:,k) = Ni4(:,k)/norm(Ni4(:,k));
    Psi4(:,k) = Psi4(:,k)/norm(Psi4(:,k));
    Gamma4(:,k) = Gamma4(:,k)/norm(Gamma4(:,k));
    Theta4(:,k) = Theta4(:,k)/norm(Theta4(:,k));

end

for k = 1:M(3)

    im0 = double(train0(randi([1,2000]),:)');
    Phi5(:,k) = reshape(im0,28^2,1);
    im1 = double(train1(randi([1,2000]),:)');
    Psi5(:,k) = reshape(im1,28^2,1);
    im2 = double(train2(randi([1,2000]),:)');
    Ni5(:,k) = reshape(im2,28^2,1);
    im3 = double(train3(randi([1,2000]),:)');
    Gamma5(:,k) = reshape(im3,28^2,1);
    im4 = double(train4(randi([1,2000]),:)');
    Theta5(:,k) = reshape(im4,28^2,1);

    Phi5(:,k) = Phi5(:,k)/norm(Phi5(:,k));
    Ni5(:,k) = Ni5(:,k)/norm(Ni5(:,k));
    Psi5(:,k) = Psi5(:,k)/norm(Psi5(:,k));
    Gamma5(:,k) = Gamma5(:,k)/norm(Gamma5(:,k));
    Theta5(:,k) = Theta5(:,k)/norm(Theta5(:,k));

end
for k = 1:M(3)

    im0 = double(train0(randi([1,2000]),:)');
    Phi6(:,k) = reshape(im0,28^2,1);
    im1 = double(train1(randi([1,2000]),:)');
    Psi6(:,k) = reshape(im1,28^2,1);
    im2 = double(train2(randi([1,2000]),:)');
    Ni6(:,k) = reshape(im2,28^2,1);
    im3 = double(train3(randi([1,2000]),:)');
    Gamma6(:,k) = reshape(im3,28^2,1);
    im4 = double(train4(randi([1,2000]),:)');
    Theta6(:,k) = reshape(im4,28^2,1);

    Phi6(:,k) = Phi6(:,k)/norm(Phi6(:,k));
    Ni6(:,k) = Ni6(:,k)/norm(Ni6(:,k));
    Psi6(:,k) = Psi6(:,k)/norm(Psi6(:,k));
    Gamma6(:,k) = Gamma6(:,k)/norm(Gamma6(:,k));
    Theta6(:,k) = Theta6(:,k)/norm(Theta6(:,k));

end



values1{1} = Phi1; values1{2} = Psi1; values1{3} = Ni1;
%values1{4} = Gamma1; values1{5} = Theta1;
values2{1} = Phi2; values2{2} = Psi2; values2{3} = Ni2;
%values2{4} = Gamma2; values2{5} = Theta2;
values3{1} = Phi3; values3{2} = Psi3; values3{3} = Ni3;
%values3{4} = Gamma3; values3{5} = Theta3;
values4{1} = Phi4; values4{2} = Psi4; values4{3} = Ni4;
%values4{4} = Gamma4; values4{5} = Theta4;
values5{1} = Phi5; values5{2} = Psi5; values5{3} = Ni5;
%values5{4} = Gamma5; values5{5} = Theta5;
values6{1} = Phi6; values6{2} = Psi6; values6{3} = Ni6;

mi = M(1)/2;
values1{1} = Phi1(:,1:mi); values1{2} = Psi1(:,1:mi); values1{3} = Ni1(:,1:mi);
values2{1} = Phi2(:,1:mi); values2{2} = Psi2(:,1:mi); values2{3} = Ni2(:,1:mi);
values3{1} = Phi3(:,1:mi); values3{2} = Psi3(:,1:mi); values3{3} = Ni3(:,1:mi);
values4{1} = Phi4(:,1:mi); values4{2} = Psi4(:,1:mi); values4{3} = Ni4(:,1:mi);
values5{1} = Phi5(:,1:mi); values5{2} = Psi5(:,1:mi); values5{3} = Ni5(:,1:mi);
values6{1} = Phi1(:,mi+1:end); values6{2} = Psi1(:,mi+1:end); values6{3} = Ni1(:,mi+1:end);
values7{1} = Phi2(:,mi+1:end); values7{2} = Psi2(:,mi+1:end); values7{3} = Ni2(:,mi+1:end);
values8{1} = Phi3(:,mi+1:end); values8{2} = Psi3(:,mi+1:end); values8{3} = Ni3(:,mi+1:end);
values9{1} = Phi4(:,mi+1:end); values9{2} = Psi4(:,mi+1:end); values9{3} = Ni4(:,mi+1:end);
values10{1} = Phi5(:,mi+1:end); values10{2} = Psi5(:,mi+1:end); values10{3} = Ni5(:,mi+1:end);
M = M/2;

for i = 1:d
    [~, R1{i}] = qr(values1{i},0);
    [~, R2{i}] = qr(values2{i},0);
    [~, R3{i}] = qr(values3{i},0);
    [~, R4{i}] = qr(values4{i},0);
    [~, R5{i}] = qr(values5{i},0);
    [~, R6{i}] = qr(values6{i},0);
    [~, R7{i}] = qr(values7{i},0);
    [~, R8{i}] = qr(values8{i},0);
    [~, R9{i}] = qr(values9{i},0);
    [~, R10{i}] = qr(values10{i},0);


    % [cond(R1{i}), cond(R2{i}), cond(R3{i})],
    % pause,

    %[m, ind] = min([cond(R1{i}), cond(R2{i}), cond(R3{i}),...
     %   cond(R4{i}), cond(R5{i})]);
    [m, ind] = min([cond(R1{i}), cond(R2{i}), cond(R3{i}), ...
        cond(R4{i}), cond(R5{i}), cond(R6{i}), cond(R7{i}), ...
        cond(R8{i}), cond(R9{i}), cond(R10{i})]);
    %[m, ind] = min([cond(R1{i}), cond(R2{i})]);
    rr = eval(['R',num2str(ind)]);
    
    % rr = R2;
    new_values1{i} = values1{i}/rr{i};
    new_values2{i} = values2{i}/rr{i};
    new_values3{i} = values3{i}/rr{i};
    new_values4{i} = values4{i}/rr{i};
    new_values5{i} = values5{i}/rr{i};
    new_values6{i} = values6{i}/rr{i};
    new_values7{i} = values7{i}/rr{i};
    new_values8{i} = values8{i}/rr{i};
    new_values9{i} = values9{i}/rr{i};
    new_values10{i} = values10{i}/rr{i};
end




new_values{1} = new_values1;
new_values{2} = new_values2;
new_values{3} = new_values3; 
new_values{4} = new_values4;
new_values{5} = new_values5;
new_values{6} = new_values6;
new_values{7} = new_values7;
new_values{8} = new_values8; 
new_values{9} = new_values9;
new_values{10} = new_values10;

values{1} = values1;
values{2} = values2;
values{3} = values3;
values{4} = values4;
values{5} = values5;
values{6} = values6;
values{7} = values7;
values{8} = values8;
values{9} = values9;
values{10} = values10;

Params.tol = 1e-6;
Params.imax = 100;
Params.tol_tr = 1e-8;
Params.r = 1000;
t_tot =0;
res_tot = [];

for k = 1:10

clearvars -except d values new_values M Params rr t_tot res_tot


addpath('./Data')
load mnist_all
addpath('./oseledets_TT-Toolbox')

fprintf('Recognising a 0: \n')
im0 = double(test0(randi(500),:)');
rhs1 = reshape(im0,28*28,1);
rhs1 = rhs1/norm(rhs1);

% rhs_vec = {rhs1, rhs1, rhs1, rhs1, rhs1};
rhs_vec = {rhs1, rhs1, rhs1};
F = tt_tensor(rhs_vec);
a = rhs1;

X = tt_zeros(M,d);


tic;
[Y2, Res2pcQr, iter3] = TT_Tensorized_LSQR(new_values, F, Params, X);
% [Y2, Res2pcQr, iter3] = Sketch_TT_Tensorized_LSQR(new_values, F, Params, X);
t_tt_prec = toc
t_tot = t_tot + t_tt_prec;

for j = 1:Y2.d
    Y2 = ttm(Y2, j, inv(rr{j})');
end

ly = OpL(values, Y2);
ltly = OpL_T(values, ly);
ltf = OpL_T(values, F);
res_vero = norm(ltf- ltly)/norm(ltf),
res_tot = [res_tot; res_vero];

end

t_tot/10,
sum(res_tot)/10,
