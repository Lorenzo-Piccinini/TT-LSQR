function[Y] = OperatorL(terms, X)

l = length(terms);
coeffs = ones(l,1);
temp = cell(l,1);
for i = 1:l
    temp{i} = TTKronOp(terms{i}, X);
end

%Y = TTsum_Randomize_then_Orthogonalize(temp, coeffs);
Y = TTsum(temp,coeffs);
end