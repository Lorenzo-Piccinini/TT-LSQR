function[X] = OperatorLT(termsT, Y)

l = length(termsT);
coeffs = ones(l,1);
temp = cell(l,1);
for i = 1:l
    temp{i} = TTKronOp(termsT{i}, Y);
end

X = TTsum_Randomize_then_Orthogonalize(temp, coeffs);

end