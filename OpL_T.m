function[Y] = OpL_T(values, X)

d = X.d;
n_terms = length(values);

for k = 1:n_terms
    T{k} = X;
    for j = 1:d
        T{k} = ttm(T{k}, j, values{k}{j});
    end
end
Y = T{1};
for k = 2:n_terms
    Y = Y + T{k};
end
end