function[Y] = OpL(values, X)


d = X.d;
n_terms = length(values);


for k = 1:n_terms
    wrk = X;
    for j = 1:d
        wrk = ttm(wrk, j, values{k}{j}');
    end
    if k==1, Y=wrk; else, Y = Y + wrk;end
end
