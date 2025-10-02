function[col_n] = col_norms(values)

d = length(values);

[n,m] = size(values{1});
% We are assuming along each mode we have same dimensions

for k = 1:d
    vv{k} = sum(values{k}.*values{k}).^(0.5);
    vv{k} = vv{k}(:);
end

col_n = ones(m^d,1);
count = ones(d,1);

for k = 1:m^d
    %disp(k)
    %count'
    for i = d:-1:1
        col_n(k) = col_n(k)*vv{i}(count(i));
    end

    count(d) = count(d)+1;
    for i = d-1:-1:1
        if count(i+1) == m + 1
            count(i+1) = 1;
            count(i) = count(i) + 1;
        end
    end
        
        
end
