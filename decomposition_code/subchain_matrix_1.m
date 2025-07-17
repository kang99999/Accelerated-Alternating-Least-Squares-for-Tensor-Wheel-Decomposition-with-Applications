function MM = subchain_matrix_1(wheels, core, n)
%%压缩乘积做法,验证重排问题
N = length(wheels);
idx = [n+1:N 1:n-1];
for j = 1:length(idx)
    if j == 1
        MM = wheels{idx(j)};
    else
        MM = tensorprod(MM,wheels{idx(j)},2*j,1);
    end
end

MM =  tensorprod(MM,core{1},3:2:2*N,idx);
m = [N+1 N+2 1:N];
MM = permute(MM,m);

for i = 1:3
    if i == 1
        szz = size(MM,1);
    else
        szz = szz * size(MM,i);
    end
end
MM = reshape(MM,szz,numel(MM)/szz);

end



