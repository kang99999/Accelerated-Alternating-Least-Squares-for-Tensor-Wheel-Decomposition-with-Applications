function NC = subchain_matrix_2(wheels, ranks_l)
%%压缩乘积的方式
N = length(wheels);
for j = 1:N-1
    if j == 1
        NC = wheels{1};
    else
        NC = tensorprod(NC,wheels{j},2*j,1);
    end
end
NC =  tensorprod(NC,wheels{N},[1 2*N],[4 1]);
v = [2:2:2*N 1:2:2*N-1];
NC = permute(NC,v);
szzz = prod(ranks_l);
NC = reshape(NC,szzz,numel(NC)/szzz);

end





