function Xn = mode_unfolding_2(X)
%这里其实展成了一个向量

Xn = reshape(X,1, numel(X));

if isa(Xn, 'sptensor')
    Xn = sptensor_mat_2_sparse(Xn);
end

end
