function Xn = mode_unfolding_1(X, n)
%mode_unfolding Does mode-n unfolding of input tensor X

sz = size(X);
N = length(sz);
perm_vec = [n n+1:N 1:n-1];
Xn = reshape(permute(X, perm_vec), [sz(n), prod(sz)/sz(n)]);

if isa(Xn, 'sptensor')
    Xn = sptensor_mat_2_sparse(Xn);
end

end
