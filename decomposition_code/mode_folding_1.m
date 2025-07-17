function Xn = mode_folding_1(X,ranks,n)
%%干嘛的，检查以下
N = length(ranks);
perm_vec_1 = ranks(n:N);
perm_vec_2 = ranks(1:n-1);
perm_vecc = [perm_vec_1 perm_vec_2];
perm_vec = [N-n+2:N 1:N-n+1];
Xn = reshape(X,perm_vecc);
Xn = permute(Xn,perm_vec);
end