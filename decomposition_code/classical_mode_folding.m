function X = classical_mode_folding(Xn, sz)
%经典mode-n展开的逆运算，不具有一般性，还原最小二乘问题一的结果

perm_vec = [2 1 3 4];
perm_vec_sz = [2 1 3 4];
X = permute(reshape(Xn, sz(perm_vec_sz)), perm_vec);

end
