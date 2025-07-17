function X = tensor_core_tensor(wheels, core, ranks_l, varargin)
%tensor_core_tensor Convertes TW tensor given by cores to full dense tensor
%这个是合成轮张量和核张量的代码
%验证完毕

% Optional parameters
params = inputParser;
addParameter(params, 'permute_for_speed', false);
parse(params, varargin{:});
permute_for_speed = params.Results.permute_for_speed;

N = length(wheels);
sz = cellfun(@(x) size(x,2), wheels);

if permute_for_speed
    [~, max_idx] = max(sz);
    perm_vec = mod((max_idx+1 : max_idx+N) - 1, N) + 1;
    inv_perm_vec(perm_vec) = 1:N;
    core = core(perm_vec);
    sz = sz(perm_vec);
end

if isa(wheels{1}, 'double') 
    NC = subchain_matrix_2(wheels, ranks_l);
    X = mode_unfolding_2(core{1}) * NC;
    X = reshape(X, sz(:)');
elseif isa(core{1}, 'sptensor')
    
end

if permute_for_speed
    X = permute(X, inv_perm_vec);
end

end
