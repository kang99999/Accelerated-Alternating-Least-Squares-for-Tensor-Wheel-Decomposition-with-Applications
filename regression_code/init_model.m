function [model,core,wheels] = init_model(para, ranks, ranks_l)

% Main code
sz = para.dim;
N = length(sz);
core{1} = randn(ranks_l);
wheels = cell(N,1);
for n = 1:N-1
    wheels{n} = randn(ranks(n), sz(n), ranks_l(n), ranks(n+1));  
end
wheels{N} = randn(ranks(N), sz(N), ranks_l(N), ranks(1));
model = tensor_core_tensor(wheels,core,ranks_l);
end

