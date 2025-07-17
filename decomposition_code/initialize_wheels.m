function [wheels] = initialize_wheels(sz, ranks, ranks_l, varargin)
% 验证完毕,注意这里产生是rank的最末一位作为r1
% wheels = initialize_wheels(sz, ranks, ranks_l)
% Handle optional inputs
params = inputParser;
addParameter(params, 'init_zero', false, @isscalar);
parse(params, varargin{:});
init_zero = params.Results.init_zero;

% Main code
N = length(sz);
wheels = cell(1,N);
for k = 1:N
    R1 = ranks(k);
    if k == N
        R2 = ranks(1);
    else
        R2 = ranks(k+1);
    end
    if init_zero
        wheels{k} = zeros(R1, sz(k), ranks_l(k), R2);
    else
        wheels{k} = randn(R1, sz(k), ranks_l(k), R2);
    end
end

end