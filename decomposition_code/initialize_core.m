function [core] = initialize_core(ranks_l, varargin)
% 验证完毕core = initialize_core(ranks_l)
% Handle optional inputs
params = inputParser;
addParameter(params, 'init_zero', false, @isscalar);
parse(params, varargin{:});
init_zero = params.Results.init_zero;

% Main code
core = cell(1);
if init_zero
    core{1} = zeros(ranks_l);
else
    core{1} = randn(ranks_l);
end

end

