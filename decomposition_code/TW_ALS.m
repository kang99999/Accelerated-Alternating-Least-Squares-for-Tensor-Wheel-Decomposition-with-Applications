function [core, wheels, total_time, final_error,varargout] = TW_ALS(X,ranks, ranks_l, varargin)
%% TW_ALS
%% Handle optional inputs

sz = size(X);

params = inputParser;
addParameter(params, 'conv_crit', 'relative error');
addParameter(params, 'tol', 1e-3, @isscalar);
addParameter(params, 'maxiters', 50, @(x) isscalar(x) & x > 0);
addParameter(params, 'verbose', true, @isscalar);
addParameter(params, 'Iwheels', initialize_wheels(sz, ranks, ranks_l), @iscell);
addParameter(params, 'Icore', initialize_core(ranks_l), @iscell);
parse(params, varargin{:});

conv_crit = params.Results.conv_crit;
tol = params.Results.tol;
maxiters = params.Results.maxiters;
verbose = params.Results.verbose;
Iwheels = params.Results.Iwheels;
Icore = params.Results.Icore;

iniTotalTime = tic;
%% Initialize core and wheels
core = Icore;
wheels = Iwheels;

% nargout： the output variables of Function
if nargout > 1 && tol > 0 && strcmp(conv_crit, 'relative error')
    conv_vec = zeros(1, maxiters);
    iter_time = zeros(1, maxiters+1);
end

%% Main loop
% Iterate until convergence, for a maximum of maxiters iterations
% Y = tensor_core_tensor(wheels, core, ranks_l);
N = length(sz);
iter_time(1) = 0;

% final_error=1000;
for it = 1:maxiters
    iniIterTime = tic;
    for n = 1:N
        % 计算系数矩阵
        G = subchain_matrix_1(wheels, core, n).';
        
        % x的矩阵化
        XnT = mode_unfolding_1(X, n).';
        
        % 解方程
        Z = (G \ XnT).';
        wheels{n} = classical_mode_folding(Z, size(wheels{n}));        
    end
    %计算系数矩阵
    C = subchain_matrix_2(wheels, ranks_l).';
    %X的矩阵化
    XT = mode_unfolding_2(X).';
    %解方程
    ZZ = (C \ XT).';
    core{1} = reshape(ZZ, size(core{1}));

    iter_time(it+1) = iter_time(it) + toc(iniIterTime);

    % Stop criterion: the original result also illustrates that one should avoid stopping too early when using ALS.
    % 
    % Check convergence: Relative error
    if tol > 0 && strcmp(conv_crit, 'relative error')
        
        % 这里是合成一个张量
%         Y = tensor_core_tensor(wheels, core, ranks_l);
        NOC=wheels{1};
        for n=2:N-1
            NOC=tensorprod(NOC,wheels{n},2*n,1);
        end
        NOC=tensorprod(NOC,wheels{N},[2*N,1],[1,4]);
        Y=tensorprod(NOC,core{1},2:2:2*N,1:N);
        % Compute current relative error
        er = norm(Y(:)-X(:))/norm(X(:));
%         final_error=er;
        
        if verbose
            fprintf('\tRelative error after iteration %d: %.8f\n', it, er);
        end
        
        % Save current error to conv_vec if required
        if nargout > 1
            conv_vec(it) = er;
        end

        % Break if relative error below threshold
        if abs(er) < tol
            if verbose
                fprintf('\tRelative error below tol; terminating...\n');
            end
            break
        end


    % Check convergence: Norm change 
    % Compute norm of TR tensor using normTR()
    % Code accompanying the paper "On algorithms for and computing with the tensor ring decomposition"
    % We delete this part because we will not use it as the stop criterion.
    % elseif

        
    % Just print iteration count
    else
        if verbose
            fprintf('\tIteration %d complete\n', it);
        end
    end
    
end
total_time = toc(iniTotalTime);
% final_error = er;
Y = tensor_core_tensor(wheels, core, ranks_l);
final_error = norm(Y(:)-X(:))/norm(X(:));

if nargout > 3 && exist('conv_vec','var') && exist('iter_time','var')
    varargout{1} = conv_vec(1:it);
    varargout{2} = iter_time(2:it+1);
else
    varargout{1} = nan;
    varargout{2} = nan;
end



end
