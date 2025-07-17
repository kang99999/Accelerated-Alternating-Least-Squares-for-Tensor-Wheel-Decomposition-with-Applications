function [core, wheels, total_time, nopre_time, final_error, varargout] = TW_ALS_Sampled(X,ranks, ranks_l, embedding_dims_1, embedding_dims_2, varargin)

iniTotalTime = tic;
%% Add relevant paths

mtimesx('SPEED');


% size of tensor
sz = size(X);

%% Handle inputs

% Optional inputs
params = inputParser;
addParameter(params, 'conv_crit', 'relative error');
addParameter(params, 'tol', 1e-6, @isscalar);
addParameter(params, 'maxiters', 50, @(x) isscalar(x) & x > 0);
addParameter(params, 'resample', true, @isscalar);
addParameter(params, 'verbose', true, @isscalar);
addParameter(params, 'no_mat_inc', false);
addParameter(params, 'breakup', false);
addParameter(params, 'alpha', 0);
addParameter(params, 'uniform_sampling', false);
addParameter(params, 'Iwheels', initialize_wheels(sz, ranks, ranks_l), @iscell);
addParameter(params, 'Icore', initialize_core(ranks_l), @iscell);
parse(params, varargin{:});

conv_crit = params.Results.conv_crit;
tol = params.Results.tol;
maxiters = params.Results.maxiters;
resample = params.Results.resample;
verbose = params.Results.verbose;
no_mat_inc = params.Results.no_mat_inc;
breakup = params.Results.breakup;
alpha = params.Results.alpha;
uniform_sampling = params.Results.uniform_sampling;
Iwheels = params.Results.Iwheels;
Icore = params.Results.Icore;

%% Check if X is path to mat file on disk
%   X_mat_flag is a flag that keeps track of if X is an array or path to
%   a mat file on disk. In the latter case, X_mat will be a matfile that
%   can be used to access elements of the mat file.
if isa(X, 'char') || isa(X, 'string')
    X_mat_flag = true;
    X_mat = matfile(X, 'Writable', false);
else
    X_mat_flag = false;
end

%% Initialize cores, sampling probabilities and sampled cores

if X_mat_flag
    sz = size(X_mat, 'Y');
    N = length(sz);
    col_cell = cell(1,N);
    for n = 1:N
        col_cell{n} = ':';
    end

    % If value for no_mat_inc is provided, make sure it is a properly shaped
    % vector.
    if no_mat_inc(1)
        if ~(size(no_mat_inc,1)==1 && size(no_mat_inc,2)==N)
            no_mat_inc = no_mat_inc(1)*ones(1,N);
        end
    end
else
    N = length(sz);
end

core = Icore;
wheels = Iwheels;
%%计算抽样概率
sampling_probs = cell(1, N);
for n = 2:N
    if uniform_sampling
        sampling_probs{n} = ones(sz(n), 1)/sz(n);
    else
        U = col(classical_mode_unfolding(wheels{n}, 2));
        sampling_probs{n} = sum(abs(U).^2, 2)/size(U, 2);
    end
end
wheels_samples = cell(1, N);

if ~breakup(1)
    breakup = ones(1,N);
end

slow_idx = cell(1,N);
sz_shifted = [1 sz(1:end-1)];
idx_prod = cumprod(sz_shifted);
sz_pts = cell(1,N);
for n = 1:N
    sz_pts{n} = round(linspace(0, sz(n), breakup(n)+1));
    slow_idx{n} = cell(1,breakup(n));
    for brk = 1:breakup(n)
        J_1 = embedding_dims_1(n);
        samples_lin_idx_2 = prod(sz_shifted(1:n))*(sz_pts{n}(brk):sz_pts{n}(brk+1)-1).';
        slow_idx{n}{brk} = repelem(samples_lin_idx_2, J_1, 1);
    end
end

if nargout > 1 && tol > 0 && strcmp(conv_crit, 'relative error')
    conv_vec = zeros(1, maxiters);
    iter_time = zeros(1, maxiters+1);
end

%% Main loop
% Iterate until convergence, for a maximum of maxiters iterations

if ~resample
    J_1 = embedding_dims(1); % Always use same embedding dim
    samples = nan(J_1, N);

    for m = 2:N-1
        samples(:, m) = randsample(sz(m), J_1, true, sampling_probs{m});
        wheels_samples{m} = wheels{m}(:, samples(:,m), :, :);
    end
end

iniNopreTime = tic;
iter_time(1) = 0;
for it = 1:maxiters
    iniIterTime = tic;
    % Inner for loop
    for n = 1:N

        % Construct sketch and sample wheels
        if resample % Resample all wheels, except nth which will be updated
            J_1 = embedding_dims_1(n);
            samples = nan(J_1, N);
            for m = 1:N
                if m ~= n
                    samples(:, m) = randsample(sz(m), J_1, true, sampling_probs{m});
                    wheels_samples{m} = wheels{m}(:, samples(:,m), :, :);
                end
            end
        else % Only resample the core that was updated in last iteration
            m = mod(n-2,N)+1;
            samples(:, m) = randsample(sz(m), J_1, true, sampling_probs{m});
            wheels_samples{m} = wheels{m}(:, samples(:,m), :, :);
        end

        % Compute the row rescaling factors
        rescaling = ones(J_1, 1);
        for m = 1:N
            if m ~= n
                rescaling = rescaling ./ sqrt(sampling_probs{m}(samples(:, m)));
            end
        end
        rescaling = rescaling ./ sqrt(J_1);

        % Construct sketched design matrix
        idx = [n+1:N 1:n-1]; 
        G_sketch = wheels_samples{idx(1)};
        for m = 2:N-1
            G_sketch = like_kronec(G_sketch,wheels_samples{idx(m)});
        end
        core{1} = mode_unfolding_1(core{1},n);
        G_sketch = tensorprod(G_sketch,core{1},3,2);
        core{1} = mode_folding_1(core{1},ranks_l,n);
        G_sketch = mode_unfolding_1(G_sketch, 2);
        G_sketch = rescaling .* G_sketch;
        if breakup(n) > 1
            if alpha > 0
                [L, U, p] = lu(G_sketch.'*G_sketch + alpha*eye(size(G_sketch,2)), 'vector');
            else
                [L, U, p] = lu(G_sketch, 'vector');
            end
            ZT = zeros(size(G_sketch,2), sz(n));
        end

        % Sample right hand side
        for brk = 1:breakup(n)
            no_cols = sz_pts{n}(brk+1)-sz_pts{n}(brk);
            samples_lin_idx_1 = 1 + (samples(:, idx)-1) * idx_prod(idx).';
            samples_lin_idx = repmat(samples_lin_idx_1, no_cols, 1) + slow_idx{n}{brk};
            X_sampled = X(samples_lin_idx);
            Xn_sketch = reshape(X_sampled, J_1, no_cols);

            % Rescale right hand side
            Xn_sketch = rescaling .* Xn_sketch;

            if breakup(n) > 1
                if alpha > 0
                    ZT(:, sz_pts{n}(brk)+1:sz_pts{n}(brk+1)) = U \ (L \ G_sketch(:,p).'*Xn_sketch);
                else
                    ZT(:, sz_pts{n}(brk)+1:sz_pts{n}(brk+1)) = U \ (L \ Xn_sketch(p, :));
                end
            end
        end
        if breakup(n) > 1
            Z = ZT.';
        else
            if alpha > 0
                Z = (( G_sketch.'*G_sketch + alpha*eye(size(G_sketch,2)) ) \ ( G_sketch.'*Xn_sketch )).';
            else
                Z = ((G_sketch.'*G_sketch) \ G_sketch.'*Xn_sketch).';
            end
        end
%%
%         Z = (G_sketch \ Xn_sketch).';
        wheels{n} = classical_mode_folding(Z, size(wheels{n}));
%%
        % Update sampling distribution for wheels
        if uniform_sampling
            sampling_probs{n} = ones(sz(n), 1)/sz(n);
        else
            U = col(classical_mode_unfolding(wheels{n}, 2));
            sampling_probs{n} = sum(abs(U).^2, 2)/size(U, 2);
        end
    end

    if resample % Resample all wheels, except nth which will be updated
        J_2 = embedding_dims_2(n);
        samples = nan(J_2, N);
        for m = 1:N
            samples(:, m) = randsample(sz(m), J_2, true, sampling_probs{m});
            wheels_samples{m} = wheels{m}(:, samples(:,m), :, :);
        end
    else % Only resample the core that was updated in last iteration
        m = mod(n-2,N)+1;
        samples(:, m) = randsample(sz(m), J_2, true, sampling_probs{m});
        wheels_samples{m} = wheels{m}(:, samples(:,m), :, :);
    end

    % Compute the row rescaling factors
    rescaling = ones(J_2, 1);
    for m = 1:N
        rescaling = rescaling ./ sqrt(sampling_probs{m}(samples(:, m)));
    end
    rescaling = rescaling ./ sqrt(J_2);

    % Construct sketched design matrix
    N_sketch = wheels_samples{1};
    for m = 2:N
        N_sketch = like_kronec(N_sketch,wheels_samples{m});
    end
    N_sketch = tensor_trace(N_sketch);
    N_sketch = rescaling .* N_sketch;
    if breakup(n) > 1
        if alpha > 0
            [L, U, p] = lu(N_sketch.'*N_sketch + alpha*eye(size(N_sketch,2)), 'vector');
        else
            [L, U, p] = lu(N_sketch, 'vector');
        end
        ZT = zeros(size(N_sketch,2), sz(n));
    end

    % Sample right hand side
    X = tensor(X);
    X_sampled = X(samples);
    Xn_sketch = X_sampled;
   
    % Rescale right hand side
    Xn_sketch = rescaling .* Xn_sketch;

    %加不加
    
    %解方程
    ZZ = (N_sketch \ Xn_sketch).';
    core{1} = reshape(ZZ, size(core{1}));
    
    % Update sampling distribution for core 这里完全不必要了
    if uniform_sampling
        sampling_probs{n} = ones(sz(n), 1)/sz(n);
    else
        U = col(classical_mode_unfolding(wheels{n}, 2));
        sampling_probs{n} = sum(abs(U).^2, 2)/size(U, 2);
    end

    iter_time(it+1) = iter_time(it) + toc(iniIterTime);

    % Stop criterion: the original result also illustrates that one should avoid stopping too early when using ALS.
    %
    % Check convergence: Relative error
    if tol > 0 && strcmp(conv_crit, 'relative error')
%          这里是合成一个张量
        NOC=wheels{1};
        for n=2:N-1
            NOC=tensorprod(NOC,wheels{n},2*n,1);
        end
        NOC=tensorprod(NOC,wheels{N},[2*N,1],[1,4]);
        Y=tensorprod(NOC,core{1},2:2:2*N,1:N);
        % Compute full tensor corresponding to cores
%         Y = tensor_core_tensor(wheels, core, ranks_l);

        % Compute current relative error
        if X_mat_flag
            XX = X_mat.Y;
            er = norm(XX(:)-Y(:))/norm(XX(:));
        else
            er = norm(Y(:)-X(:))/norm(X(:));
%             er=mseT(Y,M,datatype);
        end
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
nopre_time = toc(iniNopreTime);
total_time = toc(iniTotalTime);
% final_error = er;
Y = tensor_core_tensor(wheels, core, ranks_l);
final_error = norm(Y(:)-X(:))/norm(X(:));
% final_error=mseT(Y,M,datatype);
if nargout > 3 && exist('conv_vec','var') && exist('iter_time','var')
    varargout{1} = conv_vec(1:it);
    varargout{2} = iter_time(2:it+1);
else
    varargout{1} = nan;
    varargout{2} = nan;
end

end
