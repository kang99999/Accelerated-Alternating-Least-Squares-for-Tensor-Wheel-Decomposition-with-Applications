function [core, wheels, total_time, final_error, varargout] = FTW_ALS(X, ranks, ranks_l, varargin)
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

N = length(sz);
P = cell(N,1);
iter_time(1) = 0;
for j=1:N
    P{j}=tensorprod(wheels{j},wheels{j},2,2);
end
for it = 1:maxiters
    iniIterTime = tic;
%   update 因子G
%     XM=0;
    for n = 1:N
        %计算XM:I_n*R_n*L_n*R_(n+1)
        if n==1
            XM=tensorprod(X,wheels{2},2,2);
            for j=3:N
                XM=tensorprod(XM,wheels{j},[2,N+2],[2,1]);
            end
            XM=tensorprod(XM,core{1},3:N+1,2:N);%I1 R2 R1 L1
            XM=permute(XM,[1,3,4,2]);
        elseif n==N
            XM=tensorprod(X,wheels{1},1,2);
            for j=2:N-1
                XM=tensorprod(XM,wheels{j},[1,N+2],[2,1]);
            end
            XM=tensorprod(XM,core{1},3:N+1,1:N-1);
            XM=permute(XM,[1,3,4,2]);
        else
            XM=tensorprod(X,wheels{1},1,2);
            for j=2:n-1
                XM=tensorprod(XM,wheels{j},[1,N+2],[2,1]);
            end
            if length(n+1:N)>1 %two case: >1/=1
                XM=tensorprod(XM,wheels{n+1},2,2);
                for j=n+2:N-1
                    XM=tensorprod(XM,wheels{j},[2,N+4],[2,1]);
                end
                XM=tensorprod(XM,wheels{N},[2,3,ndims(XM)],[2,4,1]);
            else
                XM=tensorprod(XM,wheels{N},[2,3],[2,4]);%n=N-1
            end
            XM=tensorprod(XM,core{1},[2:n,n+3:N+2],[1:n-1,n+1:N]);
            XM=permute(XM,[1,2,4,3]);%I_n R_n L_n R_(n+1)
        end
        % M*M:R_n*L_n*R_(n+1)*R_n*L_n*R_(n+1)
        if n==1
            rMM=P{2};
            for j = 3:N
                rMM = y_prod(rMM,P{j});
            end
            unC=reshape(core{1},[ranks_l(1),prod(ranks_l(2:N))]);
            MM=tensorprod(rMM,unC,2,2);
            MM=tensorprod(MM,unC,4,2);
            MM=permute(MM,[2,5,1,4,6,3]);
        elseif n==N
            lMM=P{1};
            for j = 2:N-1
                lMM = y_prod(lMM,P{j});
            end
            unC=reshape(core{1},[prod(ranks_l(1:N-1)),ranks_l(N)]);
            MM=tensorprod(lMM,unC,2,1);
            MM=tensorprod(MM,unC,4,1);
            MM=permute(MM,[2,5,1,4,6,3]);
        else
            lMM=P{1};
            for j = 2:n-1
                lMM = y_prod(lMM,P{j});
            end
            rMM=P{n+1};
            for j = n+2:N
                rMM = y_prod(rMM,P{j});
            end
            unC=reshape(core{1},[prod(ranks_l(1:n-1)),ranks_l(n),prod(ranks_l(n+1:N))]);
            MM=tensorprod(lMM,unC,2,1);
            MM=tensorprod(MM,rMM,[1,3,7],[3,6,2]);
            MM=tensorprod(MM,unC,[2,7],[1,3]);
            MM=permute(MM,[1,3,4,2,6,5]);
        end
        [r1,~,l1,r2]=size(wheels{n});
        XM=reshape(XM,[sz(n),r1*l1*r2]);
        MM=reshape(MM,[r1*l1*r2,r1*l1*r2]);
        wheels{n}=permute(reshape(XM/(MM),[sz(n),r1,l1,r2]),[2 1 3 4]);
        P{n}=tensorprod(wheels{n},wheels{n},2,2);

        

    end
    %计算 XN:L1*..*LN
    XN=tensorprod(X,wheels{1},1,2);
    for j=2:N-1
        XN=tensorprod(XN,wheels{j},[1,N+2],[2,1]);
    end
    XN=tensorprod(XN,wheels{N},[1,2,N+2],[2,4,1]);
%     XM=reshape(XM,[sz(N),ranks(N),ranks_l(N),ranks(1)]);
%     XN=tensorprod(XM,wheels{N},[1,2,4],[2,1,4]);
    % NN:L1..LN*L1..LN
    NN=P{1};
    for j=2:N-1
        NN=y_prod(NN,P{j});
    end
    NN=tensorprod(NN,P{N},[1,3,4,6],[3,1,6,4]);
    NN=reshape(permute(NN,[1,3,2,4]),[prod(ranks_l),prod(ranks_l)]);
%     NN=tensorprod(NN,eye(ranks(1)),[1,3],[1,2]);
%     NN=tensorprod(NN,eye(ranks(1)),[2,4],[1,2]);
    XN=reshape(XN,[1 prod(ranks_l)]);
    core{1}=reshape(XN/(NN),ranks_l);
    iter_time(it+1) = iter_time(it) + toc(iniIterTime);

    % Stop criterion: the original result also illustrates that one should avoid stopping too early when using ALS.
    % 
    % Check convergence: Relative error
    if tol > 0 && strcmp(conv_crit, 'relative error')
        
        % 这里是合成一个张量
        NOC=wheels{1};
        for n=2:N-1
            NOC=tensorprod(NOC,wheels{n},2*n,1);
        end
        NOC=tensorprod(NOC,wheels{N},[2*N,1],[1,4]);
        Y=tensorprod(NOC,core{1},2:2:2*N,1:N);
%         Y = tensor_core_tensor2(NOC.', core, sz);

        % Compute current relative error
        er = norm(Y(:)-X(:))/norm(X(:));
%         ZTZ=Z.'*Z;
%         er=sqrt((dot_X-2*dot(XM(:),Z(:))+dot(MTM(:),ZTZ(:)))/dot_X);
%         final_error = er;
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
