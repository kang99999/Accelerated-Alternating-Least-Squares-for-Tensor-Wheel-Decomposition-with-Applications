function [core, wheels, total_time, final_error, varargout] = RFTW_ALS(X, ranks, ranks_l, M,varargin)
%% TW_ALS_NE未加结构的正规方程
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
R=cell(N,1);
Q=cell(N,1);
P=cell(N,1);

%计算QR分解
for n=1:N
    if M(n)==sz(n)
        Q{n}=eye(sz(n));
        R{n}=wheels{n};
    else
        Y=X;
        for i=1:n-1
            Y=tensorprod(Y,randn(M(i),sz(i)),1,2);
        end
        for i=n+1:N
            Y=tensorprod(Y,randn(M(i),sz(i)),2,2);
        end
        Y=reshape(Y,sz(n),[]);
        [u,~,~]=svd(Y,'econ');
        Q{n}=u(:,1:M(n));
        R{n}=permute(tensorprod(wheels{n},Q{n},2,1),[1,4,2,3]);
    end
end

for j=1:N
    P{j}=tensorprod(R{j},R{j},2,2);
end
XX=cell(N+1,1);
for n=1:N+1
    if n~=N+1
        XX{n}=X;
        for i=1:n-1
            XX{n}=tensorprod(XX{n},Q{i},1,1);
        end
        for i=n+1:N
            XX{n}=tensorprod(XX{n},Q{i},2,1);
        end
        XX{n}=ipermute(XX{n},[n,1:n-1,n+1:N]);
    else
        XX{N+1}=tensorprod(XX{N},Q{N},N,1);
    end
end
iter_time(1) = 0;
for it = 1:maxiters
    iniIterTime = tic;
    %更新G
    Y=0;
    for n = 1:N
        % 计算系数矩阵
        Y=XX{n};

        if n==1
            YV=tensorprod(Y,R{2},2,2);
            for j=3:N
                YV=tensorprod(YV,R{j},[2,N+2],[2,1]);
            end
            YV=tensorprod(YV,core{1},3:N+1,2:N);%I1 R2 R1 L1
            YV=permute(YV,[1,3,4,2]);
        elseif n==N
            YV=tensorprod(Y,R{1},1,2);
            for j=2:N-1
                YV=tensorprod(YV,R{j},[1,N+2],[2,1]);
            end
            YV=tensorprod(YV,core{1},3:N+1,1:N-1);
            YV=permute(YV,[1,3,4,2]);
        else
            YV=tensorprod(Y,R{1},1,2);
            for j=2:n-1
                YV=tensorprod(YV,R{j},[1,N+2],[2,1]);
            end
            if length(n+1:N)>1 %two case: >1/=1
                YV=tensorprod(YV,R{n+1},2,2);
                for j=n+2:N-1
                    YV=tensorprod(YV,R{j},[2,N+4],[2,1]);
                end
                YV=tensorprod(YV,R{N},[2,3,ndims(YV)],[2,4,1]);
            else
                YV=tensorprod(YV,R{N},[2,3],[2,4]);%n=N-1
            end
            YV=tensorprod(YV,core{1},[2:n,n+3:N+2],[1:n-1,n+1:N]);
            YV=permute(YV,[1,2,4,3]);%I_n R_n L_n R_(n+1)
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
        YV=reshape(YV,[sz(n),r1*l1*r2]);
        MM=reshape(MM,[r1*l1*r2,r1*l1*r2]);
        wheels{n}=permute(reshape(YV/(MM),[sz(n),r1,l1,r2]),[2 1 3 4]);
        R{n}=permute(tensorprod(wheels{n},Q{n},2,1),[1,4,2,3]);
        P{n}=tensorprod(R{n},R{n},2,2);
               
    end
% I_n*R_n*L_n*R_(n+1)
    %计算系数矩阵
%     U=subchain_matrix_2(R,ranks_l).';
    for j = 1:N
        P{j} = tensorprod(R{j},R{j},2,2);
    end

    UTU = P{1};
    for j = 2:N
        UTU = y_prod(UTU,P{j});
    end 
    UTU=tensorprod(UTU,eye(ranks(1)),[1,3],[1,2]);
    UTU=tensorprod(UTU,eye(ranks(1)),[2,4],[1,2]);

    Y=XX{N+1};
    F=tensorprod(Y,R{1},1,2);
    for j=2:N-1
        F=tensorprod(F,R{j},[1,N+2],[2,1]);
    end
    F=tensorprod(F,R{N},[1,2,N+2],[2,4,1]);
    F=reshape(F,[1 prod(ranks_l)]);
%     unY=mode_unfolding_2(Y);
%     F=unY*U;
    ZZ=F/(UTU);
    core{1}=reshape(ZZ,size(core{1}));


    iter_time(it+1) = iter_time(it) + toc(iniIterTime);

    % Stop criterion: the original result also illustrates that one should avoid stopping too early when using ALS.
    % 
    % Check convergence: Relative error
    if tol > 0 && strcmp(conv_crit, 'relative error')
        
        % 这里是合成一个张量
%         YY = tensor_core_tensor(wheels, core, ranks_l);
        NOC=wheels{1};
        for n=2:N-1
            NOC=tensorprod(NOC,wheels{n},2*n,1);
        end
        NOC=tensorprod(NOC,wheels{N},[2*N,1],[1,4]);
        YY=tensorprod(NOC,core{1},2:2:2*N,1:N);
        % Compute current relative error
        er = norm(YY(:)-X(:))/norm(X(:));
%         GTG=Z.'*Z;
%         er=sqrt((dot_X-2*dot(YV(:),Z(:))+dot(VTV(:),GTG(:)))/dot_X);
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
YY = tensor_core_tensor(wheels, core, ranks_l);
final_error = norm(YY(:)-X(:))/norm(X(:));

if nargout > 3 && exist('conv_vec','var') && exist('iter_time','var')
    varargout{1} = conv_vec(1:it);
    varargout{2} = iter_time(2:it+1);
else
    varargout{1} = nan;
    varargout{2} = nan;
end



end
