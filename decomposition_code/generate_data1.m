function [X ,wheels ,core] = generate_data1(sz, ranks, ranks_l, noise, col,varargin)
% generate_osman Generate dense (TW) low-rank tensor
% 验证完毕  带有共线性

% Handle optional inputs
params = inputParser;
addParameter(params, 'large_elem', 0);
parse(params, varargin{:});
large_elem = params.Results.large_elem;

% Construct tensor
N = length(sz);
wheels = cell(1,N);
core = cell(1,1);
core{1} = randn(ranks_l);
% core{1}=reshape(matrandcong(prod(ranks_l(1:N-1)),ranks_l(N),col),ranks_l);
for n = 1:N
    R0 = ranks(mod(n-2, N)+1);
    R1 = ranks(n);
%     wheels{n} = randn(R0, sz(n), ranks_l(n), R1);
    for i=1:R0*R1*ranks_l(n)
        for j=1:R0*R1*ranks_l(n)
            cor(i,j)=((col)^abs(i-j));
        end
    end
%     CC=matrandcong(sz(n),R0*R1*ranks_l(n),col);
    CC=mvtrnd(cor,1,30);
    wheels{n}=permute(reshape(CC,[sz(n),R0,R1,ranks_l(n)]),[2,1,4,3]);
    if large_elem > 0
        r0 = randsample(R0, 1);
        i = randsample(sz(n), 1);
        j = randsample(ranks_l(n), 1);
        r1 = randsample(R1, 1);
        wheels{n}(r0,i,j,r1) = large_elem;
    end
end

X = tensor_core_tensor(wheels, core, ranks_l);
Y=randn(sz);
X = X + noise*(norm(X(:))/norm(Y(:)))*Y;

end
