function [x_tt] = tensor_trace(x)
% x是4维张量，计算x的mode-1与mode-4的迹
% 代码没问题 三种都行，考虑哪种更快一点

N = size(x,1);
x_tt = x(1,:,:,1);
for i = 2:N
    x_tt = x_tt + x(i,:,:,i);
end
x_tt = reshape(x_tt, [size(x,2),size(x,3)]);


%[r1,r2,r3,r4] = size(x);
%x_selfcontract = zeros([r2 r3]);
%for i = 1:r2
%    for j = 1:r3
%        for k = 1:r1
%            x_selfcontract(i,j) = x_selfcontract(i,j) + x(k,i,j,k);
%        end
%    end 
%end

%[r1,r2,r3,~] = size(x);
%x_selfcontract = zeros([r2 r3]);
%for i = 1:r1
%    x_selfcontract = x_selfcontract + squeeze(x(i,:,:,i));
%end