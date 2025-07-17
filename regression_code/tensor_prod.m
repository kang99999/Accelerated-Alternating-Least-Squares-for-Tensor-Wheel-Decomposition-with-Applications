function [C] = tensor_prod(A,B)
    %% test A = randn([2 11 5 3]); B = randn([3 12 6 4]);
    [a0,a1,a2,~] = size(A);
    [~,b1,b2,b0] = size(B);
    C = tensorprod(A,B,4,1);
    C = permute(C,[1 2 4 3 5 6]);
    szz = horzcat(a0,a1*b1,a2*b2,b0);
    C = reshape(C,szz);
end