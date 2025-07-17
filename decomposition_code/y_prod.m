function [C] = y_prod(A,B)
    %% test A = randn([2 3 4 5 6 7]); B = randn([4 2 3 7 5 6]);
    [a1,a2,~,a3,a4,~] = size(A);
    [~,b1,b2,~,b3,b4] = size(B);
    C = tensorprod(A,B,[3 6],[1 4]);
    C = permute(C,[1 2 5 6 3 4 7 8]);
    szz = horzcat(a1,a2*b1,b2,a3,a4*b3,b4);
    C = reshape(C,szz);
end