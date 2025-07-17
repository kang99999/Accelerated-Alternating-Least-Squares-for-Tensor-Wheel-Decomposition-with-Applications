function C = like_kronec(A,B)
% 类K-R积代码
[r11,r12,r13,r14] = size(A);
[~,~,r23,r24] = size(B);
A = permute(A,[1 3 4 2]);
A = reshape(A,r11*r13,r14,r12);
B = permute(B,[1 3 4 2]);
B = reshape(B,r14,r23*r24,r12);
C = mtimesx(A,B);
C = reshape(C,r11,r13,r23,r24,r12);
C = permute(C,[1 5 2 3 4]);
C = reshape(C,r11,r12,r13*r23,r24);
end
