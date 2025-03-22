function K = IAS_W2K(W,k)
% IAS_W2K - convert a weight array to a 2D kernel
% On input:
%     W (weight array MxNx1x4): weight array
%     k (int): selects which MxN weight array
% On output:
%     K (MxN array): kernel
% Call:
%     K = IAS_W2K(W,1);
% Author:
%     T. Henderson
%     UU
%     Spring 2025
%

[M,N,X,Y] = size(W);
K = zeros(M,N);

for r = 1:M
    for c = 1:N
        K(r,c) = W(r,c,1,k);
    end
end
