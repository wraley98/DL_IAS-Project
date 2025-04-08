function [weights, sizeArr] = IAS_W2K(W)
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
sizeArr = [M,N,X,Y];

index = 1;

for k = 1:Y
    for ch = 1:X
                weights(index).w = W(:,:,ch,k);
                index = index + 1;
    end
end
