function Wr = IAS_K2W(K,W,k)
% IAS_K2W - insert a kernel into a weight array
% On input:
%     K (5x5 array): kernel
%     W (weight array 5x5x1x4): weight array
%     k (int): selects which 5x5 weight array
% On output:
%     Wr (5x5x1x4 array): weights array
% Call:
%     Wr = IAS_K2W(W,K,2);
% Author:
%     T. Henderson
%     UU
%     Spring 2025
%

Wr = W;

for r = 1:5
    for c = 1:5
        Wr(r,c,1,k) = K(r,c);
    end
end
