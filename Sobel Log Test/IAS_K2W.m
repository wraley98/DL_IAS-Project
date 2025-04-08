function W = IAS_K2W(Kernel , sizeArr)
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

M = sizeArr(1);
N = sizeArr(2);
X = sizeArr(3);
Y = sizeArr(4);

W = zeros(sizeArr);

index = 1;

for k = 1:Y
    for ch = 1:X
               W(:,:,ch,k)  = Kernel(index).kernel;
               index = index + 1;
    end
end
