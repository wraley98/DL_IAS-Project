function W = IAS_extract_weights(net,layer)
% IAS_extract_weights - extract kernel weights of gien layer
% On input:
%     net (neural net structure): CNN
%     layer (int): conv layer
% On output:
%     W (weight array): weights for convolutions on layer
% Call:
%     W = IAS_extract_weights(net,2);
% Author:
%     T. Henderson
%     UU
%     Spring 2025
%

W = net.Layers(layer).Weights;
