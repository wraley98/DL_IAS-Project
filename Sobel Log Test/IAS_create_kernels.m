function [Sobelkernels , Logkernels] = IAS_create_kernels(M)
% IAS_create_kernels - create MxM kernels
%  * from fspecial:
%      average
%      disk
%      gaussian (with sigma = [0.5:0.5:3])
%      log (with sigma = [0.5:0.5:3])
%  * Fourier basis windows
%      u = [1:4]
%      v = [1:4]
%  * Gabor kernels
%      wavelengths (pixels/cycle): [2:5]
%      orientations (degrees): [0:20:359]
% On input:
%     dir (string): name of directory for kernel images
% On output:
%     kernels (kernels struct): kernels
% Call:
%     kernels = IAS_create_kernels;
% Author:
%     T. Henderson
%     UU
%     Spring 2025
%

H = M;   % kernel size

sigma = [0.5:0.5:2];
num_sigma = length(sigma);

u = [1:4];
num_u = length(u);
v = [1:4];
num_v = length(v);

wavelengths = [2:5];
num_wavelengths = length(wavelengths);
orientations = [0:20:359];
num_orientations = length(orientations);

count = 0;

count = count + 1;
K = [1 1 2 1 1; 1 1 2 1 1; 0 0 0 0 0; -1 -1 -2 -1 -1; -1 -1 -2 -1 -1];
%K = fspecial("sobel");
%K = imresize(K,[5,5],'bicubic');
Sobelkernels(count).kernel = K;
Sobelkernels(count).name = 'Sobel';
Sobelkernels(count).orientation = 0;

count = count + 1;
K = [0 1 1 1 1; -1 0 2 2 1; -1 -2 0 2 1; -1 -2 -2 0 1; -1 -1 -1 -1 0];
%K = fspecial("sobel");
%K = imrotate(K,45);
%K = imresize(K,[5,5],'bicubic');
Sobelkernels(count).kernel = K;
Sobelkernels(count).name = 'Sobel';
Sobelkernels(count).orientation = 45;

count = count + 1;
K = [-1 -1 0 1 1; -1 -1 0 1 1; -2 -2 0 2 2; -1 -1 0 1 1; -1 -1 0 1 1];
%K = fspecial("sobel");
%K = imrotate(K,90);
%K = imresize(K,[5,5],'bicubic');
Sobelkernels(count).kernel = K;
Sobelkernels(count).name = 'Sobel';
Sobelkernels(count).orientation = 90;

count = count + 1;
K = [1 1 1 1 0; 1 2 2 0 -1; 1 2 0 -2 -1; 1 0 -2 -2 -1; 0 -1 -1 -1 -1];
%K = fspecial("sobel");
%K = imrotate(K,135);
%K = imresize(K,[5,5],'bicubic');
Sobelkernels(count).kernel = K;
Sobelkernels(count).name = 'Sobel';
Sobelkernels(count).orientation = 135;

count = 0;

for k = 1:num_sigma
    s = sigma(k);
    count = count + 1;
    K = fspecial('log',H,s);
    Logkernels(count).kernel = K;
    Logkernels(count).name = 'log';
    Logkernels(count).sigma = s;
end

return
