function kernels = IAS_create_kernels(M)
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

sigma = [0.5:0.5:3];
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

% fspecial kernels
count = count + 1;
K = fspecial("average",H);
kernels(count).kernel = K;
kernels(count).name = 'average';

count = count + 1;
K = [1 1 2 1 1; 1 1 2 1 1; 0 0 0 0 0; -1 -1 -2 -1 -1; -1 -1 -2 -1 -1];
%K = fspecial("sobel");
%K = imresize(K,[5,5],'bicubic');
kernels(count).kernel = K;
kernels(count).name = 'Sobel';
kernels(count).orientation = 0;

count = count + 1;
K = [0 1 1 1 1; -1 0 2 2 1; -1 -2 0 2 1; -1 -2 -2 0 1; -1 -1 -1 -1 0];
%K = fspecial("sobel");
%K = imrotate(K,45);
%K = imresize(K,[5,5],'bicubic');
kernels(count).kernel = K;
kernels(count).name = 'Sobel';
kernels(count).orientation = 45;

count = count + 1;
K = [-1 -1 0 1 1; -1 -1 0 1 1; -2 -2 0 2 2; -1 -1 0 1 1; -1 -1 0 1 1];
%K = fspecial("sobel");
%K = imrotate(K,90);
%K = imresize(K,[5,5],'bicubic');
kernels(count).kernel = K;
kernels(count).name = 'Sobel';
kernels(count).orientation = 90;

count = count + 1;
K = [1 1 1 1 0; 1 2 2 0 -1; 1 2 0 -2 -1; 1 0 -2 -2 -1; 0 -1 -1 -1 -1];
%K = fspecial("sobel");
%K = imrotate(K,135);
%K = imresize(K,[5,5],'bicubic');
kernels(count).kernel = K;
kernels(count).name = 'Sobel';
kernels(count).orientation = 135;

count = count + 1;
K = fspecial("disk",(H-1)/2);
kernels(count).kernel = K;
kernels(count).name = 'disk';

for k = 1:num_sigma
    s = sigma(k);
    count = count + 1;
    K = fspecial('gaussian',H,s);
    kernels(count).kernel = K;
    kernels(count).name = 'gaussian';
    kernels(count).sigma = s;
end

for k = 1:num_sigma
    s = sigma(k);
    count = count + 1;
    K = fspecial('log',H,s);
    kernels(count).kernel = K;
    kernels(count).name = 'log';
    kernels(count).sigma = s;
end

% Fourier Transform kernels

for k1 = 1:num_u
    u1 = u(k1);
    for k2 = 1:num_v
        v1 = v(k2);
        count = count + 1;
        K = CS6640_basis_FT(u1,v1,H,H);
        kernels(count).kernel = real(K);
        kernels(count).name = 'Fourier';
        kernels(count).u = u1;
        kernels(count).v = v1;
    end
end 
return
