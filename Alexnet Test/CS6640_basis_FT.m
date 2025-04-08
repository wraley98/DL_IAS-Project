function W = CS6640_basis_FT(u,v,M,N)
% CS6640_basis_FT - create Fourier basis window
% On input:
%     u (float): frequency in x
%     v (float): frequency in y
%     M (int): number of rows of window
%     N (int): number of cols in window
% On output:
%     W (MxN complex array): basis window; real part is convolution kernel
% Call:
%     W10 = CS6640_basis_FT(1,0,13,13);
% Author:
%     T. Henderson
%     UU
%     Fall 2021
%

W = zeros(M,N);

for x = 0:M-1
    for y = 0:N-1
        W(x+1,y+1) = exp(-2*j*pi*(u*x/M+v*y/N));
    end
end
